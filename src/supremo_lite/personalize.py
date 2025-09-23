"""
Personalized sequence generation for supremo_lite.

This module provides functions for creating personalized genomes by applying
variants to a reference genome and generating sequence windows around variants.
"""

import bisect
import warnings
import os
from typing import Dict, List, Tuple, Union, NamedTuple
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from .variant_utils import read_vcf
from .chromosome_utils import match_chromosomes_with_report, apply_chromosome_mapping
from .sequence_utils import encode_seq
from .core import TORCH_AVAILABLE
from .variant_utils import classify_variant_type, parse_vcf_info

try:
    import torch
except ImportError:
    pass  # Already handled in core


class ChromosomeOffsetTracker:
    """
    Tracks cumulative coordinate offsets per chromosome from applied variants.

    When standard variants (INS/DEL) change chromosome lengths, the original VCF
    coordinates for later BND variants become invalid. This class tracks the
    cumulative offset at each position to enable coordinate transformation.
    """

    def __init__(self):
        """Initialize empty offset tracker."""
        self.chromosome_offsets: Dict[str, List[Tuple[int, int]]] = {}  # chrom -> [(pos, cumulative_offset)]

    def add_offset(self, chrom: str, pos: int, offset: int) -> None:
        """
        Add an offset at a specific position on a chromosome.

        Args:
            chrom: Chromosome name
            pos: Genomic position (0-based) where offset occurs
            offset: Length change (+/- bases) from the variant
        """
        if chrom not in self.chromosome_offsets:
            self.chromosome_offsets[chrom] = []

        # Find insertion point and update cumulative offsets
        offset_list = self.chromosome_offsets[chrom]

        # Calculate cumulative offset at this position
        cumulative_offset = offset
        for existing_pos, existing_cumulative in offset_list:
            if existing_pos <= pos:
                cumulative_offset += existing_cumulative

        # Insert new offset entry, maintaining sorted order by position
        inserted = False
        for i, (existing_pos, existing_cumulative) in enumerate(offset_list):
            if pos < existing_pos:
                offset_list.insert(i, (pos, cumulative_offset))
                inserted = True
                # Update all downstream offsets
                for j in range(i + 1, len(offset_list)):
                    old_pos, old_cumulative = offset_list[j]
                    offset_list[j] = (old_pos, old_cumulative + offset)
                break

        if not inserted:
            offset_list.append((pos, cumulative_offset))

    def get_offset_at_position(self, chrom: str, pos: int) -> int:
        """
        Get the cumulative offset at a specific position.

        Args:
            chrom: Chromosome name
            pos: Genomic position (0-based) to query

        Returns:
            Cumulative offset at this position
        """
        if chrom not in self.chromosome_offsets:
            return 0

        offset_list = self.chromosome_offsets[chrom]
        cumulative_offset = 0

        for offset_pos, offset_cumulative in offset_list:
            if offset_pos <= pos:
                cumulative_offset = offset_cumulative
            else:
                break

        return cumulative_offset

    def transform_coordinate(self, chrom: str, pos: int) -> int:
        """
        Transform a VCF coordinate to account for applied variants.

        Args:
            chrom: Chromosome name
            pos: Original VCF position (1-based)

        Returns:
            Transformed position (1-based) in the modified sequence
        """
        # Convert to 0-based, apply offset, convert back to 1-based
        pos_0based = pos - 1
        offset = self.get_offset_at_position(chrom, pos_0based)
        return pos + offset


class SequenceSegment(NamedTuple):
    """Represents a segment within a sequence with its source and position."""
    source_type: str  # 'reference', 'novel', 'rc_reference'
    source_chrom: str  # chromosome name or 'NOVEL'
    start_pos: int  # start position in the final sequence
    end_pos: int  # end position in the final sequence
    length: int  # segment length
    orientation: str  # 'forward', 'reverse', 'novel'


class ChromosomeSegmentTracker:
    """Track which segments of each chromosome are used by fusions."""

    def __init__(self, ref_sequences):
        self.ref_sequences = ref_sequences
        self.used_segments = {chrom: [] for chrom in ref_sequences.keys()}

    def add_used_segment(self, chrom, start, end):
        """Add a used segment (0-based coordinates)."""
        self.used_segments[chrom].append((start, end))

    def get_leftover_sequences(self):
        """Calculate leftover sequences not used by any fusion."""
        leftover_sequences = {}

        for chrom, ref_seq in self.ref_sequences.items():
            segments = sorted(self.used_segments[chrom])
            leftover_parts = []

            if not segments:
                # No segments used - entire chromosome is leftover
                leftover_parts = [ref_seq]
            else:
                # Find gaps between used segments
                current_pos = 0

                for start, end in segments:
                    # Add leftover before this segment
                    if current_pos < start:
                        leftover_parts.append(ref_seq[current_pos:start])
                    current_pos = max(current_pos, end)

                # Add leftover after last segment
                if current_pos < len(ref_seq):
                    leftover_parts.append(ref_seq[current_pos:])

            # Combine leftover parts
            if leftover_parts:
                leftover_seq = ''.join(leftover_parts)
                if leftover_seq:  # Only add non-empty leftovers
                    leftover_sequences[chrom] = leftover_seq

        return leftover_sequences


class FrozenRegionTracker:
    """
    Efficiently tracks genomic regions that are 'frozen' due to applied variants.

    Frozen regions prevent overlapping variants from being applied to the same
    genomic coordinates. Uses a sorted list of non-overlapping intervals with
    binary search for lookup.
    """

    def __init__(self):
        """Initialize empty interval tracker."""
        self.intervals: List[Tuple[int, int]] = []  # sorted list of (start, end) tuples

    def is_frozen(self, pos: int) -> bool:
        """
        Check if a genomic position is within any frozen region.

        Args:
            pos: Genomic position (0-based)

        Returns:
            True if position is frozen, False otherwise

        """
        if not self.intervals:
            return False

        # Binary search for interval that could contain pos
        idx = bisect.bisect_right(self.intervals, (pos, float("inf"))) - 1

        if idx >= 0:
            start, end = self.intervals[idx]
            return start <= pos <= end

        return False

    def add_range(self, start: int, end: int) -> None:
        """
        Add a new frozen region, merging with existing overlapping intervals.

        Args:
            start: Start position of region (0-based, inclusive)
            end: End position of region (0-based, inclusive)
        """
        if start > end:
            return

        # Find insertion point and overlapping intervals
        left_idx = bisect.bisect_left(self.intervals, (start, start))
        right_idx = bisect.bisect_right(self.intervals, (end, end))

        # Check for overlap with interval before insertion point
        if left_idx > 0:
            prev_start, prev_end = self.intervals[left_idx - 1]
            if prev_end >= start - 1:  # Adjacent or overlapping
                left_idx -= 1
                start = min(start, prev_start)
                end = max(end, prev_end)

        # Merge with all overlapping intervals
        for i in range(left_idx, min(right_idx, len(self.intervals))):
            interval_start, interval_end = self.intervals[i]
            if interval_start <= end + 1:  # Adjacent or overlapping
                start = min(start, interval_start)
                end = max(end, interval_end)

        # Remove old intervals and insert merged interval
        del self.intervals[left_idx:right_idx]
        self.intervals.insert(left_idx, (start, end))


class VariantApplicator:
    """
    Applies VCF variants to a reference sequence in memory.

    Handles coordinate system transformations, frozen region tracking,
    and sequence modifications for SNVs, insertions, and deletions.
    """

    def __init__(self, sequence_str: str, variants_df: pd.DataFrame, frozen_tracker: FrozenRegionTracker = None, offset_tracker: ChromosomeOffsetTracker = None, chrom: str = None):
        """
        Initialize variant applicator for a single chromosome.

        Args:
            sequence_str: Reference sequence as string
            variants_df: DataFrame containing variants for this chromosome
            frozen_tracker: Optional existing FrozenRegionTracker to preserve overlap state across chunks
            offset_tracker: Optional ChromosomeOffsetTracker to track coordinate offsets
            chrom: Chromosome name (required if offset_tracker is provided)
        """
        self.sequence = bytearray(sequence_str.encode())  # Mutable sequence
        self.variants = variants_df.sort_values("pos1").reset_index(drop=True)
        self.frozen_tracker = frozen_tracker if frozen_tracker is not None else FrozenRegionTracker()
        self.offset_tracker = offset_tracker
        self.chrom = chrom
        self.cumulative_offset = 0  # Track length changes from applied variants
        self.applied_count = 0
        self.skipped_count = 0

    def apply_variants(self) -> Tuple[str, Dict[str, int]]:
        """
        Apply all variants to the sequence.

        Returns:
            Tuple of (modified_sequence, statistics_dict)
        """
        for _, variant in self.variants.iterrows():
            try:
                self._apply_single_variant(variant)
            except Exception as e:
                warnings.warn(f"Cannot apply variant at {variant.pos1}: {e}")
                self.skipped_count += 1

        stats = {
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "total": len(self.variants),
        }

        return self.sequence.decode(), stats

    def apply_single_variant_to_window(
        self, variant: pd.Series, window_start: int, window_end: int
    ) -> str:
        """
        Apply a single variant to a sequence window.

        Args:
            variant: Series containing variant information (pos, ref, alt)
            window_start: Start position of window (0-based)
            window_end: End position of window (0-based, exclusive)

        Returns:
            Modified sequence string
        """
        # Create a copy of the window sequence
        window_seq = self.sequence[window_start:window_end].copy()

        # Handle multiple ALT alleles - take first one
        alt_allele = variant.alt.split(",")[0]

        # Calculate variant position relative to window
        genomic_pos = variant.pos1 - 1  # Convert VCF 1-based to 0-based
        var_pos_in_window = genomic_pos - window_start

        # Check if variant is within window
        if var_pos_in_window < 0 or var_pos_in_window >= len(window_seq):
            return window_seq.decode()

        # Check if entire variant fits in window
        ref_end = var_pos_in_window + len(variant.ref)
        if ref_end > len(window_seq):
            return window_seq.decode()

        # Validate reference matches
        expected_ref = window_seq[var_pos_in_window:ref_end].decode()
        if expected_ref.upper() != variant.ref.upper():
            warnings.warn(
                f"Reference mismatch at position {variant.pos1}: "
                f"expected '{variant.ref}', found '{expected_ref}'"
            )
            return window_seq.decode()

        # Apply variant
        if len(alt_allele) == len(variant.ref):
            # SNV: Direct substitution
            window_seq[var_pos_in_window:ref_end] = alt_allele.encode()
        elif len(alt_allele) < len(variant.ref):
            # Deletion
            window_seq[var_pos_in_window : var_pos_in_window + len(alt_allele)] = (
                alt_allele.encode()
            )
            del window_seq[var_pos_in_window + len(alt_allele) : ref_end]
        else:
            # Insertion
            window_seq[var_pos_in_window:ref_end] = alt_allele.encode()

        return window_seq.decode()

    def _apply_single_variant(self, variant: pd.Series) -> None:
        """
        Apply a single variant to the sequence using variant type classifications.

        Args:
            variant: Series containing variant information (pos, ref, alt, variant_type)
        """
        # 1. VARIANT TYPE VALIDATION
        variant_type = variant.get('variant_type', 'unknown')
        
        # Define supported variant types for standard variant processing
        supported_types = {'SNV', 'MNV', 'INS', 'DEL', 'complex'}

        # Handle variants that should be processed elsewhere or are unsupported
        if variant_type == 'SV_BND':
            warnings.warn(
                f"BND variant at position {variant.pos1} reached standard variant processing. "
                f"BND variants are processed separately via breakend pair logic."
            )
            self.skipped_count += 1
            return
        elif variant_type in {'SV_INV', 'SV_DUP', 'SV_DEL', 'SV_INS', 'SV_CNV'}:
            warnings.warn(
                f"Structural variant type '{variant_type}' at position {variant.pos1} is not supported. "
                f"Supported standard variant types: {', '.join(sorted(supported_types))}. "
                f"Note: BND structural variants are supported separately."
            )
            self.skipped_count += 1
            return
        elif variant_type in {'missing', 'unknown'}:
            warnings.warn(
                f"Skipping variant with '{variant_type}' type at position {variant.pos1}. "
                f"Supported types: {', '.join(sorted(supported_types))}, BND (processed separately)"
            )
            self.skipped_count += 1
            return
        elif variant_type not in supported_types:
            warnings.warn(
                f"Skipping unknown variant type '{variant_type}' at position {variant.pos1}. "
                f"Supported types: {', '.join(sorted(supported_types))}, BND (processed separately)"
            )
            self.skipped_count += 1
            return

        # 2. BASIC VALIDATION CHECKS
        if variant.alt == variant.ref:
            self.skipped_count += 1
            return  # Skip ref-only variants

        # Handle multiple ALT alleles - take first one
        alt_allele = variant.alt.split(",")[0]

        # 3. COORDINATE CALCULATION
        genomic_pos = variant.pos1 - 1  # Convert VCF 1-based to 0-based
        buffer_pos = genomic_pos + self.cumulative_offset

        # 4. FROZEN REGION CHECK
        ref_start = genomic_pos
        ref_end = genomic_pos + len(variant.ref) - 1

        if self.frozen_tracker.is_frozen(ref_start) or self.frozen_tracker.is_frozen(
            ref_end
        ):
            self.skipped_count += 1
            return  # Skip overlapping variants

        # 5. BOUNDS CHECK
        if buffer_pos < 0 or buffer_pos + len(variant.ref) > len(self.sequence):
            raise ValueError(f"Variant position {variant.pos1} out of sequence bounds")

        # 6. REFERENCE VALIDATION
        expected_ref = self.sequence[
            buffer_pos : buffer_pos + len(variant.ref)
        ].decode()
        if expected_ref.upper() != variant.ref.upper():
            raise ValueError(
                f"Reference mismatch at position {variant.pos1}: "
                f"expected '{variant.ref}', found '{expected_ref}'"
            )

        # 7. SEQUENCE MODIFICATION
        self._modify_sequence(buffer_pos, variant.ref, alt_allele, variant_type)

        # 8. UPDATE TRACKING
        length_diff = len(alt_allele) - len(variant.ref)
        self.cumulative_offset += length_diff
        self.frozen_tracker.add_range(ref_start, ref_end)

        # Record offset for coordinate transformation if tracker is provided
        if self.offset_tracker and self.chrom and length_diff != 0:
            self.offset_tracker.add_offset(self.chrom, ref_start, length_diff)

        self.applied_count += 1

    def _modify_sequence(self, pos: int, ref_allele: str, alt_allele: str, variant_type: str) -> None:
        """
        Modify sequence at specified position using variant type classification.

        Args:
            pos: Buffer position (0-based)
            ref_allele: Reference allele sequence
            alt_allele: Alternate allele sequence
            variant_type: Classified variant type (SNV, MNV, INS, DEL, complex)
        """
        # Dispatch based on variant type classification
        if variant_type in ['SNV', 'MNV']:
            # Single or multi-nucleotide substitution
            self.sequence[pos : pos + len(ref_allele)] = alt_allele.encode()
            
        elif variant_type == 'INS':
            # Insertion: replace reference with longer alternate sequence
            self.sequence[pos : pos + len(ref_allele)] = alt_allele.encode()
            
        elif variant_type == 'DEL':
            # Deletion: replace reference with shorter alternate sequence
            self.sequence[pos : pos + len(alt_allele)] = alt_allele.encode()
            del self.sequence[pos + len(alt_allele) : pos + len(ref_allele)]
            
        elif variant_type == 'complex':
            # Complex variant: use length-based logic as fallback
            ref_len = len(ref_allele)
            alt_len = len(alt_allele)

            if alt_len == ref_len:
                # Same length substitution
                self.sequence[pos : pos + ref_len] = alt_allele.encode()
            elif alt_len < ref_len:
                # Deletion-like complex variant
                self.sequence[pos : pos + alt_len] = alt_allele.encode()
                del self.sequence[pos + alt_len : pos + ref_len]
            else:
                # Insertion-like complex variant
                self.sequence[pos : pos + ref_len] = alt_allele.encode()
        else:
            # This should not happen due to validation in _apply_single_variant
            raise ValueError(f"Unsupported variant type in sequence modification: {variant_type}")


class ChimericSequenceBuilder:
    """Builds chimeric sequences from BND rearrangements."""

    def __init__(self, reference_sequences: Dict[str, str]):
        self.reference_sequences = reference_sequences
        self.chimeric_sequences = {}
        self.sequence_segments = {}  # Store segment metadata for each sequence

    def create_fusion_from_pair(self, breakend_pair: Tuple) -> Tuple[str, str]:
        """
        Create fusion sequence from a pair of breakends.

        Returns:
            Tuple of (fusion_name, fusion_sequence)
        """
        bnd1, bnd2 = breakend_pair

        # Generate fusion name
        fusion_name = f"{bnd1.chrom}_{bnd2.chrom}_fusion_{bnd1.id}_{bnd2.id}"

        # Get sequences
        seq1 = self.reference_sequences[bnd1.chrom]
        seq2 = self.reference_sequences[bnd2.chrom]

        # Convert VCF 1-based coordinates to 0-based array indices
        pos1_0 = bnd1.pos - 1  # VCF 1-based -> 0-based array index
        pos2_0 = bnd2.pos - 1  # VCF 1-based -> 0-based array index

        # Create fusion based on orientation
        fusion_seq, segments = self._build_oriented_fusion(
            seq1, pos1_0, bnd1.orientation,
            seq2, pos2_0, bnd2.orientation,
            bnd1.inserted_seq + bnd2.inserted_seq,
            bnd1.chrom, bnd2.chrom,
            bnd1, bnd2  # Pass original breakends for VCF positions
        )

        # Store segment metadata
        self.sequence_segments[fusion_name] = segments

        return fusion_name, fusion_seq

    def _build_oriented_fusion(self, seq1: str, pos1_0: int, orient1: str,
                             seq2: str, pos2_0: int, orient2: str,
                             novel_seq: str, seq1_chrom: str, seq2_chrom: str,
                             bnd1, bnd2) -> Tuple[str, List[SequenceSegment]]:
        """
        Build fusion sequence respecting coordinated breakend pair orientations.

        Coordinated BND fusion patterns:
        1. RC Coordination Patterns (require special handling):
           - [p[t + [p[t : RC(seq2[pos2:]) + seq1[pos1:]
           - t]p] + t]p] : seq1[:pos1] + RC(seq2[:pos2])

        2. Same Direction Patterns (simple concatenation):
           - ]p]t + ]p]t : seq2[:pos2] + seq1[pos1:]
           - t[p[ + t[p[ : seq1[:pos1] + seq2[pos2:]

        """
        from .sequence_utils import rc_str as reverse_complement

        # Handle coordinated patterns by looking at both orientations together
        if orient1 == '[p[t' and orient2 == '[p[t':
            # [p[t + [p[t pattern: RC(seq2[pos2_0:]) + seq1[pos1_0:]
            left_part = reverse_complement(seq2[pos2_0:])
            right_part = seq1[pos1_0:]
            left_chrom = seq2_chrom
            right_chrom = seq1_chrom
            left_orientation = 'reverse'
            right_orientation = 'forward'

        elif orient1 == 't]p]' and orient2 == 't]p]':
            # t]p] + t]p] pattern: seq1[:pos1] + RC(seq2[:pos2]) (use VCF positions as base counts)
            left_part = seq1[:bnd1.pos]  # Include pos1 bases from seq1
            right_part = reverse_complement(seq2[:bnd2.pos])  # Include pos2 bases from seq2
            left_chrom = seq1_chrom
            right_chrom = seq2_chrom
            left_orientation = 'forward'
            right_orientation = 'reverse'

        elif orient1 == ']p]t' and orient2 == ']p]t':
            # ]p]t + ]p]t pattern: seq2[:pos2] + seq1[pos1_0:] (use VCF pos2 as base count)
            left_part = seq2[:bnd2.pos]  # Include pos2 bases from seq2
            right_part = seq1[pos1_0:]   # From pos1_0 to end of seq1
            left_chrom = seq2_chrom
            right_chrom = seq1_chrom
            left_orientation = 'forward'
            right_orientation = 'forward'

        elif orient1 == 't[p[' and orient2 == 't[p[':
            # t[p[ + t[p[ pattern: seq1[:pos1_0] + seq2[pos2_0:]
            left_part = seq1[:pos1_0]
            right_part = seq2[pos2_0:]
            left_chrom = seq1_chrom
            right_chrom = seq2_chrom
            left_orientation = 'forward'
            right_orientation = 'forward'

        elif orient1 == ']p]t' and orient2 == 't[p[':
            # ]p]t + t[p[ pattern: seq2[:pos2] + seq1[pos1_0:] (use VCF pos2 as base count)
            left_part = seq2[:bnd2.pos]  # Include pos2 bases from seq2
            right_part = seq1[pos1_0:]   # From pos1_0 to end of seq1
            left_chrom = seq2_chrom
            right_chrom = seq1_chrom
            left_orientation = 'forward'
            right_orientation = 'forward'

        elif orient1 == 't[p[' and orient2 == ']p]t':
            # t[p[ + ]p]t pattern: seq1[:pos1_0] + seq2[pos2_0:]
            left_part = seq1[:pos1_0]
            right_part = seq2[pos2_0:]
            left_chrom = seq1_chrom
            right_chrom = seq2_chrom
            left_orientation = 'forward'
            right_orientation = 'forward'

        elif orient1 == 't]p]' and orient2 == ']p]t':
            # t]p] + ]p]t pattern (mixed coordination): seq1[:pos1] + seq2[pos2_0:]
            left_part = seq1[:bnd1.pos]  # Include pos1 bases from seq1 (VCF position)
            right_part = seq2[pos2_0:]   # From pos2_0 to end of seq2
            left_chrom = seq1_chrom
            right_chrom = seq2_chrom
            left_orientation = 'forward'
            right_orientation = 'forward'

        else:
            # Unknown orientation pattern - fail fast to ensure proper implementation
            supported_patterns = [
                '[p[t + [p[t (RC coordination)',
                't]p] + t]p] (RC coordination)',
                ']p]t + ]p]t (same direction)',
                't[p[ + t[p[ (same direction)',
                't]p] + ]p]t (mixed coordination)'
            ]
            raise ValueError(
                f"Unsupported BND orientation pattern: '{orient1}' + '{orient2}'. "
                f"Supported patterns: {', '.join(supported_patterns)}. "
                f"This pattern requires explicit implementation."
            )

        # Build fusion sequence and track segments
        segments = []
        current_pos = 0

        # Add left segment
        if len(left_part) > 0:
            left_type = 'rc_reference' if left_orientation == 'reverse' else 'reference'
            segments.append(SequenceSegment(
                source_type=left_type,
                source_chrom=left_chrom,
                start_pos=current_pos,
                end_pos=current_pos + len(left_part),
                length=len(left_part),
                orientation=left_orientation
            ))
            current_pos += len(left_part)

        # Add novel sequence segment
        if len(novel_seq) > 0:
            segments.append(SequenceSegment(
                source_type='novel',
                source_chrom='NOVEL',
                start_pos=current_pos,
                end_pos=current_pos + len(novel_seq),
                length=len(novel_seq),
                orientation='novel'
            ))
            current_pos += len(novel_seq)

        # Add right segment
        if len(right_part) > 0:
            right_type = 'rc_reference' if right_orientation == 'reverse' else 'reference'
            segments.append(SequenceSegment(
                source_type=right_type,
                source_chrom=right_chrom,
                start_pos=current_pos,
                end_pos=current_pos + len(right_part),
                length=len(right_part),
                orientation=right_orientation
            ))

        # Combine parts to create fusion sequence
        fusion = left_part + novel_seq + right_part

        return fusion, segments

    def get_sequence_segments(self, sequence_name: str) -> List[SequenceSegment]:
        """Get segment metadata for a sequence."""
        return self.sequence_segments.get(sequence_name, [])


def _load_reference(reference_fn: Union[str, Dict, Fasta]) -> Union[Dict, Fasta]:
    """Load reference genome from file or return as-is if already loaded."""
    if isinstance(reference_fn, str) and os.path.isfile(reference_fn):
        return Fasta(reference_fn)
    return reference_fn




def _load_variants(variants_fn: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load variants from file or return as-is if already a DataFrame.
    Ensures variant classification happens once during loading.

    For DataFrames, assumes position column is either 'pos', 'pos1', or the second column.
    If DataFrame lacks variant_type column, classification will be added.
    """
    if isinstance(variants_fn, str):
        # Always load all variants with classification
        variants_df = read_vcf(variants_fn, classify_variants=True)
        # Rename pos to pos1 for consistency

        return variants_df
    else:
        # Handle DataFrame input
        variants_df = variants_fn.copy()

        # Always use second column as pos1, regardless of current name
        if len(variants_df.columns) >= 2:
            # Rename second column to pos1 if it's not already named that
            if variants_df.columns[1] != "pos1":
                new_columns = list(variants_df.columns)
                new_columns[1] = "pos1"
                variants_df.columns = new_columns

            # Validate that pos1 column is numeric
            if not pd.api.types.is_numeric_dtype(variants_df["pos1"]):
                raise ValueError(
                    f"Position column (second column) must be numeric, got {variants_df['pos1'].dtype}"
                )
        else:
            raise ValueError(
                "DataFrame must have at least 2 columns with position in second column"
            )

        # Ensure variant classification exists
        if 'variant_type' not in variants_df.columns:
            if len(variants_df) > 0:
                # Add variant classification to non-empty DataFrame
                variants_df['variant_type'] = variants_df.apply(
                    lambda row: classify_variant_type(
                        row['ref'], 
                        row['alt'], 
                        parse_vcf_info(row.get('info', '')) if 'info' in row else None
                    ), 
                    axis=1
                )
            else:
                # Handle empty DataFrame - just add empty column
                variants_df['variant_type'] = pd.Series(dtype='object')

        return variants_df


def get_personal_genome(reference_fn, variants_fn, encode=True, n_chunks=1, verbose=False, encoder=None):
    """
    Create a personalized genome by applying variants to a reference genome.

    This function supports all variant types including standard variants (SNV, INS, DEL)
    and complex structural variants (BND breakends). BND variants are applied after
    standard variants to create novel adjacencies between genomic positions.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame. Supports VCF 4.2 format
                    including BND (breakend) variants with bracket notation.
        encode: Return sequences as one-hot encoded arrays (default: True)
        n_chunks: Number of chunks to split variants into for processing (default: 1)
        verbose: Print progress information (default: False)
        encoder: Optional custom encoding function. If provided, should accept a single
                sequence string and return encoded array with shape (L, 4). Default: None

    Returns:
        If encode=True: A dictionary mapping chromosome names to encoded tensors/arrays
        If encode=False: A dictionary mapping chromosome names to sequence strings

    Notes:
        - Standard variants (SNV, INS, DEL) are applied first using VariantApplicator
        - BND variants are then applied using BreakendApplicator to create novel adjacencies
        - BND variants are paired automatically based on coordinate matching
        - MATEID field is optional for BND variants
        - Supports complex rearrangements like translocations, inversions, duplications

    Examples:
        # Apply standard and BND variants
        personal_genome = get_personal_genome('reference.fa', 'variants.vcf')

        # Get raw sequences without encoding
        personal_genome = get_personal_genome('reference.fa', 'variants.vcf', encode=False)
    """
    # Load variants and separate BND variants
    from .variant_utils import load_breakend_variants

    standard_variants, breakend_pairs = load_breakend_variants(variants_fn)
    reference = _load_reference(reference_fn)

    # Sort standard variants by chromosome and position for efficient processing
    if len(standard_variants) > 0:
        standard_variants = standard_variants.sort_values(["chrom", "pos1"])

    # Apply chromosome name matching once
    ref_chroms = set(reference.keys())
    # Combine chromosomes from both standard variants and breakend pairs
    standard_chroms = set(standard_variants["chrom"].unique()) if len(standard_variants) > 0 else set()
    breakend_chroms = set()
    for pair in breakend_pairs:
        breakend_chroms.add(pair[0].chrom)
        breakend_chroms.add(pair[1].chrom)
    vcf_chroms = standard_chroms | breakend_chroms

    total_variants = len(standard_variants) + len(breakend_pairs) * 2  # Each pair represents 2 variants
    if verbose:
        print(f"🧬 Processing {total_variants:,} variants ({len(standard_variants)} standard, {len(breakend_pairs)} BND pairs) across {len(vcf_chroms)} chromosomes")

    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=verbose
    )

    # Apply chromosome name mapping to all variants
    if mapping:
        if len(standard_variants) > 0:
            standard_variants = apply_chromosome_mapping(standard_variants, mapping)
        # Apply mapping to breakend pairs
        for pair in breakend_pairs:
            if pair[0].chrom in mapping:
                pair[0].chrom = mapping[pair[0].chrom]
            if pair[1].chrom in mapping:
                pair[1].chrom = mapping[pair[1].chrom]

    # Initialize personalized genome
    personal_genome = {}
    total_processed = 0

    # Initialize coordinate offset tracker for BND coordinate transformation
    offset_tracker = ChromosomeOffsetTracker()
    modified_sequences = {}

    # Step 1: Apply standard variants to each chromosome with offset tracking
    if len(standard_variants) > 0:
        for chrom, chrom_variants in standard_variants.groupby("chrom"):
            if chrom not in reference:
                if verbose:
                    print(f"⚠️  Skipping {chrom}: not found in reference")
                continue

            ref_seq = str(reference[chrom])

            if verbose:
                print(f"🔄 Processing chromosome {chrom}: {len(chrom_variants):,} standard variants")

            if n_chunks == 1:
                # Process all variants at once with offset tracking
                applicator = VariantApplicator(ref_seq, chrom_variants,
                                             offset_tracker=offset_tracker, chrom=chrom)
                personal_seq, stats = applicator.apply_variants()

                if verbose and stats["total"] > 0:
                    print(f"  ✅ Applied {stats['applied']}/{stats['total']} variants ({stats['skipped']} skipped)")

            else:
                # Process in chunks with shared FrozenRegionTracker and offset tracking
                current_sequence = ref_seq
                shared_frozen_tracker = FrozenRegionTracker()
                total_applied = 0
                total_skipped = 0

                # Split chromosome variants into n_chunks
                indices = np.array_split(np.arange(len(chrom_variants)), n_chunks)

                if verbose:
                    avg_chunk_size = len(chrom_variants) // n_chunks
                    print(f"  📦 Processing {n_chunks} chunks of ~{avg_chunk_size:,} variants each")

                for i, chunk_indices in enumerate(indices):
                    if len(chunk_indices) == 0:
                        continue

                    chunk_df = chrom_variants.iloc[chunk_indices].reset_index(drop=True)

                    # Apply variants with shared FrozenRegionTracker and offset tracking
                    applicator = VariantApplicator(current_sequence, chunk_df,
                                                 shared_frozen_tracker, offset_tracker, chrom)
                    current_sequence, stats = applicator.apply_variants()

                    total_applied += stats["applied"]
                    total_skipped += stats["skipped"]

                    if verbose:
                        print(f"    ✅ Chunk {i+1}: {stats['applied']}/{stats['total']} variants applied")

                personal_seq = current_sequence

                if verbose:
                    print(f"  🎯 Total: {total_applied}/{len(chrom_variants)} variants applied ({total_skipped} skipped)")

            # Store modified sequence for this chromosome
            modified_sequences[chrom] = personal_seq
            total_processed += len(chrom_variants)

    # Step 2: Initialize sequences for chromosomes without standard variants
    for chrom in reference.keys():
        if chrom not in modified_sequences:
            modified_sequences[chrom] = str(reference[chrom])

    # Step 3: Transform BND coordinates using accumulated offsets and apply BND pairs
    if len(breakend_pairs) > 0:
        if verbose:
            print(f"🔄 Processing {len(breakend_pairs)} BND pairs with coordinate transformation")

        # Transform BND coordinates based on offset tracker
        for pair in breakend_pairs:
            bnd1, bnd2 = pair
            # Transform coordinates to account for applied standard variants
            original_pos1 = bnd1.pos
            original_pos2 = bnd2.pos
            bnd1.pos = offset_tracker.transform_coordinate(bnd1.chrom, bnd1.pos)
            bnd2.pos = offset_tracker.transform_coordinate(bnd2.chrom, bnd2.pos)

            if verbose and (bnd1.pos != original_pos1 or bnd2.pos != original_pos2):
                print(f"  📍 Transformed BND coordinates: {bnd1.chrom}:{original_pos1}→{bnd1.pos}, {bnd2.chrom}:{original_pos2}→{bnd2.pos}")

        # Create chimeric sequences using ChimericSequenceBuilder
        sequence_builder = ChimericSequenceBuilder(modified_sequences)
        # Initialize segment tracker with only the original reference chromosomes
        original_ref_sequences = {chrom: modified_sequences[chrom] for chrom in reference.keys()}
        segment_tracker = ChromosomeSegmentTracker(original_ref_sequences)

        for i, pair in enumerate(breakend_pairs):
            if verbose:
                print(f"  🔄 Creating fusion {i+1}/{len(breakend_pairs)}: {pair[0].id}-{pair[1].id}")

            try:
                fusion_name, fusion_seq = sequence_builder.create_fusion_from_pair(pair)
                modified_sequences[fusion_name] = fusion_seq

                # Track chromosome segment usage based on fusion orientations
                bnd1, bnd2 = pair
                pos1_0 = bnd1.pos - 1  # Convert to 0-based
                pos2_0 = bnd2.pos - 1  # Convert to 0-based

                # Track segments used based on the actual fusion logic
                if bnd1.orientation == 't]p]' and bnd2.orientation == 't]p]':
                    # seq1[:pos1] + RC(seq2[:pos2])
                    segment_tracker.add_used_segment(bnd1.chrom, 0, bnd1.pos)  # VCF pos as count
                    segment_tracker.add_used_segment(bnd2.chrom, 0, bnd2.pos)  # VCF pos as count
                elif bnd1.orientation == ']p]t' and bnd2.orientation == 't[p[':
                    # seq2[:pos2] + seq1[pos1_0:]
                    segment_tracker.add_used_segment(bnd2.chrom, 0, bnd2.pos)  # VCF pos as count
                    segment_tracker.add_used_segment(bnd1.chrom, pos1_0, len(modified_sequences[bnd1.chrom]))
                elif bnd1.orientation == '[p[t' and bnd2.orientation == '[p[t':
                    # RC(seq2[pos2_0:]) + seq1[pos1_0:]
                    segment_tracker.add_used_segment(bnd2.chrom, pos2_0, len(modified_sequences[bnd2.chrom]))
                    segment_tracker.add_used_segment(bnd1.chrom, pos1_0, len(modified_sequences[bnd1.chrom]))
                elif bnd1.orientation == 't[p[' and bnd2.orientation == 't[p[':
                    # seq1[:pos1_0] + seq2[pos2_0:]
                    segment_tracker.add_used_segment(bnd1.chrom, 0, pos1_0)
                    segment_tracker.add_used_segment(bnd2.chrom, pos2_0, len(modified_sequences[bnd2.chrom]))
                elif bnd1.orientation == 't[p[' and bnd2.orientation == ']p]t':
                    # seq1[:pos1_0] + seq2[pos2_0:]
                    segment_tracker.add_used_segment(bnd1.chrom, 0, pos1_0)
                    segment_tracker.add_used_segment(bnd2.chrom, pos2_0, len(modified_sequences[bnd2.chrom]))

                if verbose:
                    print(f"    ✅ Created fusion: {fusion_name} ({len(fusion_seq)} bp)")

            except Exception as e:
                if verbose:
                    print(f"    ⚠️ Failed to create fusion for {pair[0].id}-{pair[1].id}: {e}")

        # Calculate and add leftover sequences
        leftover_sequences = segment_tracker.get_leftover_sequences()

        # Build final sequences dict with only fusions and leftovers
        final_sequences = {}

        # Add all fusion sequences (those created by BND processing)
        for name, seq in modified_sequences.items():
            if '_fusion_' in name:
                final_sequences[name] = seq

        # Add leftover sequences from original chromosomes
        final_sequences.update(leftover_sequences)

        # Replace modified_sequences with final sequences for BND case
        modified_sequences = final_sequences

        if verbose:
            print(f"  🎯 BND Summary: Created {len(breakend_pairs)} fusions, {len(leftover_sequences)} leftover sequences")

        total_processed += len(breakend_pairs) * 2

    # Step 4: Encode final sequences and store in personal_genome
    for chrom, seq in modified_sequences.items():
        personal_genome[chrom] = encode_seq(seq, encoder) if encode else seq

    if verbose:
        print(f"🎉 Complete! {len(personal_genome)} chromosomes, {total_processed:,} variants processed")

    return personal_genome


def _generate_sequence_metadata(chunk_variants, seq_len):
    """
    Generate standardized metadata for sequence functions.
    
    This centralizes metadata generation to eliminate duplication across
    get_alt_sequences, get_ref_sequences, and get_alt_ref_sequences.
    
    Args:
        chunk_variants: DataFrame of variants for this chunk
        seq_len: Length of the sequence window
        
    Returns:
        pandas.DataFrame: Comprehensive metadata with standardized columns
    """
    metadata = []
    
    for _, var in chunk_variants.iterrows():
        # Basic position calculations
        pos = var["pos1"]  # 1-based VCF position
        genomic_pos = pos - 1  # Convert to 0-based
        half_len = seq_len // 2
        window_start = max(0, genomic_pos - half_len)
        window_end = window_start + seq_len
        
        # Variant classification and length calculations
        variant_type = var.get('variant_type', 'unknown')
        ref_length = len(var["ref"])
        alt_length = len(var["alt"])
        
        # Calculate effective variant boundaries for complex variants
        if variant_type in ['SV_INV', 'SV_DUP', 'SV_BND']:
            # For structural variants, use full reported range
            effective_start = genomic_pos
            effective_end = genomic_pos + ref_length - 1  # 0-based end
            variant_info = var.get('variant_info', {})
        else:
            # For simple variants, use standard boundaries
            effective_start = genomic_pos  
            effective_end = genomic_pos + ref_length - 1  # 0-based end
            variant_info = None
            
        # Calculate position offsets
        upstream_offset = 0  # Positions before variant unaffected
        downstream_offset = alt_length - ref_length  # Net length change
        
        # Position within sequence window (0-based)
        variant_offset = genomic_pos - window_start
        
        # Extract SVTYPE and SVLEN from INFO field if available
        svtype = None
        svlen = None
        if 'info' in var and var['info'] and var['info'] != '.':
            parsed_info = parse_vcf_info(var['info'])
            svtype = parsed_info.get('SVTYPE')
            svlen = parsed_info.get('SVLEN')
        
        metadata.append({
            "chrom": var["chrom"],
            "start": window_start,
            "end": window_end,
            "variant_pos0": genomic_pos,  # 0-based absolute position
            "variant_pos1": pos,  # 1-based absolute position  
            "variant_offset": variant_offset,  # 0-based position within window
            "ref": var["ref"],
            "alt": var["alt"],
            # Enhanced variant type information
            "variant_type": variant_type,
            "ref_length": ref_length,
            "alt_length": alt_length,
            "effective_variant_start": effective_start - window_start,  # Relative to window
            "effective_variant_end": effective_end - window_start,      # Relative to window
            "position_offset_upstream": upstream_offset,
            "position_offset_downstream": downstream_offset,
            "structural_variant_info": variant_info,
            # Extract specific INFO fields for easier access
            "svtype": svtype,
            "svlen": svlen,
        })
    
    return pd.DataFrame(metadata)


def _generate_bnd_ref_sequences(breakend_pairs, reference, seq_len, encode=True, encoder=None):
    """
    Generate dual reference sequences for BND variants (no ALT sequences).

    Args:
        breakend_pairs: List of breakend pairs from load_breakend_variants
        reference: Reference genome dictionary
        seq_len: Length of sequence window
        encode: Whether to encode sequences
        encoder: Optional custom encoder

    Returns:
        Tuple of (left_ref_sequences, right_ref_sequences, metadata)
        left_ref_sequences contains left breakend reference (+ N-padding)
        right_ref_sequences contains right breakend reference (N-padding +)
    """
    from .sequence_utils import encode_seq, rc_str

    left_ref_sequences = []
    right_ref_sequences = []
    metadata = []

    if not breakend_pairs:
        return left_ref_sequences, right_ref_sequences, metadata

    # Process each breakend pair
    for bnd1, bnd2 in breakend_pairs:
        try:
            # Get chromosome sequences
            seq1 = str(reference[bnd1.chrom]) if bnd1.chrom in reference else ""
            seq2 = str(reference[bnd2.chrom]) if bnd2.chrom in reference else ""

            if not seq1 or not seq2:
                continue

            # Calculate window centered on first breakend
            center_pos = bnd1.pos - 1  # Convert to 0-based

            # Generate left reference sequence (sequence before breakend + right-side N-padding)
            # For BNDs, we want to show what was there BEFORE the fusion point
            # Then pad the right side with N's to represent the missing fusion partner
            half_len = seq_len // 2

            # Extract sequence leading up to the breakend (before the fusion point)
            left_start = max(0, center_pos - half_len)
            left_end = center_pos  # Stop at the breakend position
            left_ref_raw = seq1[left_start:left_end]

            # Pad the right side to represent where the fusion partner would attach
            left_padding_needed = seq_len - len(left_ref_raw)
            left_ref_seq = left_ref_raw + 'N' * left_padding_needed

            # Generate right reference sequence (left-side N-padding + sequence after breakend)
            # For the right side, we want to show what was there AFTER the fusion point
            # Pad the left side with N's to represent the missing fusion partner
            bnd2_center = bnd2.pos - 1  # Convert to 0-based

            # Extract sequence starting from the breakend (after the fusion point)
            right_start = bnd2_center  # Start at the breakend position
            right_end = min(len(seq2), bnd2_center + half_len)
            right_ref_raw = seq2[right_start:right_end]

            # Pad the left side to represent where the fusion partner would attach
            right_padding_needed = seq_len - len(right_ref_raw)
            right_ref_seq = 'N' * right_padding_needed + right_ref_raw

            # Apply reverse complement if needed based on orientation
            if bnd1.orientation in ['t]p]', '[p[t']:  # orientations requiring RC
                left_ref_seq = rc_str(left_ref_seq)
            if bnd2.orientation in ['t]p]', '[p[t']:
                right_ref_seq = rc_str(right_ref_seq)

            # Ensure sequences are exactly seq_len
            left_ref_seq = left_ref_seq[:seq_len].ljust(seq_len, 'N')
            right_ref_seq = right_ref_seq[:seq_len].ljust(seq_len, 'N')

            left_ref_sequences.append(left_ref_seq)
            right_ref_sequences.append(right_ref_seq)

            # Create metadata for this BND
            window_start = max(0, center_pos - seq_len // 2)
            window_end = window_start + seq_len
            metadata.append({
                'chrom': bnd1.chrom,
                'start': window_start,
                'end': window_end,
                'variant_pos0': center_pos,
                'variant_pos1': bnd1.pos,
                'variant_offset': seq_len // 2,
                'ref': bnd1.ref,
                'alt': bnd1.alt,
                'variant_type': 'SV_BND',
                'mate_chrom': bnd2.chrom,
                'mate_pos': bnd2.pos,
                'orientation_1': bnd1.orientation,
                'orientation_2': bnd2.orientation,
            })

        except Exception as e:
            # Log error but continue processing other BNDs
            import warnings
            warnings.warn(f"Failed to process BND pair {bnd1.id}-{bnd2.id}: {e}")
            continue

    # Encode sequences if requested
    if encode and left_ref_sequences:
        # Encode each sequence individually and collect them
        encoded_left_ref = []
        encoded_right_ref = []

        for i in range(len(left_ref_sequences)):
            encoded_left_ref.append(encode_seq(left_ref_sequences[i], encoder))
            encoded_right_ref.append(encode_seq(right_ref_sequences[i], encoder))

        # Stack the encoded sequences
        if TORCH_AVAILABLE:
            left_ref_sequences = torch.stack(encoded_left_ref) if encoded_left_ref else []
            right_ref_sequences = torch.stack(encoded_right_ref) if encoded_right_ref else []
        else:
            left_ref_sequences = np.stack(encoded_left_ref) if encoded_left_ref else []
            right_ref_sequences = np.stack(encoded_right_ref) if encoded_right_ref else []

    return left_ref_sequences, right_ref_sequences, metadata


def _generate_bnd_sequences(breakend_pairs, reference, seq_len, encode=True, encoder=None):
    """
    Generate ALT and reference sequences for BND variants.

    Args:
        breakend_pairs: List of breakend pairs from load_breakend_variants
        reference: Reference genome dictionary
        seq_len: Length of sequence window
        encode: Whether to encode sequences
        encoder: Optional custom encoder

    Returns:
        Tuple of (alt_sequences, left_ref_sequences, right_ref_sequences, metadata)
        For BNDs: alt_sequences contains fusion sequences
                 left_ref_sequences contains left breakend reference (+ N-padding)
                 right_ref_sequences contains right breakend reference (N-padding +)
    """
    from .sequence_utils import encode_seq, rc_str

    alt_sequences = []
    left_ref_sequences = []
    right_ref_sequences = []
    metadata = []

    if not breakend_pairs:
        return alt_sequences, left_ref_sequences, right_ref_sequences, metadata

    # Process each breakend pair
    for bnd1, bnd2 in breakend_pairs:
        try:
            # Get chromosome sequences
            seq1 = str(reference[bnd1.chrom]) if bnd1.chrom in reference else ""
            seq2 = str(reference[bnd2.chrom]) if bnd2.chrom in reference else ""

            if not seq1 or not seq2:
                continue

            # Generate fusion sequence using existing ChimericSequenceBuilder
            builder = ChimericSequenceBuilder({bnd1.chrom: seq1, bnd2.chrom: seq2})
            fusion_name, fusion_seq = builder.create_fusion_from_pair((bnd1, bnd2))

            # Calculate window centered on first breakend
            center_pos = bnd1.pos - 1  # Convert to 0-based
            window_start = max(0, center_pos - seq_len // 2)
            window_end = window_start + seq_len

            # Generate ALT sequence (fusion sequence window)
            if len(fusion_seq) >= seq_len:
                alt_seq = fusion_seq[window_start:window_end]
            else:
                # Pad if fusion is shorter than window
                alt_seq = fusion_seq + 'N' * (seq_len - len(fusion_seq))

            # Generate left reference sequence (sequence before breakend + right-side N-padding)
            # For BNDs, we want to show what was there BEFORE the fusion point
            # Then pad the right side with N's to represent the missing fusion partner
            half_len = seq_len // 2

            # Extract sequence leading up to the breakend (before the fusion point)
            left_start = max(0, center_pos - half_len)
            left_end = center_pos  # Stop at the breakend position
            left_ref_raw = seq1[left_start:left_end]

            # Pad the right side to represent where the fusion partner would attach
            left_padding_needed = seq_len - len(left_ref_raw)
            left_ref_seq = left_ref_raw + 'N' * left_padding_needed

            # Generate right reference sequence (left-side N-padding + sequence after breakend)
            # For the right side, we want to show what was there AFTER the fusion point
            # Pad the left side with N's to represent the missing fusion partner
            bnd2_center = bnd2.pos - 1  # Convert to 0-based

            # Extract sequence starting from the breakend (after the fusion point)
            right_start = bnd2_center  # Start at the breakend position
            right_end = min(len(seq2), bnd2_center + half_len)
            right_ref_raw = seq2[right_start:right_end]

            # Pad the left side to represent where the fusion partner would attach
            right_padding_needed = seq_len - len(right_ref_raw)
            right_ref_seq = 'N' * right_padding_needed + right_ref_raw

            # Apply reverse complement if needed based on orientation
            if bnd1.orientation in ['t]p]', '[p[t']:  # orientations requiring RC
                left_ref_seq = rc_str(left_ref_seq)
            if bnd2.orientation in ['t]p]', '[p[t']:
                right_ref_seq = rc_str(right_ref_seq)

            # Ensure sequences are exactly seq_len
            alt_seq = alt_seq[:seq_len].ljust(seq_len, 'N')
            left_ref_seq = left_ref_seq[:seq_len].ljust(seq_len, 'N')
            right_ref_seq = right_ref_seq[:seq_len].ljust(seq_len, 'N')

            alt_sequences.append(alt_seq)
            left_ref_sequences.append(left_ref_seq)
            right_ref_sequences.append(right_ref_seq)

            # Create metadata for this BND
            metadata.append({
                'chrom': bnd1.chrom,
                'start': window_start,
                'end': window_end,
                'variant_pos0': center_pos,
                'variant_pos1': bnd1.pos,
                'variant_offset': seq_len // 2,
                'ref': bnd1.ref,
                'alt': bnd1.alt,
                'variant_type': 'SV_BND',
                'mate_chrom': bnd2.chrom,
                'mate_pos': bnd2.pos,
                'orientation_1': bnd1.orientation,
                'orientation_2': bnd2.orientation,
                'fusion_name': fusion_name
            })

        except Exception as e:
            # Log error but continue processing other BNDs
            import warnings
            warnings.warn(f"Failed to process BND pair {bnd1.id}-{bnd2.id}: {e}")
            continue

    # Encode sequences if requested
    if encode and alt_sequences:
        # Encode each sequence individually and collect them
        encoded_alt = []
        encoded_left_ref = []
        encoded_right_ref = []

        for i in range(len(alt_sequences)):
            encoded_alt.append(encode_seq(alt_sequences[i], encoder))
            encoded_left_ref.append(encode_seq(left_ref_sequences[i], encoder))
            encoded_right_ref.append(encode_seq(right_ref_sequences[i], encoder))

        # Stack the encoded sequences
        if TORCH_AVAILABLE:
            alt_sequences = torch.stack(encoded_alt) if encoded_alt else []
            left_ref_sequences = torch.stack(encoded_left_ref) if encoded_left_ref else []
            right_ref_sequences = torch.stack(encoded_right_ref) if encoded_right_ref else []
        else:
            alt_sequences = np.stack(encoded_alt) if encoded_alt else []
            left_ref_sequences = np.stack(encoded_left_ref) if encoded_left_ref else []
            right_ref_sequences = np.stack(encoded_right_ref) if encoded_right_ref else []

    return alt_sequences, left_ref_sequences, right_ref_sequences, metadata


def get_alt_sequences(reference_fn, variants_fn, seq_len, encode=True, n_chunks=1, encoder=None):
    """
    Create sequence windows centered on each variant position with variants applied.
    Now supports both standard variants and BND variants.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to VCF file (string) or DataFrame with variant data.
                    For DataFrames, position column can be 'pos', 'pos1', or assumes second column is position.
        seq_len: Length of the sequence window
        encode: Return sequences as one-hot encoded numpy arrays (default: True)
        n_chunks: Number of chunks to split variants into (default: 1)

    Yields:
        Tuple containing (sequences, metadata_df) where:
            If encode=True: sequences is a tensor/array of shape (chunk_size, seq_len, 4) for each chunk
            If encode=False: sequences is a list of tuples containing (chrom, start, end, sequence_string) for each chunk
            metadata_df is a DataFrame with variant information including position offsets
            For BND variants: sequences contain fusion sequences
    """
    # Load reference and variants, separating BNDs from standard variants
    from .variant_utils import load_breakend_variants
    reference = _load_reference(reference_fn)

    # Load variants and separate BNDs
    standard_variants, breakend_pairs = load_breakend_variants(variants_fn)

    # Combine chromosome names from both standard variants and breakend pairs
    ref_chroms = set(reference.keys())
    standard_chroms = set(standard_variants["chrom"].unique()) if len(standard_variants) > 0 else set()
    breakend_chroms = set()
    for bnd1, bnd2 in breakend_pairs:
        breakend_chroms.add(bnd1.chrom)
        breakend_chroms.add(bnd2.chrom)
    vcf_chroms = standard_chroms | breakend_chroms

    # Use chromosome matching to handle name mismatches
    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=True
    )

    # Apply chromosome name mapping to standard variants
    if mapping and len(standard_variants) > 0:
        standard_variants = apply_chromosome_mapping(standard_variants, mapping)

    # Apply chromosome name mapping to breakend pairs
    if mapping and breakend_pairs:
        updated_pairs = []
        for bnd1, bnd2 in breakend_pairs:
            # Update chromosome names in breakend objects if needed
            if bnd1.chrom in mapping:
                bnd1.chrom = mapping[bnd1.chrom]
            if bnd2.chrom in mapping:
                bnd2.chrom = mapping[bnd2.chrom]
            if bnd1.mate_chrom in mapping:
                bnd1.mate_chrom = mapping[bnd1.mate_chrom]
            if bnd2.mate_chrom in mapping:
                bnd2.mate_chrom = mapping[bnd2.mate_chrom]
            updated_pairs.append((bnd1, bnd2))
        breakend_pairs = updated_pairs

    # Process standard variants and BNDs separately, then combine results
    # For now, we'll process all in one chunk (BND chunking is more complex)

    # Process standard variants first - yield each chunk individually
    if len(standard_variants) > 0:
        # Split standard variants into chunks
        std_indices = np.array_split(np.arange(len(standard_variants)), n_chunks)
        std_variant_chunks = (
            standard_variants.iloc[chunk_indices].reset_index(drop=True)
            for chunk_indices in std_indices
            if len(chunk_indices) > 0
        )

        for chunk_variants in std_variant_chunks:
            sequences = []
            # Generate standardized metadata using shared function
            metadata_df = _generate_sequence_metadata(chunk_variants, seq_len)

            # Group variants by chromosome for efficient processing
            for chrom, chrom_variants in chunk_variants.groupby("chrom"):
                if chrom not in reference:
                    warnings.warn(
                        f"Chromosome {chrom} not found in reference. Skipping {len(chrom_variants)} variants."
                    )
                    continue

                ref_seq = str(reference[chrom])
                chrom_length = len(ref_seq)

                # Apply all variants for this chromosome at once
                applicator = VariantApplicator(ref_seq, chrom_variants)
                modified_chrom, stats = applicator.apply_variants()

                # Vectorized calculation of window positions
                positions = chrom_variants["pos1"].values - 1  # Convert to 0-based
                half_len = seq_len // 2
                window_starts = positions - half_len
                window_ends = window_starts + seq_len

                # Process all variants in this chromosome using NumPy operations
                for idx, (_, var) in enumerate(chrom_variants.iterrows()):
                    pos = var["pos1"]
                    genomic_pos = positions[idx]
                    window_start = window_starts[idx]
                    window_end = window_ends[idx]

                    # Handle edge cases and extract window
                    if window_start < 0:
                        left_pad = -window_start
                        actual_start = 0
                    else:
                        left_pad = 0
                        actual_start = window_start

                    if window_end > len(modified_chrom):
                        right_pad = window_end - len(modified_chrom)
                        actual_end = len(modified_chrom)
                    else:
                        right_pad = 0
                        actual_end = window_end

                    # Extract window from modified chromosome
                    window_seq = modified_chrom[actual_start:actual_end]

                    # Add padding if needed
                    if left_pad > 0:
                        window_seq = "N" * left_pad + window_seq
                    if right_pad > 0:
                        window_seq = window_seq + "N" * right_pad

                    # Ensure correct length
                    if len(window_seq) != seq_len:
                        warnings.warn(
                            f"Sequence length mismatch for variant at {chrom}:{pos}. "
                            f"Expected {seq_len}, got {len(window_seq)}"
                        )
                    # Truncate or pad as needed
                    if len(window_seq) < seq_len:
                        window_seq += "N" * (seq_len - len(window_seq))
                    else:
                        window_seq = window_seq[:seq_len]

                    if encode:
                        sequences.append(encode_seq(window_seq, encoder))
                    else:
                        sequences.append(
                            (
                                chrom,
                                max(0, genomic_pos - half_len),
                                max(0, genomic_pos - half_len) + seq_len,
                                window_seq,
                            )
                        )

            # Yield each chunk immediately
            if encode and sequences:
                if TORCH_AVAILABLE:
                    sequences_result = torch.stack(sequences)
                else:
                    sequences_result = np.stack(sequences)
            else:
                sequences_result = sequences

            yield (sequences_result, metadata_df)

    # Process BND variants
    bnd_alt_sequences, bnd_left_refs, bnd_right_refs, bnd_metadata = _generate_bnd_sequences(
        breakend_pairs, reference, seq_len, encode, encoder
    )

    # Process BND variants after standard variants (if any)
    # BND variants are yielded as a single batch for now
    if len(bnd_alt_sequences) > 0:
        bnd_metadata_df = pd.DataFrame(bnd_metadata) if bnd_metadata else pd.DataFrame()

        # BND sequences are already stacked by _generate_bnd_sequences
        bnd_sequences_result = bnd_alt_sequences

        yield (bnd_sequences_result, bnd_metadata_df)


def get_ref_sequences(reference_fn, variants_fn, seq_len, encode=True, n_chunks=1, encoder=None):
    """
    Create reference sequence windows centered on each variant position (no variants applied).
    Now supports both standard variants and BND variants with dual references.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to VCF file (string) or DataFrame with variant data.
                    For DataFrames, position column can be 'pos', 'pos1', or assumes second column is position.
        seq_len: Length of the sequence window
        encode: Return sequences as one-hot encoded numpy arrays (default: True)
        n_chunks: Number of chunks to split variants into (default: 1)

    Yields:
        Tuple containing (sequences, metadata_df) where:
            If encode=True: sequences is a tensor/array of shape (chunk_size, seq_len, 4) for each chunk
            If encode=False: sequences is a list of tuples containing (chrom, start, end, sequence_string) for each chunk
            metadata_df is a DataFrame with variant information including position offsets
            For BND variants: sequences contain dual reference sequences (left + right)
    """
    # Load reference and variants, separating BNDs from standard variants
    from .variant_utils import load_breakend_variants
    reference = _load_reference(reference_fn)

    # Load variants and separate BNDs
    standard_variants, breakend_pairs = load_breakend_variants(variants_fn)

    # Combine chromosome names from both standard variants and breakend pairs
    ref_chroms = set(reference.keys())
    standard_chroms = set(standard_variants["chrom"].unique()) if len(standard_variants) > 0 else set()
    breakend_chroms = set()
    for bnd1, bnd2 in breakend_pairs:
        breakend_chroms.add(bnd1.chrom)
        breakend_chroms.add(bnd2.chrom)
    vcf_chroms = standard_chroms | breakend_chroms

    # Use chromosome matching to handle name mismatches
    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=True
    )

    # Apply chromosome name mapping to standard variants
    if mapping and len(standard_variants) > 0:
        standard_variants = apply_chromosome_mapping(standard_variants, mapping)

    # Apply chromosome name mapping to breakend pairs
    if mapping and breakend_pairs:
        updated_pairs = []
        for bnd1, bnd2 in breakend_pairs:
            # Update chromosome names in breakend objects if needed
            if bnd1.chrom in mapping:
                bnd1.chrom = mapping[bnd1.chrom]
            if bnd2.chrom in mapping:
                bnd2.chrom = mapping[bnd2.chrom]
            if bnd1.mate_chrom in mapping:
                bnd1.mate_chrom = mapping[bnd1.mate_chrom]
            if bnd2.mate_chrom in mapping:
                bnd2.mate_chrom = mapping[bnd2.mate_chrom]
            updated_pairs.append((bnd1, bnd2))
        breakend_pairs = updated_pairs

    # Process standard variants first - yield each chunk individually
    if len(standard_variants) > 0:
        # Split standard variants into chunks
        std_indices = np.array_split(np.arange(len(standard_variants)), n_chunks)
        std_variant_chunks = (
            standard_variants.iloc[chunk_indices].reset_index(drop=True)
            for chunk_indices in std_indices
            if len(chunk_indices) > 0
        )

        for chunk_variants in std_variant_chunks:
            sequences = []
            # Generate standardized metadata using shared function
            metadata_df = _generate_sequence_metadata(chunk_variants, seq_len)

            # Group variants by chromosome for efficient processing
            for chrom, chrom_variants in chunk_variants.groupby("chrom"):
                if chrom not in reference:
                    warnings.warn(
                        f"Chromosome {chrom} not found in reference. Skipping {len(chrom_variants)} variants."
                    )
                    continue

                ref_seq = str(reference[chrom])
                chrom_length = len(ref_seq)

                # Vectorized calculation of window positions
                positions = chrom_variants["pos1"].values - 1  # Convert to 0-based
                half_len = seq_len // 2
                window_starts = positions - half_len
                window_ends = window_starts + seq_len

                # Process all variants in this chromosome using NumPy operations
                for idx, (_, var) in enumerate(chrom_variants.iterrows()):
                    pos = var["pos1"]
                    genomic_pos = positions[idx]
                    window_start = window_starts[idx]
                    window_end = window_ends[idx]

                    # Handle edge cases and extract window
                    if window_start < 0:
                        left_pad = -window_start
                        actual_start = 0
                    else:
                        left_pad = 0
                        actual_start = window_start

                    if window_end > chrom_length:
                        right_pad = window_end - chrom_length
                        actual_end = chrom_length
                    else:
                        right_pad = 0
                        actual_end = window_end

                    # Extract window from reference chromosome (no variants applied)
                    window_seq = ref_seq[actual_start:actual_end]

                    # Add padding if needed
                    if left_pad > 0:
                        window_seq = "N" * left_pad + window_seq
                    if right_pad > 0:
                        window_seq = window_seq + "N" * right_pad

                    # Ensure correct length
                    if len(window_seq) != seq_len:
                        warnings.warn(
                            f"Sequence length mismatch for variant at {chrom}:{pos}. "
                            f"Expected {seq_len}, got {len(window_seq)}"
                        )
                        # Truncate or pad as needed
                        if len(window_seq) < seq_len:
                            window_seq += "N" * (seq_len - len(window_seq))
                        else:
                            window_seq = window_seq[:seq_len]

                    if encode:
                        sequences.append(encode_seq(window_seq, encoder))
                    else:
                        sequences.append(
                            (
                                chrom,
                                max(0, genomic_pos - half_len),
                                max(0, genomic_pos - half_len) + seq_len,
                                window_seq,
                            )
                        )

            # Yield each chunk immediately
            if encode and sequences:
                if TORCH_AVAILABLE:
                    sequences_result = torch.stack(sequences)
                else:
                    sequences_result = np.stack(sequences)
            else:
                sequences_result = sequences

            yield (sequences_result, metadata_df)

    # Process BND variants after standard variants (if any)
    # BND variants are yielded as dual references
    if breakend_pairs:
        bnd_left_refs, bnd_right_refs, bnd_metadata = _generate_bnd_ref_sequences(
            breakend_pairs, reference, seq_len, encode, encoder
        )

        if len(bnd_left_refs) > 0 or len(bnd_right_refs) > 0:
            bnd_metadata_df = pd.DataFrame(bnd_metadata) if bnd_metadata else pd.DataFrame()

            # For ref sequences, we return dual references as a tuple
            # This is different from get_alt_sequences which returns fusion sequences
            # BND ref sequences are already stacked by _generate_bnd_ref_sequences
            bnd_left_result = bnd_left_refs
            bnd_right_result = bnd_right_refs

            # Return dual references as a tuple (left_refs, right_refs)
            yield ((bnd_left_result, bnd_right_result), bnd_metadata_df)


def get_alt_ref_sequences(
    reference_fn, variants_fn, seq_len, encode=True, n_chunks=1, encoder=None
):
    """
    Create both reference and variant sequence windows for alt/ref ratio calculations.
    Maintains backward compatibility while supporting BND variants with dual references.

    This wrapper function calls both get_ref_sequences and get_alt_sequences to return
    matching pairs of reference and variant sequences for computing ratios.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to VCF file (string) or DataFrame with variant data.
                    For DataFrames, position column can be 'pos', 'pos1', or assumes second column is position.
        seq_len: Length of the sequence window
        encode: Return sequences as one-hot encoded numpy arrays (default: True)
        n_chunks: Number of chunks to split variants into (default: 1)
        encoder: Optional custom encoder function

    Yields:
        Tuple containing (alt_sequences, ref_sequences, metadata_df):
            For standard variants:
            - alt_sequences: Variant sequences with mutations applied
            - ref_sequences: Reference sequences without mutations
            - metadata_df: Variant metadata

            For BND variants:
            - alt_sequences: Fusion sequences from breakend pairs
            - ref_sequences: Tuple of (left_ref_sequences, right_ref_sequences)
            - metadata_df: BND metadata with orientation and mate information
    """
    # Get generators for both reference and variant sequences
    # These already handle variant loading, chromosome matching, and chunking consistently
    ref_gen = get_ref_sequences(reference_fn, variants_fn, seq_len, encode, n_chunks, encoder)
    alt_gen = get_alt_sequences(reference_fn, variants_fn, seq_len, encode, n_chunks, encoder)

    # Process chunks from both generators
    # Both generators will yield chunks in the same order:
    # 1. Standard variant chunks first (if any) - maintains existing behavior
    # 2. BND variant chunks last (if any) - new dual reference structure
    for (ref_chunk, ref_metadata), (alt_chunk, alt_metadata) in zip(ref_gen, alt_gen):
        # For standard variants: preserve existing behavior exactly
        # For BND variants: ref_chunk will be (left_refs, right_refs) tuple
        # The caller can detect BND chunks by checking if ref_chunk is a tuple
        yield (alt_chunk, ref_chunk, ref_metadata)


def get_pam_disrupting_personal_sequences(
    reference_fn,
    variants_fn,
    seq_len,
    max_pam_distance,
    pam_sequence="NGG",
    encode=True,
    n_chunks=1,
    encoder=None,
):
    """
    Generate sequences for variants that disrupt PAM sites.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        seq_len: Length of sequence windows
        max_pam_distance: Maximum distance from variant to PAM site
        pam_sequence: PAM sequence pattern (default: 'NGG' for SpCas9)
        encode: Return sequences as one-hot encoded numpy arrays (default: True)
        n_chunks: Number of chunks to split variants for processing (default: 1)

    Returns:
        A dictionary containing:
            - variants: List of variants that disrupt PAM sites
            - pam_intact: List of sequences with variant applied but PAM intact
            - pam_disrupted: List of sequences with both variant and PAM disrupted
    """
    # Load reference and variants
    reference = _load_reference(reference_fn)
    variants = _load_variants(variants_fn)

    # Get all chromosome names and apply chromosome matching
    ref_chroms = set(reference.keys())
    vcf_chroms = set(variants["chrom"].unique())

    # Use chromosome matching to handle name mismatches
    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=True
    )

    # Apply chromosome name mapping to variants
    if mapping:
        variants = apply_chromosome_mapping(variants, mapping)

    pam_disrupting_variants = []
    pam_intact_sequences = []
    pam_disrupted_sequences = []

    # Process each variant individually
    for _, var in variants.iterrows():
        chrom = var["chrom"]
        pos = var["pos1"]  # 1-based position

        # Get reference sequence for this chromosome
        if chrom not in reference:
            warnings.warn(
                f"Chromosome {chrom} not found in reference. Skipping variant at {chrom}:{pos}."
            )
            continue

        ref_seq = str(reference[chrom])
        chrom_length = len(ref_seq)

        # Convert to 0-based position
        genomic_pos = pos - 1

        # Calculate window boundaries centered on variant start
        half_len = seq_len // 2
        window_start = genomic_pos - half_len
        window_end = window_start + seq_len

        # Handle edge cases for reference sequence PAM detection
        if window_start < 0:
            left_pad = -window_start
            ref_window_start = 0
        else:
            left_pad = 0
            ref_window_start = window_start

        if window_end > chrom_length:
            right_pad = window_end - chrom_length
            ref_window_end = chrom_length
        else:
            right_pad = 0
            ref_window_end = window_end

        # Extract window from reference for PAM detection
        ref_window_seq = ref_seq[ref_window_start:ref_window_end]

        # Add padding for PAM detection
        if left_pad > 0:
            ref_window_seq = "N" * left_pad + ref_window_seq
        if right_pad > 0:
            ref_window_seq = ref_window_seq + "N" * right_pad

        # Find PAM sites in the reference sequence window
        pam_sites = []
        for i in range(len(ref_window_seq) - len(pam_sequence) + 1):
            potential_pam = ref_window_seq[i : i + len(pam_sequence)]
            # Check if the potential PAM matches the pattern
            if all(
                a == b or b == "N"
                for a, b in zip(potential_pam.upper(), pam_sequence.upper())
            ):
                pam_sites.append(i)

        # Calculate variant position in padded window
        variant_pos_in_window = left_pad + (genomic_pos - ref_window_start)

        # Filter PAM sites that are within max_pam_distance of the variant
        nearby_pam_sites = [
            p for p in pam_sites if abs(p - variant_pos_in_window) <= max_pam_distance
        ]

        if nearby_pam_sites:
            pam_disrupting_variants.append(var)

            # Create a temporary applicator with just this variant
            single_var_df = pd.DataFrame([var])
            temp_applicator = VariantApplicator(ref_seq, single_var_df)

            # Apply the variant to get the full modified chromosome
            modified_chrom, stats = temp_applicator.apply_variants()

            # Extract window from modified chromosome
            if window_start < 0:
                actual_start = 0
            else:
                actual_start = window_start

            if window_end > len(modified_chrom):
                actual_end = len(modified_chrom)
            else:
                actual_end = window_end

            modified_window = modified_chrom[actual_start:actual_end]

            # Add padding
            if left_pad > 0:
                modified_window = "N" * left_pad + modified_window
            if right_pad > 0:
                modified_window = modified_window + "N" * right_pad

            # Ensure correct length
            if len(modified_window) != seq_len:
                if len(modified_window) < seq_len:
                    modified_window += "N" * (seq_len - len(modified_window))
                else:
                    modified_window = modified_window[:seq_len]

            # Store PAM-intact sequence
            final_intact = encode_seq(modified_window, encoder) if encode else modified_window
            pam_intact_sequences.append(
                (
                    chrom,
                    max(0, genomic_pos - half_len),
                    max(0, genomic_pos - half_len) + seq_len,
                    final_intact,
                )
            )

            # Create PAM-disrupted versions
            for pam_site in nearby_pam_sites:
                # Disrupt PAM with NNN
                disrupted_seq = list(modified_window)
                for j in range(len(pam_sequence)):
                    if pam_site + j < len(disrupted_seq):
                        disrupted_seq[pam_site + j] = "N"
                disrupted_seq = "".join(disrupted_seq)

                final_disrupted = encode_seq(disrupted_seq, encoder) if encode else disrupted_seq
                pam_disrupted_sequences.append(
                    (
                        chrom,
                        max(0, genomic_pos - half_len),
                        max(0, genomic_pos - half_len) + seq_len,
                        final_disrupted,
                    )
                )

    return {
        "variants": pam_disrupting_variants,
        "pam_intact": pam_intact_sequences,
        "pam_disrupted": pam_disrupted_sequences,
    }


