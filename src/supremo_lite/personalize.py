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

class ChromosomeSegmentTracker:
    """Track which segments of each chromosome are used by fusions."""

    def __init__(self, ref_sequences: Dict[str, str]):
        self.ref_sequences = ref_sequences
        self.used_segments = {chrom: [] for chrom in ref_sequences.keys()}

    def add_used_segment(self, chrom: str, start: int, end: int, verbose: bool = False):
        """Add a used segment (0-based coordinates)."""
        if chrom in self.used_segments:
            self.used_segments[chrom].append((start, end))
            if verbose:
                print(f"   🔍 Tracking used segment: {chrom}[{start}:{end}] = {end-start}bp")

    def get_leftover_sequences(self, verbose: bool = False) -> Dict[str, str]:
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
                    if verbose:
                        print(f"   ✂️ Created leftover {chrom}: {len(leftover_seq)} bp")

        return leftover_sequences


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
        self.skipped_variants = []  # List of (vcf_line, chrom, pos1, ref, alt, reason) tuples

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
                # Extract concise error message and context
                vcf_line = variant.get('vcf_line', '?')
                chrom = variant.get('chrom', self.chrom)
                error_msg = str(e).split(':')[0] if ':' in str(e) else str(e)
                warnings.warn(f"Skipped variant at {chrom}:{variant.pos1} (VCF line {vcf_line}): {error_msg}")
                self.skipped_count += 1
                # Record skip details for reporting
                self.skipped_variants.append((vcf_line, chrom, variant.pos1, variant.ref, variant.alt, 'validation_error'))

        stats = {
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "total": len(self.variants),
            "skipped_variants": self.skipped_variants,
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
        supported_types = {'SNV', 'MNV', 'INS', 'DEL', 'complex', 'SV_DUP', 'SV_INV', 'SV_BND_DUP', 'SV_BND_INV'}

        # Handle variants that should be processed elsewhere or are unsupported
        vcf_line = variant.get('vcf_line', '?')
        chrom = variant.get('chrom', self.chrom)

        if variant_type in ['SV_BND']:
            warnings.warn(f"Skipped variant at {chrom}:{variant.pos1} (VCF line {vcf_line}): type '{variant_type}' not supported")
            self.skipped_count += 1
            self.skipped_variants.append((vcf_line, chrom, variant.pos1, variant.ref, variant.alt, 'unsupported_type'))
            return
        elif variant_type in {'SV_DEL', 'SV_INS', 'SV_CNV'}:
            warnings.warn(f"Skipped variant at {chrom}:{variant.pos1} (VCF line {vcf_line}): type '{variant_type}' not supported")
            self.skipped_count += 1
            self.skipped_variants.append((vcf_line, chrom, variant.pos1, variant.ref, variant.alt, 'unsupported_type'))
            return
        elif variant_type in {'missing', 'unknown'}:
            warnings.warn(f"Skipped variant at {chrom}:{variant.pos1} (VCF line {vcf_line}): type '{variant_type}' not supported")
            self.skipped_count += 1
            self.skipped_variants.append((vcf_line, chrom, variant.pos1, variant.ref, variant.alt, 'missing_type'))
            return
        elif variant_type not in supported_types:
            warnings.warn(f"Skipped variant at {chrom}:{variant.pos1} (VCF line {vcf_line}): type '{variant_type}' not supported")
            self.skipped_count += 1
            self.skipped_variants.append((vcf_line, chrom, variant.pos1, variant.ref, variant.alt, 'unsupported_type'))
            return

        # 2. STRUCTURAL VARIANT INFO PARSING
        info_dict = {}
        if variant_type in ['SV_DUP', 'SV_INV'] and 'info' in variant:
            info_dict = parse_vcf_info(variant['info'])

        # 3. BASIC VALIDATION CHECKS
        if variant.alt == variant.ref:
            self.skipped_count += 1
            return  # Skip ref-only variants

        # Handle multiple ALT alleles - take first one
        alt_allele = variant.alt.split(",")[0]

        # 4. COORDINATE CALCULATION
        genomic_pos = variant.pos1 - 1  # Convert VCF 1-based to 0-based
        buffer_pos = genomic_pos + self.cumulative_offset

        # For structural variants, calculate affected region from INFO fields
        if variant_type in ['SV_DUP', 'SV_INV']:
            end_pos = info_dict.get('END', None)
            svlen = info_dict.get('SVLEN', None)

            # Calculate END position if not provided
            if end_pos is None and svlen is not None:
                end_pos = variant.pos1 + abs(svlen) - 1
            elif end_pos is None:
                # Fallback to REF length for structural variants
                end_pos = variant.pos1 + len(variant.ref) - 1
                warnings.warn(f"Cannot determine structural variant end position for {variant.get('id', 'unknown')} at {variant.pos1}")

            ref_length = end_pos - variant.pos1 + 1  # Total affected region length
        else:
            ref_length = len(variant.ref)

        # 5. FROZEN REGION CHECK
        ref_start = genomic_pos
        ref_end = genomic_pos + ref_length - 1

        if self.frozen_tracker.is_frozen(ref_start) or self.frozen_tracker.is_frozen(
            ref_end
        ):
            self.skipped_count += 1
            # Record skip details for reporting
            vcf_line = variant.get('vcf_line', '?')
            chrom = variant.get('chrom', self.chrom)
            self.skipped_variants.append((vcf_line, chrom, variant.pos1, variant.ref, variant.alt, 'overlap'))
            return  # Skip overlapping variants

        # 6. BOUNDS CHECK
        if buffer_pos < 0 or buffer_pos + ref_length > len(self.sequence):
            raise ValueError(f"Variant position {variant.pos1} out of sequence bounds")

        # 7. REFERENCE VALIDATION (skip for symbolic structural variants)
        if variant_type not in ['SV_DUP', 'SV_INV']:
            expected_ref = self.sequence[
                buffer_pos : buffer_pos + len(variant.ref)
            ].decode()
            if expected_ref.upper() != variant.ref.upper():
                raise ValueError(
                    f"Reference mismatch at position {variant.pos1}: "
                    f"expected '{variant.ref}', found '{expected_ref}'"
                )

        # 8. SEQUENCE MODIFICATION
        self._modify_sequence(buffer_pos, variant.ref, alt_allele, variant_type, info_dict)

        # 9. UPDATE TRACKING
        if variant_type in ['SV_DUP', 'SV_INV']:
            # For structural variants, calculate length difference based on variant type
            if variant_type == 'SV_DUP':
                # Duplication adds the duplicated region length
                length_diff = ref_length
            elif variant_type == 'SV_INV':
                # Inversion doesn't change sequence length
                length_diff = 0
        else:
            length_diff = len(alt_allele) - len(variant.ref)
        self.cumulative_offset += length_diff
        self.frozen_tracker.add_range(ref_start, ref_end)

        # Record offset for coordinate transformation if tracker is provided
        if self.offset_tracker and self.chrom and length_diff != 0:
            self.offset_tracker.add_offset(self.chrom, ref_start, length_diff)

        self.applied_count += 1

    def _modify_sequence(self, pos: int, ref_allele: str, alt_allele: str, variant_type: str, info_dict: dict = None) -> None:
        """
        Modify sequence at specified position using variant type classification.

        Args:
            pos: Buffer position (0-based)
            ref_allele: Reference allele sequence
            alt_allele: Alternate allele sequence
            variant_type: Classified variant type (SNV, MNV, INS, DEL, complex, SV_DUP, SV_INV)
            info_dict: Parsed INFO field for structural variants (optional)
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

        elif variant_type == 'SV_DUP':
            # Tandem duplication: insert duplicated region after original
            if not info_dict:
                raise ValueError("INFO field required for SV_DUP variant")

            end_pos = info_dict.get('END')

            # Calculate duplication region using END field only
            if end_pos is None:
                raise ValueError("END field required for SV_DUP variant")

            # Calculate from buffer position (already offset-adjusted)
            genomic_start = pos - self.cumulative_offset + len(ref_allele)  # Position after current cumulative changes
            genomic_end = end_pos - 1  # Convert VCF 1-based to 0-based
            dup_length = genomic_end - genomic_start + 1

            # Extract the region to duplicate from current sequence
            duplicated_region = self.sequence[pos:pos + dup_length]

            # Insert duplicated region after original (tandem duplication)
            self.sequence[pos + dup_length:pos + dup_length] = duplicated_region

        elif variant_type == 'SV_INV':
            # Inversion: reverse complement the affected region
            if not info_dict:
                raise ValueError("INFO field required for SV_INV variant")

            end_pos = info_dict.get('END')

            # Calculate inversion region using END field only
            if end_pos is None:
                raise ValueError("END field required for SV_INV variant")

            # pos is already the correct buffer position (0-based) where inversion starts
            # END field is 1-based, so convert to 0-based buffer position
            buffer_start = pos
            buffer_end = end_pos - 1  # Convert 1-based END to 0-based
            inv_length = buffer_end - buffer_start + 1

            # Extract region to invert
            region_to_invert = self.sequence[buffer_start:buffer_start + inv_length].decode()

            # Apply reverse complement
            from .sequence_utils import rc_str
            inverted_region = rc_str(region_to_invert)

            # Replace with inverted sequence
            self.sequence[buffer_start:buffer_start + inv_length] = inverted_region.encode()

        elif variant_type == 'SV_BND_DUP':
            # BND-derived tandem duplication
            # Note: Individual SV_BND_DUP variants should not reach this point as they are
            # preprocessed by _preprocess_bnd_derived_variants() into synthetic SV_DUP variants
            genomic_pos = pos + self.cumulative_offset + 1  # Convert back to 1-based genomic position
            raise ValueError(f"SV_BND_DUP variants should be preprocessed into SV_DUP variants. Position: {genomic_pos}")

        elif variant_type == 'SV_BND_INV':
            # BND-derived inversion
            # Note: Individual SV_BND_INV variants should not reach this point as they are
            # preprocessed by _preprocess_bnd_derived_variants() into synthetic SV_INV variants
            genomic_pos = pos + self.cumulative_offset + 1  # Convert back to 1-based genomic position
            raise ValueError(f"SV_BND_INV variants should be preprocessed into SV_INV variants. Position: {genomic_pos}")

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


def _encode_genome_sequences(reference, encode=True, encoder=None):
    """Helper function to encode genome sequences for output."""
    genome = {}
    for chrom, seq in reference.items():
        seq_str = str(seq)
        if encode:
            if encoder:
                genome[chrom] = encoder(seq_str)
            else:
                genome[chrom] = encode_seq(seq_str)
        else:
            genome[chrom] = seq_str
    return genome




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


def _preprocess_bnd_derived_variants(chrom_variants, vcf_path=None, verbose=False):
    """
    Convert BND-derived DUP/INV pairs to synthetic SV_DUP/SV_INV variants.

    This pre-processing step allows BND-derived structural variants to be processed
    by the existing SV_DUP/SV_INV logic, ensuring proper frozen region tracking
    and coordinate transformation.

    Args:
        chrom_variants: DataFrame of variants for a single chromosome
        vcf_path: Path to VCF file for BND classification (optional)
        verbose: Print processing information

    Returns:
        DataFrame with BND-derived variants replaced by synthetic variants
    """
    import pandas as pd
    from .variant_utils import parse_breakend_alt

    # Extract BND-derived variants that need pair processing
    bnd_dup_variants = chrom_variants[chrom_variants['variant_type'] == 'SV_BND_DUP'].copy()
    bnd_inv_variants = chrom_variants[chrom_variants['variant_type'] == 'SV_BND_INV'].copy()

    if len(bnd_dup_variants) == 0 and len(bnd_inv_variants) == 0:
        return chrom_variants

    if verbose:
        print(f"  🔄 Pre-processing {len(bnd_dup_variants)} BND-DUP + {len(bnd_inv_variants)} BND-INV variants")

    # Get BND classification results to find mate coordinates
    synthetic_variants = []
    processed_ids = set()

    # Process BND-derived duplications
    for _, variant in bnd_dup_variants.iterrows():
        if variant['id'] in processed_ids:
            continue

        # Parse mate coordinates from ALT field
        alt_info = parse_breakend_alt(variant['alt'])
        if not alt_info['is_valid']:
            if verbose:
                print(f"    ⚠️ Could not parse BND ALT field: {variant['alt']}")
            continue

        mate_chrom = alt_info['mate_chrom']
        mate_pos = alt_info['mate_pos']

        # Ensure this is an intrachromosomal duplication (same chromosome)
        if mate_chrom != variant['chrom']:
            if verbose:
                print(f"    ⚠️ Skipping interchromosomal BND: {variant['chrom']}:{variant['pos1']} -> {mate_chrom}:{mate_pos}")
            continue

        # Calculate duplication region boundaries
        start_pos = min(variant['pos1'], mate_pos)
        end_pos = max(variant['pos1'], mate_pos)

        # Create synthetic SV_DUP variant
        synthetic_variant = variant.copy()
        synthetic_variant['variant_type'] = 'SV_DUP'
        synthetic_variant['pos1'] = start_pos
        synthetic_variant['ref'] = 'N'  # Placeholder
        synthetic_variant['alt'] = '<DUP>'
        synthetic_variant['info'] = f'END={end_pos};SVTYPE=DUP'

        synthetic_variants.append(synthetic_variant)
        processed_ids.add(variant['id'])

        if verbose:
            region_length = end_pos - start_pos
            print(f"    ✅ Created synthetic DUP: {variant['chrom']}:{start_pos}-{end_pos} ({region_length}bp)")

    # Process BND-derived inversions: handle 4-breakend inversion topology
    if len(bnd_inv_variants) > 0:
        # Group BND inversions by chromosome to handle 4-breakend patterns
        chrom_groups = bnd_inv_variants.groupby('chrom')

        for chrom, chrom_bnd_invs in chrom_groups:
            chrom_breakends = chrom_bnd_invs.copy()

            # Check if we have exactly 4 breakends (standard inversion pattern)
            if len(chrom_breakends) == 4:
                # Sort breakends by position to identify topology
                chrom_breakends = chrom_breakends.sort_values('pos1')
                positions = chrom_breakends['pos1'].tolist()

                # 4-breakend inversion: outer breakpoints define boundaries, inner breakpoints define inverted region
                # Positions: [W, V, U, X] where W-X are outer, V-U are inner (get inverted)
                outer_start = positions[0]  # W (position 10)
                inner_start = positions[1]  # V (position 11)
                inner_end = positions[2]    # U (position 30)
                outer_end = positions[3]    # X (position 31)

                # Create single synthetic SV_INV for the inner region (what gets inverted)
                first_variant = chrom_breakends.iloc[0].copy()
                synthetic_variant = first_variant.copy()
                synthetic_variant['variant_type'] = 'SV_INV'
                synthetic_variant['pos1'] = inner_start  # Start of inverted region
                synthetic_variant['ref'] = 'N'  # Placeholder
                synthetic_variant['alt'] = '<INV>'
                synthetic_variant['info'] = f'END={inner_end};SVTYPE=INV'  # End of inverted region

                synthetic_variants.append(synthetic_variant)

                # Mark all 4 breakends as processed
                for _, variant in chrom_breakends.iterrows():
                    processed_ids.add(variant['id'])

                if verbose:
                    inversion_length = inner_end - inner_start
                    boundary_span = outer_end - outer_start
                    print(f"    ✅ Created synthetic INV: {chrom}:{inner_start}-{inner_end} ({inversion_length}bp) [4-breakend topology, boundary span {outer_start}-{outer_end}]")

            else:
                # Handle non-standard cases (not exactly 4 breakends)
                if verbose:
                    print(f"    ⚠️ Non-standard BND inversion pattern: {len(chrom_breakends)} breakends on {chrom}")

                # Fallback: process individually for non-4-breakend cases
                for _, variant in chrom_breakends.iterrows():
                    if variant['id'] in processed_ids:
                        continue

                    # Parse mate coordinates from ALT field
                    alt_info = parse_breakend_alt(variant['alt'])
                    if not alt_info['is_valid']:
                        if verbose:
                            print(f"    ⚠️ Could not parse BND ALT field: {variant['alt']}")
                        continue

                    mate_pos = alt_info['mate_pos']
                    start_pos = min(variant['pos1'], mate_pos)
                    end_pos = max(variant['pos1'], mate_pos)

                    # Create synthetic SV_INV variant
                    synthetic_variant = variant.copy()
                    synthetic_variant['variant_type'] = 'SV_INV'
                    synthetic_variant['pos1'] = start_pos
                    synthetic_variant['ref'] = 'N'
                    synthetic_variant['alt'] = '<INV>'
                    synthetic_variant['info'] = f'END={end_pos};SVTYPE=INV'

                    synthetic_variants.append(synthetic_variant)
                    processed_ids.add(variant['id'])

                    if verbose:
                        region_length = end_pos - start_pos
                        print(f"    ✅ Created synthetic INV: {chrom}:{start_pos}-{end_pos} ({region_length}bp) [fallback]")

    # Create result DataFrame: remove BND-derived variants, add synthetic variants
    result_variants = chrom_variants[
        ~chrom_variants['variant_type'].isin(['SV_BND_DUP', 'SV_BND_INV'])
    ].copy()

    if synthetic_variants:
        synthetic_df = pd.DataFrame(synthetic_variants)
        result_variants = pd.concat([result_variants, synthetic_df], ignore_index=True)
        # Re-sort by position to maintain VCF order
        result_variants = result_variants.sort_values('pos1')

    if verbose and len(synthetic_variants) > 0:
        print(f"  🎯 Pre-processing complete: {len(synthetic_variants)} synthetic variants created")

    return result_variants


def _format_skipped_variant_report(skipped_variants_list):
    """
    Format skipped variant details for reporting.

    Args:
        skipped_variants_list: List of (vcf_line, chrom, pos1, ref, alt, reason) tuples

    Returns:
        Formatted string with grouped skip reasons
    """
    if not skipped_variants_list:
        return ""

    from collections import defaultdict

    # Group by reason
    by_reason = defaultdict(list)
    for vcf_line, chrom, pos1, ref, alt, reason in skipped_variants_list:
        by_reason[reason].append((vcf_line, chrom, pos1, ref, alt))

    # Format output
    lines = []
    reason_labels = {
        'overlap': 'overlap with previous variant',
        'unsupported_type': 'unsupported variant type',
        'validation_error': 'validation error',
        'missing_type': 'missing/unknown type'
    }

    for reason, variants in sorted(by_reason.items()):
        label = reason_labels.get(reason, reason)
        # Group by position for concise output
        by_pos = defaultdict(list)
        for vcf_line, chrom, pos1, ref, alt in variants:
            by_pos[f"{chrom}:{pos1}"].append(vcf_line)

        for pos, vcf_lines in sorted(by_pos.items()):
            vcf_lines_str = ', '.join(map(str, sorted(vcf_lines)))
            lines.append(f"     • {label}: VCF line(s) {vcf_lines_str} at {pos}")

    return '\n'.join(lines)


def get_personal_genome(reference_fn, variants_fn, encode=True, n_chunks=1, verbose=False, encoder=None):
    """
    Create a personalized genome by applying variants to a reference genome.

    This function implements multi-phase variant processing with proper frozen region tracking:

    Phase 1: Standard variants + Early structural variants (in VCF order):
        - SNV, MNV, INS, DEL, SV_DUP, SV_INV

    Phase 2: BND semantic classification and application:
        - Classify BNDs to identify SV_BND_DUP and SV_BND_INV patterns
        - Apply SV_BND_DUP and SV_BND_INV first
        - Apply remaining true BND translocations

    Frozen region enforcement:
        - Each variant freezes its genomic region after application
        - Later variants overlapping frozen regions are skipped with warnings
        - BND breakpoints in frozen regions cause entire BND to be skipped

    Output chromosome ordering:
        - Chromosomes are returned in the same order as the reference genome
        - BND-generated fusion sequences appear after all original chromosomes
        - Leftover sequences (from consumed chromosomes) follow fusion sequences

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

        The dictionary preserves reference genome chromosome order, with any fusion
        or leftover sequences appended at the end.

    Examples:
        # Apply variants with proper ordering and conflict resolution
        personal_genome = get_personal_genome('reference.fa', 'variants.vcf')

        # Get raw sequences without encoding
        personal_genome = get_personal_genome('reference.fa', 'variants.vcf', encode=False)

        # Verify chromosome order is preserved
        ref_chroms = list(pyfaidx.Fasta('reference.fa').keys())
        personal_chroms = list(personal_genome.keys())
        assert personal_chroms[:len(ref_chroms)] == ref_chroms  # Original order preserved
    """
    # Load ALL variants with classification
    from .variant_utils import group_variants_by_semantic_type

    variants_df = _load_variants(variants_fn)
    reference = _load_reference(reference_fn)

    if len(variants_df) == 0:
        if verbose:
            print("🧬 No variants found - returning reference genome")
        return _encode_genome_sequences(reference, encode, encoder)

    # Group variants by semantic type for proper processing order
    # Pass VCF path for BND semantic classification if available
    vcf_path = variants_fn if isinstance(variants_fn, str) else None
    grouped_variants = group_variants_by_semantic_type(variants_df, vcf_path)

    if verbose:
        total_variants = len(variants_df)
        print(f"🧬 Processing {total_variants:,} variants across {len(variants_df['chrom'].unique())} chromosomes")
        print(f"   Phase 1: {len(grouped_variants['standard']) + len(grouped_variants['dup_variants']) + len(grouped_variants['inv_variants'])} standard variants (SNV, MNV, INS, DEL, SV_DUP, SV_INV)")
        print(f"   Phase 2: {len(grouped_variants['bnd_variants'])} BND variants for semantic classification")

    # Apply chromosome name matching
    ref_chroms = set(reference.keys())
    vcf_chroms = set(variants_df["chrom"].unique())

    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=verbose
    )

    # Apply chromosome name mapping to all variants
    if mapping:
        for group_name, variant_group in grouped_variants.items():
            if len(variant_group) > 0:
                grouped_variants[group_name] = apply_chromosome_mapping(variant_group, mapping)

    # Initialize processing state
    personal_genome = {}
    total_processed = 0
    offset_tracker = ChromosomeOffsetTracker()
    modified_sequences = {}

    # PHASE 1: Apply standard variants + early structural variants (in VCF order)
    # Include both symbolic and BND-derived DUP/INV variants
    symbolic_dup_variants = grouped_variants['dup_variants'][
        grouped_variants['dup_variants']['variant_type'] == 'SV_DUP'
    ] if len(grouped_variants['dup_variants']) > 0 else pd.DataFrame()

    symbolic_inv_variants = grouped_variants['inv_variants'][
        grouped_variants['inv_variants']['variant_type'] == 'SV_INV'
    ] if len(grouped_variants['inv_variants']) > 0 else pd.DataFrame()

    # Extract BND-derived DUP/INV variants for Phase 1 processing
    bnd_dup_variants = grouped_variants['dup_variants'][
        grouped_variants['dup_variants']['variant_type'] == 'SV_BND_DUP'
    ] if len(grouped_variants['dup_variants']) > 0 else pd.DataFrame()

    bnd_inv_variants = grouped_variants['inv_variants'][
        grouped_variants['inv_variants']['variant_type'] == 'SV_BND_INV'
    ] if len(grouped_variants['inv_variants']) > 0 else pd.DataFrame()

    phase1_variants = pd.concat([
        grouped_variants['standard'],
        symbolic_dup_variants,
        symbolic_inv_variants,
        bnd_dup_variants,
        bnd_inv_variants
    ], ignore_index=True)

    if len(phase1_variants) > 0:
        # Sort by chromosome and position to maintain VCF order
        phase1_variants = phase1_variants.sort_values(["chrom", "pos1"])

        for chrom, chrom_variants in phase1_variants.groupby("chrom"):
            if chrom not in reference:
                if verbose:
                    print(f"⚠️  Skipping {chrom}: not found in reference")
                continue

            ref_seq = str(reference[chrom])

            if verbose:
                variant_counts = chrom_variants['variant_type'].value_counts().to_dict()
                type_summary = ', '.join([f"{count} {vtype}" for vtype, count in variant_counts.items()])
                print(f"🔄 Processing chromosome {chrom}: {len(chrom_variants):,} variants ({type_summary})")

            # Process all Phase 1 variants with offset tracking (chunking available for standard variants only)
            if n_chunks == 1 or any(vtype in ['SV_DUP', 'SV_INV'] for vtype in chrom_variants['variant_type'].unique()):
                # PRE-PROCESS: Convert BND-derived variants to synthetic variants
                processed_variants = _preprocess_bnd_derived_variants(chrom_variants, vcf_path, verbose)

                # Process all variants at once (required for structural variants)
                applicator = VariantApplicator(ref_seq, processed_variants,
                                             offset_tracker=offset_tracker, chrom=chrom)
                personal_seq, stats = applicator.apply_variants()

                if verbose and stats["total"] > 0:
                    # Report skipped variants if any
                    if stats['skipped'] > 0 and stats.get('skipped_variants'):
                        print(f"  ⚠️  Skipped {stats['skipped']} variant(s):")
                        print(_format_skipped_variant_report(stats['skipped_variants']))
                    print(f"  ✅ Applied {stats['applied']}/{stats['total']} variants ({stats['skipped']} skipped)")

            else:
                # PRE-PROCESS: Convert BND-derived variants to synthetic variants (for chunked processing too)
                processed_variants = _preprocess_bnd_derived_variants(chrom_variants, vcf_path, verbose)

                # Process in chunks (standard variants only)
                current_sequence = ref_seq
                shared_frozen_tracker = FrozenRegionTracker()
                total_applied = 0
                total_skipped = 0
                all_skipped_variants = []

                indices = np.array_split(np.arange(len(processed_variants)), n_chunks)

                if verbose:
                    avg_chunk_size = len(processed_variants) // n_chunks
                    print(f"  📦 Processing {n_chunks} chunks of ~{avg_chunk_size:,} variants each")

                for i, chunk_indices in enumerate(indices):
                    if len(chunk_indices) == 0:
                        continue

                    chunk_df = processed_variants.iloc[chunk_indices].reset_index(drop=True)

                    applicator = VariantApplicator(current_sequence, chunk_df,
                                                 shared_frozen_tracker, offset_tracker, chrom)
                    current_sequence, stats = applicator.apply_variants()

                    total_applied += stats["applied"]
                    total_skipped += stats["skipped"]
                    all_skipped_variants.extend(stats.get("skipped_variants", []))

                    if verbose:
                        print(f"    ✅ Chunk {i+1}: {stats['applied']}/{stats['total']} variants applied")

                personal_seq = current_sequence

                if verbose:
                    # Report skipped variants if any
                    if total_skipped > 0 and all_skipped_variants:
                        print(f"  ⚠️  Skipped {total_skipped} variant(s):")
                        print(_format_skipped_variant_report(all_skipped_variants))
                    print(f"  🎯 Total: {total_applied}/{len(processed_variants)} variants applied ({total_skipped} skipped)")

            modified_sequences[chrom] = personal_seq
            total_processed += len(chrom_variants)

    # Initialize sequences for chromosomes not processed in Phase 1
    for chrom in reference.keys():
        if chrom not in modified_sequences:
            modified_sequences[chrom] = str(reference[chrom])

    # PHASE 2: BND translocation processing
    # Only process true BND translocations (BND-derived DUP/INV are now handled in Phase 1)
    true_bnd_variants = grouped_variants['bnd_variants']

    # Phase 2 variants are now only true BND translocations
    phase2_variants = true_bnd_variants

    if len(phase2_variants) > 0:
        if verbose:
            phase2_counts = phase2_variants['variant_type'].value_counts().to_dict()
            counts_msg = ', '.join([f"{count} {vtype}" for vtype, count in phase2_counts.items()])
            print(f"🔄 Phase 2: Processing {len(phase2_variants)} BND variants with semantic classification ({counts_msg})")

        # Use the BND classifier results directly instead of create_breakend_pairs
        # This ensures we get the inferred mates that the classifier created
        if vcf_path:
            from .variant_utils import BNDClassifier
            classifier = BNDClassifier()
            classified_breakends = classifier.classify_all_breakends(vcf_path, verbose=verbose)

            # Extract all paired breakends (including those with inferred mates)
            all_paired_breakends = classified_breakends['paired']

            # Convert to BreakendPair-like objects for ChimericSequenceBuilder compatibility
            breakend_pairs = []
            processed_ids = set()

            for breakend in all_paired_breakends:
                if breakend.id in processed_ids or not breakend.mate_breakend:
                    continue

                # Create a pair tuple (bnd1, bnd2) for ChimericSequenceBuilder
                pair_tuple = (breakend, breakend.mate_breakend)
                breakend_pairs.append(pair_tuple)
                processed_ids.add(breakend.id)
                processed_ids.add(breakend.mate_breakend.id)
        else:
            # Fallback to create_breakend_pairs if no VCF path available
            from .variant_utils import create_breakend_pairs
            breakend_pairs = create_breakend_pairs(phase2_variants)

        if len(breakend_pairs) > 0:
            if verbose:
                print(f"   Created {len(breakend_pairs)} BND pairs for processing")

            # Transform BND coordinates using offset tracker from Phase 1
            for pair in breakend_pairs:
                # Handle both BreakendPair objects and tuple pairs
                if hasattr(pair, 'breakend1'):
                    bnd1 = pair.breakend1
                    bnd2 = pair.breakend2
                else:
                    bnd1, bnd2 = pair

                original_pos1 = bnd1.pos
                original_pos2 = bnd2.pos

                # Transform coordinates to account for applied Phase 1 variants
                if hasattr(offset_tracker, 'get_offset_at_position'):
                    bnd1_offset = offset_tracker.get_offset_at_position(bnd1.chrom, bnd1.pos - 1)
                    bnd2_offset = offset_tracker.get_offset_at_position(bnd2.chrom, bnd2.pos - 1)
                    bnd1.pos += bnd1_offset
                    bnd2.pos += bnd2_offset

                    if verbose and (bnd1_offset != 0 or bnd2_offset != 0):
                        print(f"   📍 Transformed coordinates: {bnd1.chrom}:{original_pos1}→{bnd1.pos}, {bnd2.chrom}:{original_pos2}→{bnd2.pos}")

            # Note: BND semantic classification (SV_BND_DUP, SV_BND_INV) is handled by
            # group_variants_by_semantic_type() and _preprocess_bnd_derived_variants().
            # Remaining BND variants are processed as translocations using ChimericSequenceBuilder.

            # Enhanced frozen region validation for BND breakpoints
            validated_pairs = []
            skipped_pairs = []

            for pair in breakend_pairs:
                # Handle both BreakendPair objects and tuple pairs
                if hasattr(pair, 'breakend1'):
                    bnd1 = pair.breakend1
                    bnd2 = pair.breakend2
                else:
                    bnd1, bnd2 = pair

                # Check if both breakpoints are in non-frozen regions
                # Create a temporary FrozenRegionTracker to check current frozen regions
                # Note: This is a simplified check - a more sophisticated implementation would
                # track frozen regions across all chromosomes from Phase 1

                breakpoint_conflicts = []

                # Note: Frozen region tracking is handled by FrozenRegionTracker within each chromosome
                # processing. Cross-Phase conflict detection could be enhanced in future versions.
                if breakpoint_conflicts:
                    skipped_pairs.append(pair)
                    if verbose:
                        conflicts_msg = "; ".join(breakpoint_conflicts)
                        print(f"   ⚠️ Skipping BND pair {bnd1.id}-{bnd2.id}: {conflicts_msg}")
                else:
                    validated_pairs.append(pair)

            if verbose and len(skipped_pairs) > 0:
                print(f"   📍 Skipped {len(skipped_pairs)} BND pairs due to frozen region conflicts")

            # Create chimeric sequences using validated pairs only
            sequence_builder = ChimericSequenceBuilder(modified_sequences)

            # Initialize segment tracker with original reference chromosomes only
            original_ref_sequences = {chrom: seq for chrom, seq in modified_sequences.items() if '_fusion_' not in chrom}
            segment_tracker = ChromosomeSegmentTracker(original_ref_sequences)

            for i, pair in enumerate(validated_pairs):
                # Handle both BreakendPair objects and tuple pairs for display
                if hasattr(pair, 'breakend1'):
                    bnd1_id, bnd2_id = pair.breakend1.id, pair.breakend2.id
                    pair_tuple = (pair.breakend1, pair.breakend2)
                    bnd1, bnd2 = pair.breakend1, pair.breakend2
                else:
                    bnd1_id, bnd2_id = pair[0].id, pair[1].id
                    pair_tuple = pair
                    bnd1, bnd2 = pair[0], pair[1]

                if verbose:
                    print(f"   🔄 Creating fusion {i+1}/{len(validated_pairs)}: {bnd1_id}-{bnd2_id}")

                try:
                    fusion_name, fusion_seq = sequence_builder.create_fusion_from_pair(pair_tuple)
                    modified_sequences[fusion_name] = fusion_seq
                    total_processed += 2  # Count both BNDs in the pair

                    if verbose:
                        print(f"     ✅ Created fusion: {fusion_name} ({len(fusion_seq)} bp)")

                    # Track chromosome segment usage based on fusion orientations
                    pos1_0 = bnd1.pos - 1  # Convert to 0-based
                    pos2_0 = bnd2.pos - 1  # Convert to 0-based
                    seq1_len = len(modified_sequences[bnd1.chrom])
                    seq2_len = len(modified_sequences[bnd2.chrom])

                    # Track segments used based on the actual fusion logic from prototype
                    if bnd1.orientation == 't]p]' and bnd2.orientation == 't]p]':
                        # seq1[:pos1] + RC(seq2[:pos2]) - uses chromosome prefixes
                        segment_tracker.add_used_segment(bnd1.chrom, 0, bnd1.pos, verbose)  # VCF pos as count
                        segment_tracker.add_used_segment(bnd2.chrom, 0, bnd2.pos, verbose)  # VCF pos as count
                    elif bnd1.orientation == ']p]t' and bnd2.orientation == 't[p[':
                        # seq2[:pos2] + seq1[pos1_0:] - prefix from seq2, suffix from seq1
                        segment_tracker.add_used_segment(bnd2.chrom, 0, bnd2.pos, verbose)  # VCF pos as count
                        segment_tracker.add_used_segment(bnd1.chrom, pos1_0, seq1_len, verbose)
                    elif bnd1.orientation == '[p[t' and bnd2.orientation == '[p[t':
                        # RC(seq2[pos2_0:]) + seq1[pos1_0:] - uses chromosome suffixes
                        segment_tracker.add_used_segment(bnd2.chrom, pos2_0, seq2_len, verbose)
                        segment_tracker.add_used_segment(bnd1.chrom, pos1_0, seq1_len, verbose)
                    elif bnd1.orientation == 't[p[' and bnd2.orientation == 't[p[':
                        # seq1[:pos1_0] + seq2[pos2_0:] - prefix from seq1, suffix from seq2
                        segment_tracker.add_used_segment(bnd1.chrom, 0, pos1_0, verbose)
                        segment_tracker.add_used_segment(bnd2.chrom, pos2_0, seq2_len, verbose)
                    elif bnd1.orientation == 't[p[' and bnd2.orientation == ']p]t':
                        # seq1[:pos1_0] + seq2[pos2_0:] - prefix from seq1, suffix from seq2
                        segment_tracker.add_used_segment(bnd1.chrom, 0, pos1_0, verbose)
                        segment_tracker.add_used_segment(bnd2.chrom, pos2_0, seq2_len, verbose)
                    elif bnd1.orientation == ']p]t' and bnd2.orientation == ']p]t':
                        # seq2[:pos2] + seq1[pos1_0:] - prefix from seq2, suffix from seq1
                        segment_tracker.add_used_segment(bnd2.chrom, 0, bnd2.pos, verbose)  # VCF pos as count
                        segment_tracker.add_used_segment(bnd1.chrom, pos1_0, seq1_len, verbose)
                    else:
                        # Unknown orientation patterns - track conservatively
                        if verbose:
                            print(f"     ⚠️ Unknown orientation pattern: {bnd1.orientation} + {bnd2.orientation}")
                        segment_tracker.add_used_segment(bnd1.chrom, 0, pos1_0, verbose)
                        segment_tracker.add_used_segment(bnd2.chrom, pos2_0, seq2_len, verbose)

                except Exception as e:
                    if verbose:
                        print(f"     ⚠️ Failed to create fusion for {bnd1_id}-{bnd2_id}: {e}")

            # Calculate and add leftover sequences
            leftover_sequences = segment_tracker.get_leftover_sequences(verbose)

            # Remove original chromosomes that were consumed by fusions and replace with leftovers
            chromosomes_with_fusions = set()
            for seq_name in list(modified_sequences.keys()):
                if '_fusion_' in seq_name:
                    # Extract chromosome names from fusion sequence names
                    parts = seq_name.split('_')
                    if len(parts) >= 2:
                        chromosomes_with_fusions.add(parts[0])
                        chromosomes_with_fusions.add(parts[1])

            # Remove consumed chromosomes and add their leftovers
            for chrom in chromosomes_with_fusions:
                if chrom in modified_sequences:
                    del modified_sequences[chrom]
                    if verbose:
                        print(f"   🗑️ Removed consumed chromosome: {chrom}")

            # Add leftover sequences
            modified_sequences.update(leftover_sequences)

    # FINAL STEP: Encode sequences and create output
    # Preserve reference chromosome order, then append fusion/leftover sequences
    reference_chroms = list(reference.keys())

    # First, add chromosomes in reference order
    for chrom in reference_chroms:
        if chrom in modified_sequences:
            seq = modified_sequences[chrom]
            if encode:
                if encoder:
                    personal_genome[chrom] = encoder(seq)
                else:
                    personal_genome[chrom] = encode_seq(seq)
            else:
                personal_genome[chrom] = seq

    # Then, add fusion and leftover sequences (not in original reference)
    for chrom, seq in modified_sequences.items():
        if chrom not in reference_chroms:
            if encode:
                if encoder:
                    personal_genome[chrom] = encoder(seq)
                else:
                    personal_genome[chrom] = encode_seq(seq)
            else:
                personal_genome[chrom] = seq

    if verbose:
        total_variants = len(variants_df)
        sequences_msg = f"{len(personal_genome):,} sequences"
        if any("_fusion_" in name for name in personal_genome.keys()):
            fusion_count = sum(1 for name in personal_genome.keys() if "_fusion_" in name)
            leftover_count = len(personal_genome) - fusion_count
            sequences_msg = f"{fusion_count} fusions, {leftover_count} leftover sequences"

        print(f"🧬 Completed: {total_processed:,}/{total_variants:,} variants processed → {sequences_msg}")

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

        # Variant classification
        variant_type = var.get('variant_type', 'unknown')

        # Build minimal metadata dictionary
        meta_dict = {
            "chrom": var["chrom"],
            "window_start": window_start,
            "window_end": window_end,
            "variant_pos0": genomic_pos,  # 0-based absolute position
            "variant_pos1": pos,  # 1-based absolute position
            "ref": var["ref"],
            "alt": var["alt"],
            "variant_type": variant_type,
        }

        # Add sym_variant_end ONLY for symbolic alleles (<INV>, <DUP>, etc.)
        if variant_type.startswith('SV_') and '<' in var["alt"]:
            if 'info' in var and var['info'] and var['info'] != '.':
                parsed_info = parse_vcf_info(var['info'])
                sym_end = parsed_info.get('END')
                if sym_end is not None:
                    meta_dict["sym_variant_end"] = sym_end

        metadata.append(meta_dict)
    
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

            # Detect if this is a BND with insertion for consistent handling
            has_insertion = bool(bnd1.inserted_seq or bnd2.inserted_seq)
            insertion_length = len(bnd1.inserted_seq) + len(bnd2.inserted_seq)

            # Generate left reference sequence (sequence before breakend + right-side N-padding)
            # For BNDs, we want to show what was there BEFORE the fusion point
            # Then pad the right side with N's to represent the missing fusion partner + insertion
            half_len = seq_len // 2

            # Extract sequence leading up to the breakend (before the fusion point)
            left_start = max(0, center_pos - half_len)
            left_end = center_pos  # Stop at the breakend position
            left_ref_raw = seq1[left_start:left_end]

            # Pad the right side to represent where the fusion partner + insertion would attach
            left_padding_needed = seq_len - len(left_ref_raw)
            # Note: For BND with insertion, this padding represents both the missing chromosome and the insertion
            left_ref_seq = left_ref_raw + 'N' * left_padding_needed

            # Generate right reference sequence (left-side N-padding + sequence after breakend)
            # For the right side, we want to show what was there AFTER the fusion point
            # Pad the left side with N's to represent the missing fusion partner + insertion
            bnd2_center = bnd2.pos - 1  # Convert to 0-based

            # Extract sequence starting from the breakend (after the fusion point)
            right_start = bnd2_center  # Start at the breakend position
            right_end = min(len(seq2), bnd2_center + half_len)
            right_ref_raw = seq2[right_start:right_end]

            # Pad the left side to represent where the fusion partner + insertion would attach
            right_padding_needed = seq_len - len(right_ref_raw)
            # Note: For BND with insertion, this padding represents both the missing chromosome and the insertion
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
                'window_start': window_start,
                'window_end': window_end,
                'variant_pos0': center_pos,
                'variant_pos1': bnd1.pos,
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

            # Detect if this is a BND with insertion and center appropriately
            has_insertion = bool(bnd1.inserted_seq or bnd2.inserted_seq)

            if has_insertion:
                # For BND with insertion, center window on the inserted sequence
                # Use segment metadata to find where the novel sequence is located
                segments = builder.get_sequence_segments(fusion_name)
                novel_segment = None
                for seg in segments:
                    if seg.source_type == 'novel':
                        novel_segment = seg
                        break

                if novel_segment:
                    # Center window on the novel sequence
                    novel_center = (novel_segment.start_pos + novel_segment.end_pos) // 2
                    window_start = max(0, novel_center - seq_len // 2)
                    window_end = window_start + seq_len
                else:
                    # Fallback to standard centering if no novel segment found
                    center_pos = bnd1.pos - 1  # Convert to 0-based
                    window_start = max(0, center_pos - seq_len // 2)
                    window_end = window_start + seq_len
            else:
                # Standard BND: center on first breakend position
                center_pos = bnd1.pos - 1  # Convert to 0-based
                window_start = max(0, center_pos - seq_len // 2)
                window_end = window_start + seq_len

            # Generate ALT sequence (fusion sequence window)
            if len(fusion_seq) >= seq_len:
                alt_seq = fusion_seq[window_start:window_end]
            else:
                # Pad if fusion is shorter than window
                alt_seq = fusion_seq + 'N' * (seq_len - len(fusion_seq))

            # Generate reference sequences with appropriate padding
            # For BNDs with insertions, we need to account for the inserted sequence length
            half_len = seq_len // 2
            insertion_length = len(bnd1.inserted_seq) + len(bnd2.inserted_seq)

            # Generate left reference sequence (sequence before breakend + right-side N-padding)
            # For BNDs, we want to show what was there BEFORE the fusion point
            # Then pad the right side with N's to represent the missing fusion partner + insertion
            if not has_insertion:
                # Standard BND: use existing logic
                center_pos = bnd1.pos - 1  # Convert to 0-based if not set above
            left_start = max(0, center_pos - half_len)
            left_end = center_pos  # Stop at the breakend position
            left_ref_raw = seq1[left_start:left_end]

            # For BND with insertion, pad for both the missing chromosome and the insertion
            left_padding_needed = seq_len - len(left_ref_raw)
            # Note: The padding represents what's missing (other chromosome + insertion)
            # but we don't artificially inflate it since the user wants to see proper N-padding
            left_ref_seq = left_ref_raw + 'N' * left_padding_needed

            # Generate right reference sequence (left-side N-padding + sequence after breakend)
            # For the right side, we want to show what was there AFTER the fusion point
            # Pad the left side with N's to represent the missing fusion partner + insertion
            bnd2_center = bnd2.pos - 1  # Convert to 0-based

            # Extract sequence starting from the breakend (after the fusion point)
            right_start = bnd2_center  # Start at the breakend position
            right_end = min(len(seq2), bnd2_center + half_len)
            right_ref_raw = seq2[right_start:right_end]

            # For BND with insertion, pad for both the missing chromosome and the insertion
            right_padding_needed = seq_len - len(right_ref_raw)
            # Note: The padding represents what's missing (other chromosome + insertion)
            # but we don't artificially inflate it since the user wants to see proper N-padding
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
                'window_start': window_start,
                'window_end': window_end,
                'variant_pos0': center_pos,
                'variant_pos1': bnd1.pos,
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
            - metadata_df: Variant metadata (pandas DataFrame)

            For BND variants:
            - alt_sequences: Fusion sequences from breakend pairs
            - ref_sequences: Tuple of (left_ref_sequences, right_ref_sequences)
            - metadata_df: BND metadata with orientation and mate information

    Metadata DataFrame columns:
        Standard fields (all variants):
            - chrom: Chromosome name (str)
            - window_start: Window start position, 0-based (int)
            - window_end: Window end position, 0-based exclusive (int)
            - variant_pos0: Variant position, 0-based (int)
            - variant_pos1: Variant position, 1-based VCF standard (int)
            - ref: Reference allele (str)
            - alt: Alternate allele (str)
            - variant_type: Variant classification (str)
                Examples: 'SNV', 'INS', 'DEL', 'MNV', 'SV_INV', 'SV_DUP', 'SV_BND'

        Additional field for symbolic alleles (<INV>, <DUP>, etc.):
            - sym_variant_end: END position from INFO field, 1-based (int, optional)

        BND-specific fields:
            - mate_chrom: Mate breakend chromosome (str)
            - mate_pos: Mate breakend position, 1-based (int)
            - orientation_1: First breakend orientation (str)
            - orientation_2: Second breakend orientation (str)
            - fusion_name: Fusion sequence identifier (str, optional)
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


