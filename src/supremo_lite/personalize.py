"""
Personalized sequence generation for supremo_lite.

This module provides functions for creating personalized genomes by applying
variants to a reference genome and generating sequence windows around variants.
"""

import bisect
import warnings
import os
from typing import Dict, List, Tuple, Union
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

    def __init__(self, sequence_str: str, variants_df: pd.DataFrame, frozen_tracker: FrozenRegionTracker = None):
        """
        Initialize variant applicator for a single chromosome.

        Args:
            sequence_str: Reference sequence as string
            variants_df: DataFrame containing variants for this chromosome
            frozen_tracker: Optional existing FrozenRegionTracker to preserve overlap state across chunks
        """
        self.sequence = bytearray(sequence_str.encode())  # Mutable sequence
        self.variants = variants_df.sort_values("pos1").reset_index(drop=True)
        self.frozen_tracker = frozen_tracker if frozen_tracker is not None else FrozenRegionTracker()
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
        
        # Define supported and unsupported variant types
        supported_types = {'SNV', 'MNV', 'INS', 'DEL', 'complex'}
        unsupported_types = {
            'SV_INV', 'SV_DUP', 'SV_DEL', 'SV_INS', 'SV_CNV', 'SV_BND', 
            'missing', 'unknown'
        }
        
        # Skip unsupported variant types with warning
        if variant_type in unsupported_types:
            warnings.warn(
                f"Skipping unsupported variant type '{variant_type}' at position {variant.pos1}. "
                f"Supported types are: {', '.join(sorted(supported_types))}"
            )
            self.skipped_count += 1
            return
            
        # Skip unknown variant types with warning
        if variant_type not in supported_types:
            warnings.warn(
                f"Skipping unknown variant type '{variant_type}' at position {variant.pos1}. "
                f"Supported types are: {', '.join(sorted(supported_types))}"
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


def get_personal_genome(reference_fn, variants_fn, encode=True, n_chunks=1, verbose=False):
    """
    Create a personalized genome by applying variants to a reference genome.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        encode: Return sequences as one-hot encoded arrays (default: True)
        n_chunks: Number of chunks to split variants into for processing (default: 1)
        verbose: Print progress information (default: False)

    Returns:
        If encode=True: A dictionary mapping chromosome names to encoded tensors/arrays
        If encode=False: A dictionary mapping chromosome names to sequence strings
    """
    # Load all variants once with classification
    variants = _load_variants(variants_fn)
    reference = _load_reference(reference_fn)

    # Sort variants by chromosome and position for efficient processing
    variants = variants.sort_values(["chrom", "pos1"])

    # Apply chromosome name matching once
    ref_chroms = set(reference.keys())
    vcf_chroms = set(variants["chrom"].unique())
    
    if verbose:
        print(f"🧬 Processing {len(variants):,} variants across {len(vcf_chroms)} chromosomes")

    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=verbose
    )

    # Apply chromosome name mapping to all variants
    if mapping:
        variants = apply_chromosome_mapping(variants, mapping)

    # Initialize personalized genome
    personal_genome = {}
    total_processed = 0

    # Process each chromosome
    for chrom, chrom_variants in variants.groupby("chrom"):
        if chrom not in reference:
            if verbose:
                print(f"⚠️  Skipping {chrom}: not found in reference")
            continue

        ref_seq = str(reference[chrom])
        
        if verbose:
            print(f"🔄 Processing chromosome {chrom}: {len(chrom_variants):,} variants")

        if n_chunks == 1:
            # Process all variants at once
            applicator = VariantApplicator(ref_seq, chrom_variants)
            personal_seq, stats = applicator.apply_variants()
            
            if verbose and stats["total"] > 0:
                print(f"  ✅ Applied {stats['applied']}/{stats['total']} variants ({stats['skipped']} skipped)")

        else:
            # Process in chunks with shared FrozenRegionTracker
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
                
                # Apply variants with shared FrozenRegionTracker
                applicator = VariantApplicator(current_sequence, chunk_df, shared_frozen_tracker)
                current_sequence, stats = applicator.apply_variants()

                total_applied += stats["applied"]
                total_skipped += stats["skipped"]

                if verbose:
                    print(f"    ✅ Chunk {i+1}: {stats['applied']}/{stats['total']} variants applied")

            personal_seq = current_sequence
            
            if verbose:
                print(f"  🎯 Total: {total_applied}/{len(chrom_variants)} variants applied ({total_skipped} skipped)")

        # Store result
        personal_genome[chrom] = encode_seq(personal_seq) if encode else personal_seq
        total_processed += len(chrom_variants)

    # Add chromosomes with no variants
    for chrom in ref_chroms:
        if chrom not in personal_genome:
            ref_seq = str(reference[chrom])
            personal_genome[chrom] = encode_seq(ref_seq) if encode else ref_seq

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
        })
    
    return pd.DataFrame(metadata)


def get_alt_sequences(reference_fn, variants_fn, seq_len, encode=True, n_chunks=1):
    """
    Create sequence windows centered on each variant position with variants applied.

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
    """
    # Load reference and variants
    reference = _load_reference(reference_fn)

    # For chromosome matching, we need to load all variants first to get chromosome names
    all_variants = _load_variants(variants_fn)
    ref_chroms = set(reference.keys())
    vcf_chroms = set(all_variants["chrom"].unique())

    # Use chromosome matching to handle name mismatches
    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=True
    )

    # Apply chromosome name mapping to variants
    if mapping:
        all_variants = apply_chromosome_mapping(all_variants, mapping)

    # Split DataFrame into chunks using numpy array_split
    # n_chunks parameter is used directly
    indices = np.array_split(np.arange(len(all_variants)), n_chunks)
    variant_chunks = (
        all_variants.iloc[chunk_indices].reset_index(drop=True)
        for chunk_indices in indices
        if len(chunk_indices) > 0
    )

    # Process each chunk using vectorized operations
    for chunk_variants in variant_chunks:
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
                    sequences.append(encode_seq(window_seq))
                else:
                    sequences.append(
                        (
                            chrom,
                            max(0, genomic_pos - half_len),
                            max(0, genomic_pos - half_len) + seq_len,
                            window_seq,
                        )
                    )


        # Yield chunk results
        if encode and sequences:
            if TORCH_AVAILABLE:
                sequences_result = torch.stack(sequences)
            else:
                sequences_result = np.stack(sequences)
        else:
            sequences_result = sequences

        # Always return metadata as tuple
        yield (sequences_result, metadata_df)


def get_ref_sequences(reference_fn, variants_fn, seq_len, encode=True, n_chunks=1):
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
    """
    # Load reference and variants
    reference = _load_reference(reference_fn)

    # For chromosome matching, we need to load all variants first to get chromosome names
    all_variants = _load_variants(variants_fn)
    ref_chroms = set(reference.keys())
    vcf_chroms = set(all_variants["chrom"].unique())

    # Use chromosome matching to handle name mismatches
    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms, vcf_chroms, verbose=True
    )

    # Apply chromosome name mapping to variants
    if mapping:
        all_variants = apply_chromosome_mapping(all_variants, mapping)

    # Split DataFrame into chunks using numpy array_split
    # n_chunks parameter is used directly
    indices = np.array_split(np.arange(len(all_variants)), n_chunks)
    variant_chunks = (
        all_variants.iloc[chunk_indices].reset_index(drop=True)
        for chunk_indices in indices
        if len(chunk_indices) > 0
    )

    # Process each chunk using vectorized operations
    for chunk_variants in variant_chunks:
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
                    sequences.append(encode_seq(window_seq))
                else:
                    sequences.append(
                        (
                            chrom,
                            max(0, genomic_pos - half_len),
                            max(0, genomic_pos - half_len) + seq_len,
                            window_seq,
                        )
                    )


        # Yield chunk results
        if encode and sequences:
            if TORCH_AVAILABLE:
                sequences_result = torch.stack(sequences)
            else:
                sequences_result = np.stack(sequences)
        else:
            sequences_result = sequences

        # Always return metadata as tuple
        yield (sequences_result, metadata_df)


def get_alt_ref_sequences(
    reference_fn, variants_fn, seq_len, encode=True, n_chunks=1
):
    """
    Create both reference and variant sequence windows for alt/ref ratio calculations.

    This wrapper function calls both get_ref_sequences and get_alt_sequences to return
    matching pairs of reference and variant sequences for computing ratios.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to VCF file (string) or DataFrame with variant data.
                    For DataFrames, position column can be 'pos', 'pos1', or assumes second column is position.
        seq_len: Length of the sequence window
        encode: Return sequences as one-hot encoded numpy arrays (default: True)
        n_chunks: Number of chunks to split variants into (default: 1)

    Yields:
        Tuple containing (alt_sequences, ref_sequences, metadata_df):
            - alt_sequences: Variant sequences (same format as get_alt_sequences)
            - ref_sequences: Reference sequences (same format as get_ref_sequences) 
            - metadata_df: Enhanced DataFrame with comprehensive variant metadata including columns:
                          'chrom', 'start', 'end', 'variant_pos0', 'variant_pos1', 'variant_offset',
                          'ref', 'alt', 'variant_type', 'ref_length', 'alt_length', 
                          'effective_variant_start', 'effective_variant_end', 
                          'position_offset_upstream', 'position_offset_downstream', 
                          'structural_variant_info' 
    """
    # Get generators for both reference and variant sequences
    # These already handle variant loading, chromosome matching, and chunking
    ref_gen = get_ref_sequences(reference_fn, variants_fn, seq_len, encode, n_chunks)
    alt_gen = get_alt_sequences(reference_fn, variants_fn, seq_len, encode, n_chunks)

    # Process chunks from both generators
    for (ref_chunk, ref_metadata), (alt_chunk, alt_metadata) in zip(ref_gen, alt_gen):
        # Use the rich metadata from ref_sequences (alt_metadata is identical)
        # Both now contain standardized metadata with all required columns
        yield (alt_chunk, ref_chunk, ref_metadata)


def get_pam_disrupting_personal_sequences(
    reference_fn,
    variants_fn,
    seq_len,
    max_pam_distance,
    pam_sequence="NGG",
    encode=True,
    n_chunks=1,
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
            final_intact = encode_seq(modified_window) if encode else modified_window
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

                final_disrupted = encode_seq(disrupted_seq) if encode else disrupted_seq
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


