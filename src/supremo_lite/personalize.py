"""
Personalized sequence generation for supremo_lite.

This module provides functions for creating personalized genomes by applying
variants to a reference genome and generating sequence windows around variants.
"""

import bisect
import warnings
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pyfaidx import Fasta
from .variant_utils import read_vcf
from .sequence_utils import encode_seq



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
        idx = bisect.bisect_right(self.intervals, (pos, float('inf'))) - 1
        
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
    
    def __init__(self, sequence_str: str, variants_df: pd.DataFrame):
        """
        Initialize variant applicator for a single chromosome.
        
        Args:
            sequence_str: Reference sequence as string
            variants_df: DataFrame containing variants for this chromosome
        """
        self.sequence = bytearray(sequence_str.encode())  # Mutable sequence
        self.variants = variants_df.sort_values('pos').reset_index(drop=True)
        self.frozen_tracker = FrozenRegionTracker()
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
                warnings.warn(f"Cannot apply variant at {variant.pos}: {e}")
                self.skipped_count += 1
        
        stats = {
            'applied': self.applied_count,
            'skipped': self.skipped_count,
            'total': len(self.variants)
        }
        
        return self.sequence.decode(), stats
    
    def _apply_single_variant(self, variant: pd.Series) -> None:
        """
        Apply a single variant to the sequence.
        
        Args:
            variant: Series containing variant information (pos, ref, alt)
        """
        # 1. VALIDATION CHECKS
        if variant.alt.startswith('<'):
            raise ValueError(f"Symbolic variant not supported: {variant.alt}")
        
        if variant.alt == variant.ref:
            self.skipped_count += 1
            return  # Skip ref-only variants
        
        # Handle multiple ALT alleles - take first one
        alt_allele = variant.alt.split(',')[0]
        
        # 2. COORDINATE CALCULATION
        genomic_pos = variant.pos - 1  # Convert VCF 1-based to 0-based
        buffer_pos = genomic_pos + self.cumulative_offset
        
        # 3. FROZEN REGION CHECK
        ref_start = genomic_pos
        ref_end = genomic_pos + len(variant.ref) - 1
        
        if (self.frozen_tracker.is_frozen(ref_start) or 
            self.frozen_tracker.is_frozen(ref_end)):
            self.skipped_count += 1
            return  # Skip overlapping variants
        
        # 4. BOUNDS CHECK
        if buffer_pos < 0 or buffer_pos + len(variant.ref) > len(self.sequence):
            raise ValueError(f"Variant position {variant.pos} out of sequence bounds")
        
        # 5. REFERENCE VALIDATION
        expected_ref = self.sequence[buffer_pos:buffer_pos + len(variant.ref)].decode()
        if expected_ref.upper() != variant.ref.upper():
            raise ValueError(
                f"Reference mismatch at position {variant.pos}: "
                f"expected '{variant.ref}', found '{expected_ref}'"
            )
        
        # 6. SEQUENCE MODIFICATION
        self._modify_sequence(buffer_pos, variant.ref, alt_allele)
        
        # 7. UPDATE TRACKING
        length_diff = len(alt_allele) - len(variant.ref)
        self.cumulative_offset += length_diff
        self.frozen_tracker.add_range(ref_start, ref_end)
        self.applied_count += 1
    
    def _modify_sequence(self, pos: int, ref_allele: str, alt_allele: str) -> None:
        """
        Modify sequence at specified position with variant alleles.
        
        Args:
            pos: Buffer position (0-based)
            ref_allele: Reference allele sequence
            alt_allele: Alternate allele sequence
        """
        ref_len = len(ref_allele)
        alt_len = len(alt_allele)
        
        if alt_len == ref_len:
            # SNV: Direct substitution
            self.sequence[pos:pos + ref_len] = alt_allele.encode()
            
        elif alt_len < ref_len:
            # Deletion: Replace + remove extra bases
            self.sequence[pos:pos + alt_len] = alt_allele.encode()
            del self.sequence[pos + alt_len:pos + ref_len]
            
        else:
            # Insertion: Replace + insert extra bases
            self.sequence[pos:pos + ref_len] = alt_allele.encode()

def get_personal_genome(reference_fn, variants_fn, encode=True):
    """
    Create a personalized genome by applying variants to a reference genome.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        encode: Return sequences as one-hot encoded numpy arrays (default: True)

    Returns:
        A dictionary mapping chromosome names to personalized sequences (either
        as strings or one-hot encoded numpy arrays)
    """
    # Read variants if filename provided
    if isinstance(variants_fn, str):
        variants = read_vcf(variants_fn)
    else:
        variants = variants_fn

    # Sort variants by chromosome and position
    variants = variants.sort_values(["chrom", "pos"])

    # If the fasta is a file path, load it into memory here
    if isinstance(reference_fn, str) and os.path.isfile(reference_fn):
        reference = Fasta(reference_fn)
    else:
        reference = reference_fn  # Assume it's already a dictionary-like object

    # Get all chromosome names from the reference
    ref_chroms = list(reference.keys())
    
    # Build a dictionary of variants grouped by chromosome
    chrom_to_vars = {}
    missing_chroms = set()
    for chrom, group in variants.groupby("chrom"):
        if chrom in ref_chroms:
            chrom_to_vars[chrom] = group
        else:
            missing_chroms.add(chrom)
    
    # Warn about missing chromosomes
    if missing_chroms:
        warnings.warn(
            f"Chromosomes {', '.join(missing_chroms)} in variants not found in reference. Skipping."
        )

    # Initialize personalized genome
    personal_genome = {}

    # Process each chromosome in the reference
    for chrom in ref_chroms:
        ref_seq = str(reference[chrom])
        
        if chrom in chrom_to_vars:
            chrom_vars = chrom_to_vars[chrom]
            applicator = VariantApplicator(ref_seq, chrom_vars)
            personal_seq, stats = applicator.apply_variants()
            
            # Report statistics if any variants were processed
            if stats['total'] > 0:
                applied = stats['applied']
                skipped = stats['skipped']
                total = stats['total']
                if skipped > 0:
                    warnings.warn(
                        f"Chromosome {chrom}: {applied}/{total} variants applied, "
                        f"{skipped} skipped due to overlaps or errors"
                    )
            sequence = personal_seq
            # Pad sequence with Ns if shorter than reference
            ref_len=len(ref_seq)
            seq_len=len(sequence)
            if seq_len < ref_len:
                sequence += 'N' * (ref_len - seq_len)
        else:
            sequence = ref_seq

        # Apply encoding if requested
        personal_genome[chrom] = encode_seq(sequence) if encode else sequence

    return personal_genome

def get_personal_sequences(reference_fn, variants_fn, seq_len):
    """
    Create sequence windows centered on each variant position.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        seq_len: Length of the sequence window

    Returns:
        A list of tuples containing (chrom, start, end, sequence)
    """
    # Read variants if filename provided
    if isinstance(variants_fn, str):
        variants = read_vcf(variants_fn)
    else:
        variants = variants_fn

    sequences = []

    # For each variant, extract a sequence window
    for _, var in variants.iterrows():
        chrom = var["chrom"]
        pos = var["pos"]  # 1-based position

        # Calculate window boundaries
        window_start = max(0, pos - seq_len // 2)
        window_end = window_start + seq_len

        # Get reference sequence for this chromosome
        if hasattr(reference_fn, "__getitem__"):  # Dictionary-like
            try:
                ref_seq = reference_fn[chrom][window_start:window_end]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
            except IndexError:
                # Handle case where window extends beyond chromosome
                ref_seq = reference_fn[chrom][window_start:]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
                ref_seq += "N" * (seq_len - len(ref_seq))  # Pad with Ns
        else:  # Assume it's a file path
            # This is a simplified implementation
            ref_seq = "N" * seq_len  # Placeholder
            warnings.warn(
                "File-based sequence extraction not implemented yet. Returning N's."
            )

        sequences.append((chrom, window_start, window_end, ref_seq))

    return sequences


def get_pam_disrupting_personal_sequences(
    reference_fn, variants_fn, seq_len, max_pam_distance, pam_sequence="NGG"
):
    """
    Generate sequences for variants that disrupt PAM sites.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        seq_len: Length of sequence windows
        max_pam_distance: Maximum distance from variant to PAM site
        pam_sequence: PAM sequence pattern (default: 'NGG' for SpCas9)

    Returns:
        A dictionary of sequences with PAM sites, and variants that disrupt them
    """
    # Read variants if filename provided
    if isinstance(variants_fn, str):
        variants = read_vcf(variants_fn)
    else:
        variants = variants_fn

    # Filter variants that are near PAM sites
    pam_disrupting_variants = []
    pam_intact_sequences = []
    pam_disrupted_sequences = []

    # For each variant, check if it disrupts a PAM site
    for _, var in variants.iterrows():
        chrom = var["chrom"]
        pos = var["pos"]  # 1-based position
        ref = var["ref"]
        alt = var["alt"]

        # Calculate window boundaries
        window_start = max(0, pos - seq_len // 2)
        window_end = window_start + seq_len

        # Get reference sequence for this region
        if hasattr(reference_fn, "__getitem__"):  # Dictionary-like
            try:
                ref_seq = reference_fn[chrom][window_start:window_end]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
            except IndexError:
                # Handle case where window extends beyond chromosome
                ref_seq = reference_fn[chrom][window_start:]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
                ref_seq += "N" * (seq_len - len(ref_seq))  # Pad with Ns
        else:  # Assume it's a file path
            # This is a simplified implementation
            ref_seq = "N" * seq_len  # Placeholder
            warnings.warn(
                "File-based sequence extraction not implemented yet. Returning N's."
            )

        # Find PAM sites in the reference sequence
        pam_sites = []
        for i in range(len(ref_seq) - len(pam_sequence) + 1):
            potential_pam = ref_seq[i : i + len(pam_sequence)]
            # Check if the potential PAM matches the pattern
            if all(
                a == b or b == "N"
                for a, b in zip(potential_pam.upper(), pam_sequence.upper())
            ):
                pam_sites.append(i)

        # Filter PAM sites that are within max_pam_distance of the variant
        variant_pos_in_window = pos - window_start
        nearby_pam_sites = [
            p for p in pam_sites if abs(p - variant_pos_in_window) <= max_pam_distance
        ]

        if nearby_pam_sites:
            pam_disrupting_variants.append(var)

            # Create sequence with the variant but PAM intact
            variant_seq = (
                ref_seq[: variant_pos_in_window - 1]
                + alt
                + ref_seq[variant_pos_in_window - 1 + len(ref) :]
            )
            pam_intact_sequences.append((chrom, window_start, window_end, variant_seq))

            # Create sequences with PAM disrupted
            for pam_site in nearby_pam_sites:
                # Make a version where the PAM is disrupted
                disrupted_seq = (
                    variant_seq[:pam_site] + "NNN" + variant_seq[pam_site + 3 :]
                )  # Disrupt with NNN
                pam_disrupted_sequences.append(
                    (chrom, window_start, window_end, disrupted_seq)
                )

    return {
        "variants": pam_disrupting_variants,
        "pam_intact": pam_intact_sequences,
        "pam_disrupted": pam_disrupted_sequences,
    }