"""
In-silico mutagenesis functionality for supremo_lite.

This module provides functions for generating saturation mutagenesis sequences,
where each position in a sequence is systematically mutated.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Union
from .core import nt_to_1h, TORCH_AVAILABLE
from .sequence_utils import encode_seq
from .chromosome_utils import match_chromosomes_with_report, apply_chromosome_mapping

try:
    import torch
except ImportError:
    pass  # Already handled in core


def _read_bed_file(bed_regions: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Read BED file or validate BED DataFrame format.

    Args:
        bed_regions: Path to BED file or DataFrame with BED format

    Returns:
        DataFrame with columns: chrom, start, end

    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file path doesn't exist
    """
    if isinstance(bed_regions, str):
        # Read BED file
        try:
            bed_df = pd.read_csv(
                bed_regions,
                sep="\t",
                header=None,
                comment="#",
                usecols=[0, 1, 2],  # Only read first 3 columns
                names=["chrom", "start", "end"],
            )
        except Exception as e:
            raise ValueError(f"Error reading BED file '{bed_regions}': {e}")

    elif isinstance(bed_regions, pd.DataFrame):
        bed_df = bed_regions.copy()

        # Validate required columns exist
        required_cols = ["chrom", "start", "end"]
        if not all(col in bed_df.columns for col in required_cols):
            available_cols = list(bed_df.columns)
            raise ValueError(
                f"BED DataFrame must contain columns {required_cols}. "
                f"Found columns: {available_cols}"
            )

        # Select only required columns
        bed_df = bed_df[required_cols].copy()

    else:
        raise ValueError(
            f"bed_regions must be a file path (str) or DataFrame, got {type(bed_regions)}"
        )

    # Validate data types and values
    try:
        bed_df["start"] = pd.to_numeric(bed_df["start"], errors="coerce")
        bed_df["end"] = pd.to_numeric(bed_df["end"], errors="coerce")
    except Exception as e:
        raise ValueError(f"Invalid BED coordinates: {e}")

    # Check for invalid coordinates
    invalid_coords = bed_df["start"] >= bed_df["end"]
    if invalid_coords.any():
        n_invalid = invalid_coords.sum()
        warnings.warn(
            f"Found {n_invalid} BED regions with invalid coordinates (start >= end). These will be removed."
        )
        bed_df = bed_df[~invalid_coords].reset_index(drop=True)

    # Check for negative coordinates
    negative_coords = (bed_df["start"] < 0) | (bed_df["end"] < 0)
    if negative_coords.any():
        n_negative = negative_coords.sum()
        warnings.warn(
            f"Found {n_negative} BED regions with negative coordinates. These will be removed."
        )
        bed_df = bed_df[~negative_coords].reset_index(drop=True)

    if len(bed_df) == 0:
        raise ValueError("No valid BED regions found after filtering")

    return bed_df


def get_sm_sequences(chrom, start, end, reference_fasta):
    """
    Generate sequences with all alternate nucleotides at every position (saturation mutagenesis).

    Args:
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (0-based, exclusive)
        reference_fasta: Reference genome object

    Returns:
        Tuple of (reference one-hot, alt one-hot tensor, metadata DataFrame)
    """
    # Get the reference sequence
    ref_seq = reference_fasta[chrom][start:end]
    if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
        ref_seq = ref_seq.seq

    ref_1h = encode_seq(ref_seq)

    alt_seqs = []
    metadata = []

    # For each position, substitute with each alternate base
    for i in range(len(ref_seq)):
        ref_nt = ref_seq[i]
        for alt in sorted({"A", "C", "G", "T"} - {ref_nt.upper()}):
            # Create a clone and substitute the base
            if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
                alt_1h = ref_1h.clone()
                alt_1h[i] = torch.tensor(nt_to_1h[alt])
            else:
                alt_1h = ref_1h.copy()
                alt_1h[i] = nt_to_1h[alt]

            alt_seqs.append(alt_1h)
            metadata.append([chrom, start, end, i, ref_nt, alt])

    # Stack the alternate sequences
    if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
        alt_seqs_stacked = torch.stack(alt_seqs)
    else:
        alt_seqs_stacked = np.stack(alt_seqs)

    # Create a DataFrame for the metadata
    metadata_df = pd.DataFrame(
        metadata, columns=["chrom", "start", "end", "offset", "ref", "alt"]
    )

    return ref_1h, alt_seqs_stacked, metadata_df


def get_sm_subsequences(
    chrom, anchor, anchor_radius, seq_len, reference_fasta, bed_regions=None
):
    """
    Generate sequences with all alternate nucleotides at positions around an anchor
    (saturation mutagenesis).

    Args:
        chrom: Chromosome name
        anchor: Anchor position (0-based)
        anchor_radius: Number of bases to include on either side of the anchor
        seq_len: Total sequence length
        reference_fasta: Reference genome object
        bed_regions: Optional BED file path or DataFrame to limit mutagenesis to specific regions.
                    BED format: chrom, start, end (0-based, half-open intervals).
                    If provided, only positions within BED regions will be mutated.

    Returns:
        Tuple of (reference one-hot, alt one-hot tensor, metadata DataFrame)

    Note:
        When bed_regions is provided, chromosome name matching is applied to handle
        naming differences between the reference and BED file (e.g., 'chr1' vs '1').
    """
    # Calculate sequence boundaries
    start = anchor - seq_len // 2
    end = start + seq_len

    # Get the reference sequence
    ref_seq = reference_fasta[chrom][start:end]
    if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
        ref_seq = ref_seq.seq

    assert (
        len(ref_seq) == seq_len
    ), f"Expected sequence length {seq_len}, got {len(ref_seq)}"

    ref_1h = encode_seq(ref_seq)

    alt_seqs = []
    metadata = []

    # Calculate the range to mutate
    anchor_offset = anchor - start
    assert anchor_radius <= anchor_offset, "Anchor radius exceeds start of sequence"

    # Process BED regions if provided
    valid_positions = None
    if bed_regions is not None:
        # Parse BED file/DataFrame
        bed_df = _read_bed_file(bed_regions)

        # Apply chromosome name matching
        ref_chroms = {chrom}  # Current chromosome from reference
        bed_chroms = set(bed_df["chrom"].unique())

        # Use chromosome matching to handle name mismatches
        mapping, unmatched = match_chromosomes_with_report(
            ref_chroms, bed_chroms, verbose=False
        )

        if mapping:
            bed_df = apply_chromosome_mapping(bed_df, mapping)

        # Filter to current chromosome
        chrom_bed_regions = bed_df[bed_df["chrom"] == chrom].copy()

        if len(chrom_bed_regions) == 0:
            warnings.warn(
                f"No BED regions found for chromosome {chrom}. No mutagenesis will be performed."
            )
            valid_positions = set()  # Empty set means no valid positions
        else:
            # Calculate intersection between mutagenesis window and BED regions
            mut_start = anchor_offset - anchor_radius
            mut_end = anchor_offset + anchor_radius

            valid_positions = set()
            for _, bed_region in chrom_bed_regions.iterrows():
                # Convert to sequence-relative coordinates
                bed_start_rel = max(0, bed_region["start"] - start)
                bed_end_rel = min(seq_len, bed_region["end"] - start)

                # Find intersection with mutagenesis window
                intersect_start = max(mut_start, bed_start_rel)
                intersect_end = min(mut_end, bed_end_rel)

                if intersect_start < intersect_end:
                    # Add positions in intersection to valid set
                    valid_positions.update(
                        range(int(intersect_start), int(intersect_end))
                    )

            if not valid_positions:
                warnings.warn(
                    f"No BED regions overlap with mutagenesis window "
                    f"[{anchor - anchor_radius}:{anchor + anchor_radius}] on {chrom}. "
                    f"No mutagenesis will be performed."
                )

    # For each position around the anchor, substitute with each alternate base
    for i in range(anchor_offset - anchor_radius, anchor_offset + anchor_radius):
        # Skip position if BED regions are provided and this position is not valid
        if valid_positions is not None and i not in valid_positions:
            continue
        ref_nt = ref_seq[i]
        for alt in sorted({"A", "C", "G", "T"} - {ref_nt.upper()}):
            # Create a clone and substitute the base
            if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
                alt_1h = ref_1h.clone()
                alt_1h[i] = torch.tensor(nt_to_1h[alt])
            else:
                alt_1h = ref_1h.copy()
                alt_1h[i] = nt_to_1h[alt]

            alt_seqs.append(alt_1h)
            metadata.append([chrom, start, end, i, ref_nt, alt])

    # Stack the alternate sequences
    if alt_seqs:
        if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
            alt_seqs_stacked = torch.stack(alt_seqs)
        else:
            alt_seqs_stacked = np.stack(alt_seqs)
    else:
        # No mutations generated (e.g., due to BED filtering)
        if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
            alt_seqs_stacked = torch.empty((0, seq_len, 4), dtype=ref_1h.dtype)
        else:
            alt_seqs_stacked = np.empty((0, seq_len, 4), dtype=ref_1h.dtype)

    # Create a DataFrame for the metadata
    metadata_df = pd.DataFrame(
        metadata, columns=["chrom", "start", "end", "offset", "ref", "alt"]
    )

    return ref_1h, alt_seqs_stacked, metadata_df
