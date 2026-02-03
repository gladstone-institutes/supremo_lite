"""
In-silico saturation mutagenesis functionality for supremo_lite.

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


def _dinucleotide_shuffle(sequence: str, random_state=None) -> str:
    """
    Shuffle a sequence while preserving nucleotide composition.

    Uses Fisher-Yates shuffle on sequence positions while keeping
    the first and last nucleotides fixed. This preserves single
    nucleotide frequencies.

    For biological controls, this is often sufficient as it disrupts
    motifs while maintaining GC content and base composition.

    Args:
        sequence: Input DNA sequence string (ACGT only)
        random_state: Optional numpy random state or seed for reproducibility

    Returns:
        Shuffled sequence with preserved nucleotide composition
    """
    if len(sequence) <= 2:
        return sequence

    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    seq = sequence.upper()

    # Convert to list for shuffling
    seq_list = list(seq)

    # Keep first and last nucleotide fixed, shuffle the middle
    middle = seq_list[1:-1]
    rng.shuffle(middle)

    # Reconstruct sequence
    result = [seq_list[0]] + middle + [seq_list[-1]]

    return "".join(result)


def _scramble_region(sequence: str, start: int, end: int, random_state=None) -> str:
    """
    Scramble a specific region within a sequence using dinucleotide shuffle.

    Args:
        sequence: Full sequence string
        start: Start position of region to scramble (0-based)
        end: End position of region to scramble (exclusive)
        random_state: Optional random state for reproducibility

    Returns:
        Sequence with the specified region scrambled
    """
    if start < 0 or end > len(sequence) or start >= end:
        raise ValueError(
            f"Invalid region [{start}, {end}) for sequence of length {len(sequence)}"
        )

    prefix = sequence[:start]
    region = sequence[start:end]
    suffix = sequence[end:]

    scrambled_region = _dinucleotide_shuffle(region, random_state)

    return prefix + scrambled_region + suffix


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


def get_sm_sequences(chrom, start, end, reference_fasta, encoder=None):
    """
    Generate sequences with all alternate nucleotides at every position (saturation mutagenesis).

    Args:
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (0-based, exclusive)
        reference_fasta: Reference genome object
        encoder: Optional custom encoding function. If provided, should accept a single
                sequence string and return encoded array with shape (4, L). Default: None

    Returns:
        Tuple of (reference one-hot, alt one-hot tensor, metadata DataFrame)
    """
    # Get the reference sequence
    ref_seq = reference_fasta[chrom][start:end]
    if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
        ref_seq = ref_seq.seq

    ref_1h = encode_seq(ref_seq, encoder)

    alt_seqs = []
    metadata = []

    # For each position, substitute with each alternate base
    for i in range(len(ref_seq)):
        ref_nt = ref_seq[i]
        for alt in sorted({"A", "C", "G", "T"} - {ref_nt.upper()}):
            # Create a clone and substitute the base
            if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
                alt_1h = ref_1h.clone()
                alt_1h[:, i] = torch.tensor(nt_to_1h[alt], dtype=alt_1h.dtype)
            else:
                alt_1h = ref_1h.copy()
                alt_1h[:, i] = nt_to_1h[alt]

            alt_seqs.append(alt_1h)
            metadata.append([chrom, start, end, i, ref_nt, alt])

    # Stack the alternate sequences
    if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
        alt_seqs_stacked = torch.stack(alt_seqs)
    else:
        alt_seqs_stacked = np.stack(alt_seqs)

    # Create a DataFrame for the metadata
    metadata_df = pd.DataFrame(
        metadata,
        columns=[
            "chrom",
            "window_start",
            "window_end",
            "variant_offset0",
            "ref",
            "alt",
        ],
    )

    return ref_1h, alt_seqs_stacked, metadata_df


def get_sm_subsequences(
    chrom,
    seq_len,
    reference_fasta,
    anchor=None,
    anchor_radius=None,
    bed_regions=None,
    encoder=None,
    auto_map_chromosomes=False,
):
    """
    Generate sequences with all alternate nucleotides at positions in specified regions
    (saturation mutagenesis).

    Supports two mutually exclusive approaches for defining mutation intervals:
    1. Anchor-based: Use anchor + anchor_radius to define a single centered region
    2. BED-based: Use bed_regions to define one or more arbitrary genomic regions

    In both cases, sequences of length seq_len are generated, centered on the mutation interval(s).

    Args:
        chrom: Chromosome name
        seq_len: Total sequence length for each window
        reference_fasta: Reference genome object
        anchor: Anchor position (0-based). Required when using anchor_radius.
               Must be provided together with anchor_radius.
               Mutually exclusive with bed_regions.
        anchor_radius: Number of bases to include on either side of the anchor for mutations.
                      Required when using anchor. Must be provided together with anchor.
                      Mutually exclusive with bed_regions.
        bed_regions: BED file path or DataFrame defining mutation intervals.
                    BED format: chrom, start, end (0-based, half-open intervals).
                    Each BED region defines positions to mutate, centered in a seq_len window.
                    Mutually exclusive with anchor + anchor_radius.
        encoder: Optional custom encoding function. If provided, should accept a single
                sequence string and return encoded array with shape (4, L). Default: None
        auto_map_chromosomes: Automatically map chromosome names between reference and BED file
                             when they don't match exactly (e.g., 'chr1' <-> '1', 'chrM' <-> 'MT').
                             Only applies when bed_regions is provided. Default: False.

    Returns:
        Tuple of (reference one-hot, alt one-hot tensor, metadata DataFrame)

    Raises:
        ValueError: If invalid parameter combinations are provided
        ChromosomeMismatchError: If auto_map_chromosomes=False and chromosome names in BED file
                                and reference don't match exactly (only when bed_regions is provided)

    Examples:
        # Approach 1: Anchor-based (single region)
        ref, alts, meta = get_sm_subsequences(
            chrom='chr1',
            seq_len=200,
            reference_fasta=ref,
            anchor=1050,
            anchor_radius=10  # Mutate positions 1040-1060 in a 200bp window
        )

        # Approach 2: BED-based (multiple regions)
        ref, alts, meta = get_sm_subsequences(
            chrom='chr1',
            seq_len=200,
            reference_fasta=ref,
            bed_regions='regions.bed'  # Each region centered in 200bp window
        )
    """
    # Validate parameter combinations
    has_anchor = anchor is not None
    has_anchor_radius = anchor_radius is not None
    has_bed = bed_regions is not None

    # Check for invalid combinations
    if (has_anchor or has_anchor_radius) and has_bed:
        raise ValueError(
            "Cannot use both (anchor + anchor_radius) and bed_regions. "
            "These are mutually exclusive approaches."
        )

    # Validate anchor approach
    if has_anchor or has_anchor_radius:
        if not (has_anchor and has_anchor_radius):
            raise ValueError(
                "anchor and anchor_radius must be provided together. "
                "Both are required when using the anchor-based approach."
            )
    elif not has_bed:
        # Neither approach was specified
        raise ValueError("Must provide either (anchor + anchor_radius) or bed_regions.")

    alt_seqs = []
    metadata = []

    # Handle the two approaches differently
    if has_anchor:
        # APPROACH 1: Anchor-based (single region)
        # Calculate sequence boundaries centered on anchor
        start = anchor - seq_len // 2
        end = start + seq_len

        # Get the reference sequence
        ref_seq = reference_fasta[chrom][start:end]
        if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
            ref_seq = ref_seq.seq

        assert (
            len(ref_seq) == seq_len
        ), f"Expected sequence length {seq_len}, got {len(ref_seq)}"

        ref_1h = encode_seq(ref_seq, encoder)

        # Calculate the range to mutate
        anchor_offset = anchor - start
        # Validate anchor_radius
        assert anchor_radius <= anchor_offset, "Anchor radius exceeds start of sequence"

        # Create set of positions to mutate (within anchor_radius of anchor)
        mut_start = anchor_offset - anchor_radius
        mut_end = anchor_offset + anchor_radius
        valid_positions = set(range(mut_start, mut_end))

        # Mutate positions
        for i in sorted(valid_positions):
            ref_nt = ref_seq[i]
            for alt in sorted({"A", "C", "G", "T"} - {ref_nt.upper()}):
                # Create a clone and substitute the base
                if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
                    alt_1h = ref_1h.clone()
                    alt_1h[:, i] = torch.tensor(nt_to_1h[alt], dtype=alt_1h.dtype)
                else:
                    alt_1h = ref_1h.copy()
                    alt_1h[:, i] = nt_to_1h[alt]

                alt_seqs.append(alt_1h)
                metadata.append([chrom, start, end, i, ref_nt, alt])

    else:
        # APPROACH 2: BED-based (multiple regions)
        # Each BED region gets its own seq_len window centered on it
        ref_1h = None  # Will be set for first region

        # Parse BED file/DataFrame
        bed_df = _read_bed_file(bed_regions)

        # Apply chromosome name matching
        ref_chroms = {chrom}
        bed_chroms = set(bed_df["chrom"].unique())

        mapping, unmatched = match_chromosomes_with_report(
            ref_chroms,
            bed_chroms,
            verbose=False,
            auto_map_chromosomes=auto_map_chromosomes,
        )

        if mapping:
            bed_df = apply_chromosome_mapping(bed_df, mapping)

        # Filter to current chromosome
        chrom_bed_regions = bed_df[bed_df["chrom"] == chrom].copy()

        if len(chrom_bed_regions) == 0:
            warnings.warn(
                f"No BED regions found for chromosome {chrom}. No mutagenesis will be performed."
            )
        else:
            # Process each BED region
            for _, bed_region in chrom_bed_regions.iterrows():
                region_start = bed_region["start"]
                region_end = bed_region["end"]
                region_center = (region_start + region_end) // 2

                # Calculate sequence window centered on this BED region
                window_start = region_center - seq_len // 2
                window_end = window_start + seq_len

                # Adjust window to stay within chromosome bounds
                chrom_obj = reference_fasta[chrom]
                chrom_len = (
                    len(chrom_obj)
                    if hasattr(chrom_obj, "__len__")
                    else len(chrom_obj.seq)
                )
                if window_start < 0:
                    window_start = 0
                    window_end = min(seq_len, chrom_len)
                elif window_end > chrom_len:
                    window_end = chrom_len
                    window_start = max(0, chrom_len - seq_len)

                # Get the reference sequence for this window
                region_seq = reference_fasta[chrom][window_start:window_end]
                if hasattr(region_seq, "seq"):
                    region_seq = region_seq.seq

                if len(region_seq) != seq_len:
                    warnings.warn(
                        f"Region {chrom}:{region_start}-{region_end} produces sequence of length "
                        f"{len(region_seq)} instead of {seq_len} (chromosome length: {chrom_len}). "
                        f"Skipping this region."
                    )
                    continue

                region_1h = encode_seq(region_seq, encoder)

                # Set ref_1h for the first valid region (for return value)
                if ref_1h is None:
                    ref_1h = region_1h

                # Calculate which positions to mutate (BED region relative to window)
                mut_start_rel = max(0, region_start - window_start)
                mut_end_rel = min(seq_len, region_end - window_start)

                # Check if BED region overlaps with the extracted window
                if mut_start_rel >= mut_end_rel:
                    warnings.warn(
                        f"BED region {chrom}:{region_start}-{region_end} is outside chromosome bounds "
                        f"(length: {chrom_len}). Skipping this region."
                    )
                    continue

                # Mutate positions within this BED region
                for i in range(mut_start_rel, mut_end_rel):
                    ref_nt = region_seq[i]
                    for alt in sorted({"A", "C", "G", "T"} - {ref_nt.upper()}):
                        # Create a clone and substitute the base
                        if TORCH_AVAILABLE and isinstance(region_1h, torch.Tensor):
                            alt_1h = region_1h.clone()
                            alt_1h[:, i] = torch.tensor(
                                nt_to_1h[alt], dtype=alt_1h.dtype
                            )
                        else:
                            alt_1h = region_1h.copy()
                            alt_1h[:, i] = nt_to_1h[alt]

                        alt_seqs.append(alt_1h)
                        metadata.append(
                            [chrom, window_start, window_end, i, ref_nt, alt]
                        )

        # If no regions were processed, create empty ref_1h
        if ref_1h is None:
            # Create a dummy empty sequence
            if TORCH_AVAILABLE:
                ref_1h = torch.zeros((4, seq_len), dtype=torch.float32)
            else:
                ref_1h = np.zeros((4, seq_len), dtype=np.float32)

    # Stack the alternate sequences
    if alt_seqs:
        if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
            alt_seqs_stacked = torch.stack(alt_seqs)
        else:
            alt_seqs_stacked = np.stack(alt_seqs)
    else:
        # No mutations generated (e.g., due to BED filtering)
        if TORCH_AVAILABLE and isinstance(ref_1h, torch.Tensor):
            alt_seqs_stacked = torch.empty((0, 4, seq_len), dtype=ref_1h.dtype)
        else:
            alt_seqs_stacked = np.empty((0, 4, seq_len), dtype=ref_1h.dtype)

    # Create a DataFrame for the metadata
    metadata_df = pd.DataFrame(
        metadata,
        columns=[
            "chrom",
            "window_start",
            "window_end",
            "variant_offset0",
            "ref",
            "alt",
        ],
    )

    return ref_1h, alt_seqs_stacked, metadata_df


def get_scrambled_subsequences(
    chrom: str,
    seq_len: int,
    reference_fasta,
    bed_regions: Union[str, pd.DataFrame],
    n_scrambles: int = 1,
    encoder=None,
    auto_map_chromosomes: bool = False,
    random_state=None,
):
    """
    Generate sequences with BED-defined regions scrambled using dinucleotide shuffle.

    This function creates control sequences where specific regions (defined by BED file)
    are scrambled while preserving dinucleotide frequencies. Useful for generating
    negative controls that maintain sequence composition properties.

    Args:
        chrom: Chromosome name
        seq_len: Total sequence length for each window
        reference_fasta: Reference genome object (pyfaidx.Fasta or dict-like)
        bed_regions: BED file path or DataFrame defining regions to scramble.
                    BED format: chrom, start, end (0-based, half-open intervals).
                    Each BED region is scrambled within its centered seq_len window.
        n_scrambles: Number of scrambled versions to generate per region (default: 1)
        encoder: Optional custom encoding function
        auto_map_chromosomes: Automatically map chromosome names between reference
                             and BED file (e.g., 'chr1' <-> '1'). Default: False.
        random_state: Random seed or numpy random generator for reproducibility.

    Returns:
        Tuple of (ref_seqs, scrambled_seqs, metadata):
        - ref_seqs: One-hot encoded reference sequences, shape (N, 4, seq_len)
        - scrambled_seqs: Scrambled sequences, shape (N * n_scrambles, 4, seq_len)
        - metadata: DataFrame with columns:
            - chrom: Chromosome name
            - window_start: Start of sequence window (0-based)
            - window_end: End of sequence window (0-based, exclusive)
            - scramble_start: Start of scrambled region within window (0-based)
            - scramble_end: End of scrambled region within window (0-based, exclusive)
            - scramble_idx: Index of this scramble (0 to n_scrambles-1)
            - original_seq: Original sequence in scrambled region
            - scrambled_seq: Scrambled sequence in that region

    Raises:
        ValueError: If bed_regions is not provided or has invalid format
    """
    if bed_regions is None:
        raise ValueError("bed_regions is required for get_scrambled_subsequences()")

    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Parse BED file
    bed_df = _read_bed_file(bed_regions)

    # Apply chromosome name matching
    ref_chroms = {chrom}
    bed_chroms = set(bed_df["chrom"].unique())

    mapping, unmatched = match_chromosomes_with_report(
        ref_chroms,
        bed_chroms,
        verbose=False,
        auto_map_chromosomes=auto_map_chromosomes,
    )

    if mapping:
        bed_df = apply_chromosome_mapping(bed_df, mapping)

    # Filter to target chromosome
    chrom_bed_regions = bed_df[bed_df["chrom"] == chrom].copy()

    if len(chrom_bed_regions) == 0:
        warnings.warn(
            f"No BED regions found for chromosome {chrom}. Returning empty results."
        )
        # Return empty results with correct structure
        if TORCH_AVAILABLE:
            empty_ref = torch.empty((0, 4, seq_len), dtype=torch.float32)
            empty_scrambled = torch.empty((0, 4, seq_len), dtype=torch.float32)
        else:
            empty_ref = np.empty((0, 4, seq_len), dtype=np.float32)
            empty_scrambled = np.empty((0, 4, seq_len), dtype=np.float32)

        empty_meta = pd.DataFrame(
            columns=[
                "chrom",
                "window_start",
                "window_end",
                "scramble_start",
                "scramble_end",
                "scramble_idx",
                "original_seq",
                "scrambled_seq",
            ]
        )
        return empty_ref, empty_scrambled, empty_meta

    ref_sequences = []
    scrambled_sequences = []
    metadata = []

    # Process each BED region
    for _, bed_region in chrom_bed_regions.iterrows():
        region_start = int(bed_region["start"])
        region_end = int(bed_region["end"])
        region_center = (region_start + region_end) // 2

        # Calculate sequence window centered on BED region
        window_start = region_center - seq_len // 2
        window_end = window_start + seq_len

        # Adjust window to stay within chromosome bounds
        chrom_obj = reference_fasta[chrom]
        chrom_len = len(chrom_obj) if hasattr(chrom_obj, "__len__") else len(chrom_obj)

        if window_start < 0:
            window_start = 0
            window_end = min(seq_len, chrom_len)
        elif window_end > chrom_len:
            window_end = chrom_len
            window_start = max(0, chrom_len - seq_len)

        # Get reference sequence
        ref_seq_obj = reference_fasta[chrom][window_start:window_end]
        if hasattr(ref_seq_obj, "seq"):
            ref_seq = str(ref_seq_obj.seq)
        else:
            ref_seq = str(ref_seq_obj)

        if len(ref_seq) != seq_len:
            warnings.warn(
                f"Region {chrom}:{region_start}-{region_end} produces sequence of length "
                f"{len(ref_seq)} instead of {seq_len}. Skipping."
            )
            continue

        # Calculate scramble region relative to window
        scramble_start_rel = max(0, region_start - window_start)
        scramble_end_rel = min(seq_len, region_end - window_start)

        if scramble_start_rel >= scramble_end_rel:
            warnings.warn(
                f"BED region {chrom}:{region_start}-{region_end} is outside window bounds. Skipping."
            )
            continue

        # Store reference sequence
        ref_1h = encode_seq(ref_seq, encoder)
        ref_sequences.append(ref_1h)

        # Get original region sequence for metadata
        original_region = ref_seq[scramble_start_rel:scramble_end_rel]

        # Generate n_scrambles scrambled versions
        for scramble_idx in range(n_scrambles):
            scrambled_seq = _scramble_region(
                ref_seq, scramble_start_rel, scramble_end_rel, random_state=rng
            )

            scrambled_1h = encode_seq(scrambled_seq, encoder)
            scrambled_sequences.append(scrambled_1h)

            scrambled_region = scrambled_seq[scramble_start_rel:scramble_end_rel]

            metadata.append(
                {
                    "chrom": chrom,
                    "window_start": window_start,
                    "window_end": window_end,
                    "scramble_start": scramble_start_rel,
                    "scramble_end": scramble_end_rel,
                    "scramble_idx": scramble_idx,
                    "original_seq": original_region,
                    "scrambled_seq": scrambled_region,
                }
            )

    # Stack sequences
    if ref_sequences:
        if TORCH_AVAILABLE and isinstance(ref_sequences[0], torch.Tensor):
            ref_stacked = torch.stack(ref_sequences)
            scrambled_stacked = torch.stack(scrambled_sequences)
        else:
            ref_stacked = np.stack(ref_sequences)
            scrambled_stacked = np.stack(scrambled_sequences)
    else:
        if TORCH_AVAILABLE:
            ref_stacked = torch.empty((0, 4, seq_len), dtype=torch.float32)
            scrambled_stacked = torch.empty((0, 4, seq_len), dtype=torch.float32)
        else:
            ref_stacked = np.empty((0, 4, seq_len), dtype=np.float32)
            scrambled_stacked = np.empty((0, 4, seq_len), dtype=np.float32)

    metadata_df = pd.DataFrame(metadata)

    return ref_stacked, scrambled_stacked, metadata_df
