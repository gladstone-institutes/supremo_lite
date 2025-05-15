"""
In-silico mutagenesis functionality for supremo_lite.

This module provides functions for generating saturation mutagenesis sequences,
where each position in a sequence is systematically mutated.
"""

import pandas as pd
from .core import nt_to_1h, TORCH_AVAILABLE
from .sequence_utils import encode_seq

try:
    import torch
except ImportError:
    pass  # Already handled in core


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


def get_sm_subsequences(chrom, anchor, anchor_radius, seq_len, reference_fasta):
    """
    Generate sequences with all alternate nucleotides at positions around an anchor
    (saturation mutagenesis).

    Args:
        chrom: Chromosome name
        anchor: Anchor position (0-based)
        anchor_radius: Number of bases to include on either side of the anchor
        seq_len: Total sequence length
        reference_fasta: Reference genome object

    Returns:
        Tuple of (reference one-hot, alt one-hot tensor, metadata DataFrame)
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

    # For each position around the anchor, substitute with each alternate base
    for i in range(anchor_offset - anchor_radius, anchor_offset + anchor_radius):
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
