"""
Sequence transformation utilities for supremo_lite.

This module provides functions for encoding, decoding, and manipulating
DNA sequences.
"""

import numpy as np
from .core import nt_to_1h, nts, TORCH_AVAILABLE, BRISKET_AVAILABLE

try:
    import torch
except ImportError:
    pass  # Already handled in core module

if BRISKET_AVAILABLE:
    try:
        from brisket import encode_seq as brisket_encode_seq
    except ImportError:
        pass  # Already handled in core module


def encode_seq(seq, encoder=None):
    """
    Convert a nucleotide string to a one-hot encoded tensor/array.

    Args:
        seq: A string of nucleotides or a list of such strings
        encoder: Optional custom encoding function. If provided, should accept a single
                sequence string and return encoded array with shape (L, 4).

    Returns:
        A tensor/array with shape (L, 4) for a single sequence or (N, L, 4) for a list,
        where L is the sequence length and N is the number of sequences.

    Encoding scheme (default):
        'A' = [1, 0, 0, 0]
        'C' = [0, 1, 0, 0]
        'G' = [0, 0, 1, 0]
        'T' = [0, 0, 0, 1]
        'N' = [0, 0, 0, 0]
    """
    if isinstance(seq, list):
        # For a list of sequences, encode each separately and stack
        encoded = np.stack([encode_seq(s, encoder) for s in seq])
        if TORCH_AVAILABLE:
            return torch.from_numpy(encoded).float()
        return encoded

    # Use custom encoder if provided
    if encoder is not None:
        encoded = encoder(seq)
        if TORCH_AVAILABLE:
            return torch.from_numpy(encoded).float()
        return encoded

    # For a single sequence, use brisket if available for performance, otherwise fallback
    if BRISKET_AVAILABLE:
        try:
            # Use brisket for fast encoding
            encoded = brisket_encode_seq(seq.upper())
            
        except Exception:
            # Fallback to original implementation if brisket fails
            # TODO: warn user
            encoded = np.array([nt_to_1h[nt] for nt in seq])
    else:
        # Original implementation
        encoded = np.array([nt_to_1h[nt] for nt in seq])

    if TORCH_AVAILABLE:
        return torch.from_numpy(encoded).float()
    return encoded


def decode_seq(seq_1h):
    """
    Convert a one-hot encoded tensor/array back to a nucleotide string.

    Args:
        seq_1h: A tensor/array with shape (L, 4) or (N, L, 4)

    Returns:
        A string or list of strings of nucleotides
    """
    # Convert to numpy if it's a torch tensor
    if TORCH_AVAILABLE and isinstance(seq_1h, torch.Tensor):
        seq_1h = seq_1h.numpy()

    # Handle batch dimension if present
    if len(seq_1h.shape) == 3:
        return [decode_seq(s) for s in seq_1h]

    # Get the index of the maximum value along the last dimension
    indices = seq_1h.argmax(axis=-1)
    seq = nts[indices]

    return "".join(seq)


def rc(seq_1h):
    """
    Reverse complement a one-hot encoded tensor/array.

    Args:
        seq_1h: A tensor/array with shape (L, 4) or (N, L, 4)

    Returns:
        The reverse complement with the same shape
    """
    if TORCH_AVAILABLE and isinstance(seq_1h, torch.Tensor):
        return torch.flip(seq_1h, dims=[-2, -1])
    return np.flip(seq_1h, axis=(0, 1))


def rc_str(seq):
    """
    Reverse complement a nucleotide string.

    Args:
        seq: A string of nucleotides

    Returns:
        The reverse complement string
    """
    t = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(t)[::-1]
