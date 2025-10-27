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
                sequence string and return encoded array with shape (4, L).

    Returns:
        A tensor/array with shape (4, L) for a single sequence or (N, 4, L) for a list,
        where L is the sequence length and N is the number of sequences.

    Encoding scheme (default):
        'A' = [1, 0, 0, 0] (first channel)
        'C' = [0, 1, 0, 0] (second channel)
        'G' = [0, 0, 1, 0] (third channel)
        'T' = [0, 0, 0, 1] (fourth channel)
        'N' = [0, 0, 0, 0] (all channels zero)
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
            import warnings

            warnings.warn(
                "Fast encoding with brisket failed, falling back to slower NumPy implementation. "
                "This may impact performance for large sequences.",
                UserWarning,
                stacklevel=2,
            )
            encoded = np.array([nt_to_1h[nt] for nt in seq]).T
    else:
        # Original implementation
        encoded = np.array([nt_to_1h[nt] for nt in seq]).T

    if TORCH_AVAILABLE:
        return torch.from_numpy(encoded).float()
    return encoded


def decode_seq(seq_1h):
    """
    Convert a one-hot encoded tensor/array back to a nucleotide string.

    Args:
        seq_1h: A tensor/array with shape (4, L) or (N, 4, L)

    Returns:
        A string or list of strings of nucleotides
    """
    # Convert to numpy if it's a torch tensor
    if TORCH_AVAILABLE and isinstance(seq_1h, torch.Tensor):
        seq_1h = seq_1h.numpy()

    # Handle batch dimension if present
    if len(seq_1h.shape) == 3:
        return [decode_seq(s) for s in seq_1h]

    # Get the index of the maximum value along the channel dimension (first dimension)
    indices = seq_1h.argmax(axis=0)
    seq = nts[indices]

    return "".join(seq)


def rc(seq_1h):
    """
    Reverse complement a one-hot encoded tensor/array.

    Args:
        seq_1h: A tensor/array with shape (4, L) or (N, 4, L)

    Returns:
        The reverse complement with the same shape
    """
    if TORCH_AVAILABLE and isinstance(seq_1h, torch.Tensor):
        # Reverse channels for complement: [A, C, G, T] → [T, G, C, A]
        # Then flip the sequence dimension
        return seq_1h[..., [3, 2, 1, 0], :].flip(dims=[-1])
    # NumPy version: reverse channels and flip sequence dimension
    return np.flip(seq_1h[..., [3, 2, 1, 0], :], axis=-1)


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
