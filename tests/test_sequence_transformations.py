"""
Tests for the sequence transformation functions in supremo_lite.

This file tests the encoding, decoding, and reverse complement functions.
"""

import unittest
import numpy as np
import supremo_lite as sl
from supremo_lite.core import BRISKET_AVAILABLE

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSequenceTransformations(unittest.TestCase):

    def test_encode_single_seq(self):
        """Test encoding a single sequence."""
        seq = "ACGT"
        encoded = sl.encode_seq(seq)

        # Check shape
        self.assertEqual(encoded.shape, (4, 4))

        # Check values for specific bases
        if TORCH_AVAILABLE:
            # Convert tensor to numpy for comparison
            encoded_np = encoded.numpy()
        else:
            encoded_np = encoded

        # A = [1, 0, 0, 0]
        self.assertTrue(np.array_equal(encoded_np[:, 0], [1, 0, 0, 0]))
        # C = [0, 1, 0, 0]
        self.assertTrue(np.array_equal(encoded_np[:, 1], [0, 1, 0, 0]))
        # G = [0, 0, 1, 0]
        self.assertTrue(np.array_equal(encoded_np[:, 2], [0, 0, 1, 0]))
        # T = [0, 0, 0, 1]
        self.assertTrue(np.array_equal(encoded_np[:, 3], [0, 0, 0, 1]))

    def test_encode_batch(self):
        """Test encoding a batch of sequences."""
        seqs = ["ACGT", "TGCA"]
        encoded = sl.encode_seq(seqs)

        # Check shape
        self.assertEqual(encoded.shape, (2, 4, 4))

        # Check specific values for the first sequence
        if TORCH_AVAILABLE:
            # Convert tensor to numpy for comparison
            encoded_np = encoded.numpy()
        else:
            encoded_np = encoded

        # First sequence: A = [1, 0, 0, 0]
        self.assertTrue(np.array_equal(encoded_np[0, :, 0], [1, 0, 0, 0]))
        # Second sequence: T = [0, 0, 0, 1]
        self.assertTrue(np.array_equal(encoded_np[1, :, 0], [0, 0, 0, 1]))

    def test_decode_single_seq(self):
        """Test decoding a single sequence."""
        seq = "ACGT"
        encoded = sl.encode_seq(seq)
        decoded = sl.decode_seq(encoded)

        self.assertEqual(decoded, seq)

    def test_decode_batch(self):
        """Test decoding a batch of sequences."""
        seqs = ["ACGT", "TGCA", "GATC"]
        encoded = sl.encode_seq(seqs)
        decoded = sl.decode_seq(encoded)

        self.assertEqual(len(decoded), len(seqs))
        for i, seq in enumerate(seqs):
            self.assertEqual(decoded[i], seq)

    def test_rc_str(self):
        """Test reverse complement of a string."""
        seq = "ACGTACGT"
        rc = sl.rc_str(seq)

        # Manual reverse complement calculation
        expected = seq[::-1].translate(str.maketrans("ACGT", "TGCA"))
        self.assertEqual(rc, expected)

        # Try with lowercase
        seq_lower = "acgtacgt"
        rc_lower = sl.rc_str(seq_lower)
        expected_lower = seq_lower[::-1].translate(str.maketrans("acgt", "tgca"))
        self.assertEqual(rc_lower, expected_lower)

    def test_rc_onehot(self):
        """Test reverse complement of one-hot encoded sequence."""
        seq = "ACGT"
        encoded = sl.encode_seq(seq)
        rc_encoded = sl.rc(encoded)
        rc_decoded = sl.decode_seq(rc_encoded)

        # Should match the string reverse complement
        expected = sl.rc_str(seq)
        self.assertEqual(rc_decoded, expected)

    def test_ambiguous_bases(self):
        """Test handling of ambiguous bases."""
        seq = "ACGTN"

        encoded = sl.encode_seq(seq)

        # Get N encoding
        if TORCH_AVAILABLE:
            n_encoding = encoded[:, -1].numpy()
        else:
            n_encoding = encoded[:, -1]

        # Both implementations now encode ambiguous bases as [0,0,0,0]
        self.assertTrue(np.allclose(n_encoding, [0, 0, 0, 0]))

        # Check decoding - N becomes 'A' (first in argmax of [0,0,0,0])
        decoded = sl.decode_seq(encoded)
        self.assertEqual(len(decoded), 5)
        self.assertTrue(decoded[:4] == "ACGT")
        self.assertEqual(decoded[-1], "A")


if __name__ == "__main__":
    unittest.main()
