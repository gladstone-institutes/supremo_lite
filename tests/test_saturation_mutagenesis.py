"""
Tests for the in-silico mutagenesis functions in supremo_lite.

This file tests the saturation mutagenesis functions.
"""

import unittest
import numpy as np
import supremo_lite as sl

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MockSequence:
    """Mock sequence class for testing."""

    def __init__(self, seq):
        self.seq = seq


class MockChromosome:
    """Mock chromosome class for testing."""

    def __init__(self, seq="ACGT" * 25):
        self.sequence = seq

    def __getitem__(self, slice_obj):
        start = slice_obj.start if slice_obj.start is not None else 0
        stop = slice_obj.stop if slice_obj.stop is not None else len(self.sequence)

        # Return the sliced sequence
        return MockSequence(self.sequence[start:stop])


class MockReference:
    """Mock reference genome for testing."""

    def __init__(self):
        self.chromosomes = {
            "chr1": MockChromosome(),
            "chr2": MockChromosome("TGCA" * 25),
        }

    def __getitem__(self, chrom):
        return self.chromosomes[chrom]


class TestSaturationMutagenesis(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        self.reference = MockReference()

    def test_get_sm_sequences(self):
        """Test generating saturation mutagenesis sequences."""
        chrom = "chr1"
        start = 0
        end = 10  # ACGTACGTAC

        ref_1h, alt_seqs, metadata = sl.get_sm_sequences(
            chrom, start, end, self.reference
        )

        # Check reference shape
        self.assertEqual(ref_1h.shape, (10, 4))  # 10 bases, 4 nucleotides

        # Check alternative sequences - should have 3 alternatives for each position
        # 10 positions * 3 alternatives = 30 sequences
        self.assertEqual(len(alt_seqs), 30)

        # Verify metadata
        self.assertEqual(len(metadata), 30)
        self.assertEqual(
            list(metadata.columns), ["chrom", "start", "end", "offset", "ref", "alt"]
        )

        # Check each position gets all 3 alternatives
        pos_counts = metadata["offset"].value_counts()
        self.assertEqual(len(pos_counts), 10)  # all 10 positions should be present
        self.assertTrue(
            all(count == 3 for count in pos_counts)
        )  # each should have 3 alternatives

        # Check that for each reference base, we get the right alternatives
        for i, ref_base in enumerate("ACGTACGTAC"):
            # Get all rows for this position
            pos_rows = metadata[metadata["offset"] == i]

            # Check that the reference base is correct
            self.assertEqual(pos_rows["ref"].iloc[0], ref_base)

            # Check that the alternatives don't include the reference
            for alt in pos_rows["alt"]:
                self.assertNotEqual(alt, ref_base)

            # Check that we have all 3 alternatives
            expected_alts = sorted(set("ACGT") - {ref_base})
            actual_alts = sorted(pos_rows["alt"])
            self.assertEqual(actual_alts, expected_alts)

    def test_get_sm_subsequences(self):
        """Test generating saturation mutagenesis sequences around an anchor."""
        chrom = "chr1"
        anchor = 50  # Middle of the sequence
        anchor_radius = 5
        seq_len = 100

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom, anchor, anchor_radius, seq_len, self.reference
        )

        # Check reference shape
        self.assertEqual(ref_1h.shape, (100, 4))  # 100 bases, 4 nucleotides

        # Check alternative sequences - should have 3 alternatives for positions in the radius
        # 2*anchor_radius positions * 3 alternatives = 30 sequences
        expected_count = 2 * anchor_radius * 3
        self.assertEqual(len(alt_seqs), expected_count)

        # Verify metadata
        self.assertEqual(len(metadata), expected_count)

        # Check that all positions are within the radius
        anchor_offset = anchor - (
            anchor - seq_len // 2
        )  # Anchor position in the sequence
        for pos in metadata["offset"]:
            self.assertTrue(pos >= anchor_offset - anchor_radius)
            self.assertTrue(pos < anchor_offset + anchor_radius)

    def test_tensor_output(self):
        """Test that PyTorch tensors are returned when available."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        chrom = "chr1"
        start = 0
        end = 4  # ACGT

        ref_1h, alt_seqs, _ = sl.get_sm_sequences(chrom, start, end, self.reference)

        # Check that we get tensors
        self.assertTrue(isinstance(ref_1h, torch.Tensor))
        self.assertTrue(isinstance(alt_seqs, torch.Tensor))

        # Check shapes
        self.assertEqual(ref_1h.shape, (4, 4))
        self.assertEqual(alt_seqs.shape, (12, 4, 4))  # 4 positions * 3 alternatives


if __name__ == "__main__":
    unittest.main()
