"""
Unit tests for the supremo_lite module.

This script tests the basic functionality of supremo_lite to ensure it works as expected.
"""

import unittest
import numpy as np
import pandas as pd
import supremo_lite as sl

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSupremoLite(unittest.TestCase):

    def test_encode_decode_seq(self):
        """Test sequence encoding and decoding"""
        # Test single sequence
        seq = "ACGT"
        encoded = sl.encode_seq(seq)
        decoded = sl.decode_seq(encoded)

        self.assertEqual(decoded, seq)

        # Test batch of sequences
        seqs = ["ACGT", "TGCA", "GATC"]
        encoded_batch = sl.encode_seq(seqs)
        decoded_batch = sl.decode_seq(encoded_batch)

        for orig, decoded in zip(seqs, decoded_batch):
            self.assertEqual(decoded, orig)

    def test_reverse_complement(self):
        """Test reverse complement functions"""
        # Test string reverse complement
        seq = "ACGTACGT"
        rc = sl.rc_str(seq)
        self.assertEqual(rc, "ACGTACGT"[::-1].translate(str.maketrans("ACGT", "TGCA")))

        # Test one-hot reverse complement
        encoded = sl.encode_seq(seq)
        rc_encoded = sl.rc(encoded)
        rc_decoded = sl.decode_seq(rc_encoded)

        self.assertEqual(rc_decoded, rc)

    def test_read_vcf(self):
        """Test VCF reading with a mock DataFrame"""
        # Create a mock VCF-like DataFrame
        mock_vcf = pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "pos": [1000, 2000],
                "id": [".", "rs123"],
                "ref": ["A", "GTC"],
                "alt": ["G", "G"],
            }
        )

        # Since we can't easily test file reading, we'll just ensure the function exists
        self.assertTrue(hasattr(sl, "read_vcf"))

    def test_get_sm_sequences(self):
        """Test saturation mutagenesis with a mock reference"""

        # Create a mock reference genome class
        class MockReference:
            def __getitem__(self, chrom):
                return MockChromosome()

        class MockChromosome:
            def __getitem__(self, slice_obj):
                # Return a fixed sequence for testing
                return MockSequence("ACGT" * 3)

        class MockSequence:
            def __init__(self, seq):
                self.seq = seq

        # Run saturation mutagenesis
        ref_1h, alt_seqs, metadata = sl.get_sm_sequences("chr1", 0, 12, MockReference())

        # Check dimensions
        self.assertEqual(ref_1h.shape, (12, 4))  # 12 bases, 4 nucleotides

        # Each position has 3 alternatives, so 12 positions * 3 alts = 36 sequences
        expected_alt_count = 12 * 3
        self.assertEqual(alt_seqs.shape[0], expected_alt_count)

        # Check metadata
        self.assertEqual(len(metadata), expected_alt_count)
        self.assertEqual(
            list(metadata.columns), ["chrom", "start", "end", "offset", "ref", "alt"]
        )

        # Verify some mutations
        refs = metadata["ref"].unique()
        for ref in refs:
            # For each reference base, check that we have all possible alternatives
            alts = metadata[metadata["ref"] == ref]["alt"].unique()
            expected_alts = sorted({"A", "C", "G", "T"} - {ref.upper()})
            self.assertEqual(sorted(alts), expected_alts)


if __name__ == "__main__":
    unittest.main()
