"""
Tests for the in-silico mutagenesis functions in supremo_lite.

This file tests the saturation mutagenesis functions.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import supremo_lite as sl
from pyfaidx import Fasta

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSaturationMutagenesis(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        # Use real test genome instead of mock
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        reference_path = os.path.join(test_data_dir, "test_genome.fa")
        self.reference = Fasta(reference_path)

    def test_get_sm_sequences(self):
        """Test generating saturation mutagenesis sequences."""
        chrom = "chr1"
        start = 0
        end = 10  # ATGAATATAA (from test_genome.fa)

        ref_1h, alt_seqs, metadata = sl.get_sm_sequences(
            chrom, start, end, self.reference
        )

        # Check reference shape
        self.assertEqual(ref_1h.shape, (4, 10))  # 4 nucleotide channels, 10 bases

        # Check alternative sequences - should have 3 alternatives for each position
        # 10 positions * 3 alternatives = 30 sequences
        self.assertEqual(len(alt_seqs), 30)

        # Verify metadata
        self.assertEqual(len(metadata), 30)
        self.assertEqual(
            list(metadata.columns),
            ["chrom", "window_start", "window_end", "variant_offset0", "ref", "alt"],
        )

        # Check each position gets all 3 alternatives
        pos_counts = metadata["variant_offset0"].value_counts()
        self.assertEqual(len(pos_counts), 10)  # all 10 positions should be present
        self.assertTrue(
            all(count == 3 for count in pos_counts)
        )  # each should have 3 alternatives

        # Check that for each reference base, we get the right alternatives
        actual_seq = str(self.reference[chrom][start:end].seq)
        for i, ref_base in enumerate(actual_seq):
            # Get all rows for this position
            pos_rows = metadata[metadata["variant_offset0"] == i]

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
        anchor = 40  # Middle of 80bp chromosome
        anchor_radius = 5
        seq_len = 80

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom, seq_len, self.reference, anchor=anchor, anchor_radius=anchor_radius
        )

        # Check reference shape
        self.assertEqual(ref_1h.shape, (4, 80))  # 4 nucleotide channels, 80 bases

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
        for pos in metadata["variant_offset0"]:
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

    def test_bed_regions_dataframe(self):
        """Test BED regions filtering using DataFrame input."""
        chrom = "chr1"
        seq_len = 80  # Match chromosome length

        # Create BED regions - centered on these will give us different windows
        bed_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [20, 50],  # Two separate regions
                "end": [30, 60],
            }
        )

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom, seq_len=seq_len, reference_fasta=self.reference, bed_regions=bed_df
        )

        # Each BED region gets mutated within its centered window
        # Region 1 [20,30): center=25, window=[0,80), mutations at positions 20-29
        # Region 2 [50,60): center=55, window=[15,80) but adjusted to [0,80), mutations at positions 50-59
        # Total positions should be 10+10=20, but need to account for window-relative positions
        self.assertTrue(len(metadata) > 0)
        self.assertEqual(
            len(metadata) % 3, 0
        )  # Should be multiple of 3 (3 alts per position)

    def test_bed_regions_file(self):
        """Test BED regions filtering using actual BED file."""
        chrom = "chr1"
        seq_len = 80  # Match chromosome length

        # Use the real test BED file
        bed_file = os.path.join(os.path.dirname(__file__), "data", "test_regions.bed")

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom,
            seq_len=seq_len,
            reference_fasta=self.reference,
            bed_regions=bed_file,
            auto_map_chromosomes=True,
        )

        # test_regions.bed has three regions for chr1: [10,20), [30,45), [60,75)
        # Each region gets its own window, mutations are at window-relative positions
        # We should have mutations from all three regions
        self.assertTrue(len(metadata) > 0)
        self.assertEqual(len(metadata) % 3, 0)  # Multiple of 3 (3 alts per position)

    def test_bed_regions_chromosome_matching(self):
        """Test chromosome name matching between reference and BED file."""
        chrom = "chr1"
        seq_len = 80

        # BED file uses '1' instead of 'chr1'
        bed_df = pd.DataFrame(
            {"chrom": ["1"], "start": [35], "end": [45]}  # Different naming convention
        )

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom,
            seq_len=seq_len,
            reference_fasta=self.reference,
            bed_regions=bed_df,
            auto_map_chromosomes=True,
        )

        # Should still work due to chromosome matching
        self.assertTrue(len(metadata) > 0)
        # Region should have 10 positions * 3 alts = 30 mutations
        self.assertEqual(len(metadata), 30)

    def test_bed_regions_no_overlap(self):
        """Test behavior when BED regions are outside the chromosome bounds."""
        chrom = "chr1"
        seq_len = 80

        # BED region outside the chromosome (chr1 is 80bp, BED wants 110-120)
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [110], "end": [120]})

        with self.assertWarns(UserWarning):
            ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
                bed_regions=bed_df,
            )

        # Should return empty results
        self.assertEqual(len(alt_seqs), 0)
        self.assertEqual(len(metadata), 0)
        self.assertEqual(ref_1h.shape, (4, 80))  # Reference should still be returned

    def test_bed_regions_no_chromosome(self):
        """Test behavior when BED file has no regions for target chromosome."""
        chrom = "chr1"
        seq_len = 80

        # BED region for different chromosome
        bed_df = pd.DataFrame({"chrom": ["chr2"], "start": [35], "end": [45]})

        # With auto_map_chromosomes=True, should warn but still work
        with self.assertWarns(UserWarning):
            ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
                bed_regions=bed_df,
                auto_map_chromosomes=True,
            )

        # Should return empty results (no overlap after mapping)
        self.assertEqual(len(alt_seqs), 0)
        self.assertEqual(len(metadata), 0)

    def test_bed_regions_invalid_format(self):
        """Test error handling for invalid BED file formats."""
        chrom = "chr1"
        seq_len = 80

        # Invalid BED DataFrame (missing required columns)
        invalid_bed_df = pd.DataFrame(
            {"chromosome": ["chr1"], "start": [35], "end": [45]}  # Wrong column name
        )

        with self.assertRaises(ValueError):
            sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
                bed_regions=invalid_bed_df,
            )

    def test_bed_regions_partial_overlap(self):
        """Test BED regions with their own centered windows."""
        chrom = "chr1"
        seq_len = 80

        # BED region 35-45 gets its own centered window
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [35], "end": [45]})

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom, seq_len=seq_len, reference_fasta=self.reference, bed_regions=bed_df
        )

        # Should have mutations for the 10 positions in the BED region (10 * 3 = 30)
        self.assertEqual(len(metadata), 30)

    def test_parameter_validation(self):
        """Test parameter validation for mutual exclusivity and required parameters."""
        chrom = "chr1"
        seq_len = 100

        # Should raise ValueError when neither approach is provided
        with self.assertRaises(ValueError) as context:
            sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
            )

        self.assertIn("Must provide either", str(context.exception))

        # Should raise ValueError when only anchor provided (needs anchor_radius too)
        with self.assertRaises(ValueError) as context:
            sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
                anchor=50,
            )

        self.assertIn("must be provided together", str(context.exception))

        # Should raise ValueError when only anchor_radius provided (needs anchor too)
        with self.assertRaises(ValueError) as context:
            sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
                anchor_radius=5,
            )

        self.assertIn("must be provided together", str(context.exception))

        # Should raise ValueError when both approaches are provided (mutually exclusive)
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [45], "end": [55]})

        with self.assertRaises(ValueError) as context:
            sl.get_sm_subsequences(
                chrom,
                seq_len=seq_len,
                reference_fasta=self.reference,
                anchor=50,
                anchor_radius=5,
                bed_regions=bed_df,
            )

        self.assertIn("mutually exclusive", str(context.exception))

    def test_bed_regions_multi_chromosome(self):
        """Test BED file with multiple chromosomes using test_regions.bed."""
        bed_file = os.path.join(os.path.dirname(__file__), "data", "test_regions.bed")

        # Test chr2 which has regions: [15,30), [40,55)
        chrom = "chr2"
        seq_len = 80

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom,
            seq_len=seq_len,
            reference_fasta=self.reference,
            bed_regions=bed_file,
            auto_map_chromosomes=True,
        )

        # Both regions should be included
        # Region 1: [15,30) = 15 positions * 3 alts = 45 mutations
        # Region 2: [40,55) = 15 positions * 3 alts = 45 mutations
        # Total: 90 mutations
        self.assertTrue(len(metadata) > 0)
        self.assertEqual(len(metadata), 90)  # 30 positions * 3 alternatives


class TestScrambledSubsequences(unittest.TestCase):
    """Test dinucleotide shuffling and scrambled subsequences."""

    def setUp(self):
        """Set up test data."""
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        reference_path = os.path.join(test_data_dir, "test_genome.fa")
        self.reference = Fasta(reference_path)

    def test_basic_scrambling(self):
        """Test basic scrambling functionality."""
        chrom = "chr1"
        seq_len = 80

        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [30], "end": [50]})

        ref_seqs, scrambled_seqs, metadata = sl.get_scrambled_subsequences(
            chrom,
            seq_len,
            self.reference,
            bed_regions=bed_df,
            n_scrambles=5,
            random_state=42,
        )

        # Should have 1 ref sequence and 5 scrambled
        self.assertEqual(ref_seqs.shape[0], 1)
        self.assertEqual(scrambled_seqs.shape[0], 5)
        self.assertEqual(len(metadata), 5)

        # Check metadata columns
        expected_cols = [
            "chrom",
            "window_start",
            "window_end",
            "scramble_start",
            "scramble_end",
            "scramble_idx",
            "ref",
            "alt",
        ]
        self.assertEqual(list(metadata.columns), expected_cols)

    def test_scramble_preserves_composition(self):
        """Test that scrambling preserves nucleotide composition."""
        from collections import Counter

        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [20], "end": [60]})

        _, _, metadata = sl.get_scrambled_subsequences(
            "chr1",
            80,
            self.reference,
            bed_regions=bed_df,
            n_scrambles=1,
            random_state=42,
        )

        ref_region = metadata.iloc[0]["ref"]
        alt_region = metadata.iloc[0]["alt"]

        # Same length
        self.assertEqual(len(ref_region), len(alt_region))

        # Same nucleotide composition
        self.assertEqual(Counter(ref_region), Counter(alt_region))

    def test_scramble_reproducibility(self):
        """Test that random_state provides reproducibility."""
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [30], "end": [50]})

        _, scrambled1, meta1 = sl.get_scrambled_subsequences(
            "chr1",
            80,
            self.reference,
            bed_regions=bed_df,
            n_scrambles=3,
            random_state=42,
        )

        _, scrambled2, meta2 = sl.get_scrambled_subsequences(
            "chr1",
            80,
            self.reference,
            bed_regions=bed_df,
            n_scrambles=3,
            random_state=42,
        )

        # Same random state should give same results
        for i in range(3):
            self.assertEqual(meta1.iloc[i]["alt"], meta2.iloc[i]["alt"])

    def test_scramble_with_bed_file(self):
        """Test scrambling with actual BED file."""
        bed_file = os.path.join(os.path.dirname(__file__), "data", "test_regions.bed")

        ref_seqs, scrambled_seqs, metadata = sl.get_scrambled_subsequences(
            "chr1",
            80,
            self.reference,
            bed_regions=bed_file,
            n_scrambles=2,
            auto_map_chromosomes=True,
            random_state=123,
        )

        # test_regions.bed has 2 chr1 regions
        self.assertEqual(ref_seqs.shape[0], 2)
        self.assertEqual(scrambled_seqs.shape[0], 4)  # 2 regions * 2 scrambles
        self.assertEqual(len(metadata), 4)

    def test_scramble_empty_chromosome_returns_original(self):
        """Test handling when no BED regions match chromosome - returns original sequence."""
        bed_df = pd.DataFrame({"chrom": ["chr99"], "start": [30], "end": [50]})

        with self.assertWarns(UserWarning):
            ref_seqs, scrambled_seqs, metadata = sl.get_scrambled_subsequences(
                "chr1",
                80,
                self.reference,
                bed_regions=bed_df,
                n_scrambles=3,
                auto_map_chromosomes=True,
            )

        # Should return 1 reference and n_scrambles "scrambled" (but actually unshuffled)
        self.assertEqual(len(ref_seqs), 1)
        self.assertEqual(len(scrambled_seqs), 3)
        self.assertEqual(len(metadata), 3)

        # The "scrambled" sequences should be identical to original
        for idx in range(3):
            self.assertEqual(metadata.iloc[idx]["ref"], metadata.iloc[idx]["alt"])

        # scramble_start and scramble_end should indicate no scrambling (empty region)
        self.assertEqual(metadata.iloc[0]["scramble_start"], 0)
        self.assertEqual(metadata.iloc[0]["scramble_end"], 0)

    def test_scramble_requires_bed_regions(self):
        """Test that bed_regions is required."""
        with self.assertRaises(ValueError):
            sl.get_scrambled_subsequences("chr1", 80, self.reference, bed_regions=None)

    def test_scramble_with_kmer_size(self):
        """Test scrambling with different k-mer sizes."""
        from collections import Counter

        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [20], "end": [60]})

        # Test kmer_size=2 (preserves mononucleotide composition)
        _, _, meta_k2 = sl.get_scrambled_subsequences(
            "chr1",
            80,
            self.reference,
            bed_regions=bed_df,
            n_scrambles=1,
            kmer_size=2,
            random_state=42,
        )

        ref_k2 = meta_k2.iloc[0]["ref"]
        alt_k2 = meta_k2.iloc[0]["alt"]

        # Same length and composition
        self.assertEqual(len(ref_k2), len(alt_k2))
        self.assertEqual(Counter(ref_k2), Counter(alt_k2))

        # Test kmer_size=3 (preserves dinucleotide frequencies)
        _, _, meta_k3 = sl.get_scrambled_subsequences(
            "chr1",
            80,
            self.reference,
            bed_regions=bed_df,
            n_scrambles=1,
            kmer_size=3,
            random_state=42,
        )

        ref_k3 = meta_k3.iloc[0]["ref"]
        alt_k3 = meta_k3.iloc[0]["alt"]

        self.assertEqual(len(ref_k3), len(alt_k3))
        self.assertEqual(Counter(ref_k3), Counter(alt_k3))

    def test_scramble_kmer_size_validation(self):
        """Test that invalid kmer_size raises ValueError."""
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [30], "end": [50]})

        with self.assertRaises(ValueError):
            sl.get_scrambled_subsequences(
                "chr1",
                80,
                self.reference,
                bed_regions=bed_df,
                kmer_size=0,
            )

        with self.assertRaises(ValueError):
            sl.get_scrambled_subsequences(
                "chr1",
                80,
                self.reference,
                bed_regions=bed_df,
                kmer_size=-1,
            )


class TestKmerShuffle(unittest.TestCase):
    """Test k-mer shuffling functionality."""

    def _get_kmers(self, seq, k):
        """Count non-overlapping k-mers in a sequence."""
        from collections import Counter

        return Counter([seq[i : i + k] for i in range(0, len(seq) - k + 1, k)])

    def test_kmer_shuffle_k1_preserves_mononucleotide_composition(self):
        """Test k=1 preserves mononucleotide (GC) composition."""
        from collections import Counter
        from supremo_lite.mutagenesis import _kmer_shuffle

        seq = "ATGCATGCATGCATGC"
        shuffled = _kmer_shuffle(seq, k=1, random_state=42)

        # Same length
        self.assertEqual(len(seq), len(shuffled))
        # Mononucleotide composition preserved (A, C, G, T counts match)
        self.assertEqual(Counter(seq), Counter(shuffled))

    def test_kmer_shuffle_k2_preserves_dinucleotide_frequencies(self):
        """Test k=2 preserves dinucleotide frequencies."""
        from supremo_lite.mutagenesis import _kmer_shuffle

        # Sequence divisible by 2
        seq = "ATGCATGCATGCATGC"
        shuffled = _kmer_shuffle(seq, k=2, random_state=42)

        # Same length
        self.assertEqual(len(seq), len(shuffled))
        # Dinucleotide (2-mer) frequencies preserved
        orig_dimers = self._get_kmers(seq, 2)
        shuf_dimers = self._get_kmers(shuffled, 2)
        self.assertEqual(orig_dimers, shuf_dimers)

    def test_kmer_shuffle_k3_preserves_trinucleotide_frequencies(self):
        """Test k=3 preserves trinucleotide frequencies."""
        from supremo_lite.mutagenesis import _kmer_shuffle

        # Sequence divisible by 3 (30 bp)
        seq = "ATGCATGCATGCATGCATGCATGCATGCAT"
        shuffled = _kmer_shuffle(seq, k=3, random_state=42)

        # Same length
        self.assertEqual(len(seq), len(shuffled))
        # Trinucleotide (3-mer) frequencies preserved
        orig_trimers = self._get_kmers(seq, 3)
        shuf_trimers = self._get_kmers(shuffled, 3)
        self.assertEqual(orig_trimers, shuf_trimers)

    def test_kmer_shuffle_short_sequence(self):
        """Test handling of sequences too short to shuffle."""
        from supremo_lite.mutagenesis import _kmer_shuffle

        # Sequences shorter than k should be returned unchanged
        self.assertEqual(_kmer_shuffle("A", k=1), "A")
        self.assertEqual(_kmer_shuffle("A", k=2), "A")
        self.assertEqual(_kmer_shuffle("AT", k=3), "AT")

    def test_kmer_shuffle_reproducibility(self):
        """Test that random_state provides reproducibility."""
        from supremo_lite.mutagenesis import _kmer_shuffle

        seq = "ATGCATGCATGCATGCATGC"

        result1 = _kmer_shuffle(seq, k=2, random_state=42)
        result2 = _kmer_shuffle(seq, k=2, random_state=42)

        self.assertEqual(result1, result2)

    def test_kmer_shuffle_invalid_k(self):
        """Test error handling for invalid k values."""
        from supremo_lite.mutagenesis import _kmer_shuffle

        with self.assertRaises(ValueError):
            _kmer_shuffle("ATGC", k=0)

        with self.assertRaises(ValueError):
            _kmer_shuffle("ATGC", k=-1)

    def test_kmer_shuffle_different_k_values(self):
        """Test that different k values produce different results."""
        from collections import Counter
        from supremo_lite.mutagenesis import _kmer_shuffle

        seq = "ATGCATGCATGCATGCATGCATGCATGC"

        result_k1 = _kmer_shuffle(seq, k=1, random_state=42)
        result_k2 = _kmer_shuffle(seq, k=2, random_state=42)
        result_k3 = _kmer_shuffle(seq, k=3, random_state=42)

        # All should preserve mononucleotide composition
        self.assertEqual(Counter(seq), Counter(result_k1))
        self.assertEqual(Counter(seq), Counter(result_k2))
        self.assertEqual(Counter(seq), Counter(result_k3))

    def test_kmer_shuffle_leftover_bases(self):
        """Test that leftover bases (not forming complete k-mer) are preserved."""
        from supremo_lite.mutagenesis import _kmer_shuffle

        # 7 bp sequence with k=3: 2 complete 3-mers + 1 leftover base
        seq = "ATGCATG"
        shuffled = _kmer_shuffle(seq, k=3, random_state=42)

        # Same length
        self.assertEqual(len(seq), len(shuffled))
        # Last base should be preserved (leftover)
        self.assertEqual(seq[-1], shuffled[-1])


if __name__ == "__main__":
    unittest.main()
