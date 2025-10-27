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

    def test_bed_regions_dataframe(self):
        """Test BED regions filtering using DataFrame input."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 5
        seq_len = 100

        # Create BED regions that cover positions 45-50 (5 positions in mutagenesis window)
        bed_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [45, 48],  # Cover positions 45-47 and 48-50
                "end": [48, 51],
            }
        )

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom, anchor, anchor_radius, seq_len, self.reference, bed_regions=bed_df
        )

        # Should have mutations for positions 45-50 (6 positions)
        # But mutagenesis window is anchor_offset-radius to anchor_offset+radius
        # anchor_offset = 50, so window is 45-55
        # BED regions cover 45-51, intersection is 45-51 (6 positions)
        # Each position gets 3 alternatives = 18 total mutations
        expected_positions = {45, 46, 47, 48, 49, 50}  # 6 positions
        actual_positions = set(metadata["offset"].unique())

        # Check that we have the right positions
        self.assertTrue(len(actual_positions) <= len(expected_positions))
        self.assertTrue(actual_positions.issubset(expected_positions))

        # Check each position has 3 alternatives
        for pos in actual_positions:
            pos_count = (metadata["offset"] == pos).sum()
            self.assertEqual(pos_count, 3)

    def test_bed_regions_file(self):
        """Test BED regions filtering using file input."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 3
        seq_len = 100

        # Create temporary BED file
        bed_content = "chr1\t48\t52\nexcluded_chrom\t100\t200\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
            f.write(bed_content)
            bed_file = f.name

        try:
            ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
                chrom,
                anchor,
                anchor_radius,
                seq_len,
                self.reference,
                bed_regions=bed_file,
                auto_map_chromosomes=True,
            )

            # BED region covers positions 48-51, mutagenesis window is 47-53
            # Intersection should be positions 48-51 (4 positions)
            expected_positions = {48, 49, 50, 51}
            actual_positions = set(metadata["offset"].unique())

            self.assertTrue(actual_positions.issubset(expected_positions))

        finally:
            os.unlink(bed_file)

    def test_bed_regions_chromosome_matching(self):
        """Test chromosome name matching between reference and BED file."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 2
        seq_len = 100

        # BED file uses '1' instead of 'chr1'
        bed_df = pd.DataFrame(
            {"chrom": ["1"], "start": [48], "end": [52]}  # Different naming convention
        )

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom,
            anchor,
            anchor_radius,
            seq_len,
            self.reference,
            bed_regions=bed_df,
            auto_map_chromosomes=True,
        )

        # Should still work due to chromosome matching
        self.assertTrue(len(metadata) > 0)

        # Check positions are within expected range
        positions = set(metadata["offset"].unique())
        expected_window = set(
            range(48, 52)
        )  # BED region intersection with mutagenesis window
        self.assertTrue(positions.issubset(expected_window))

    def test_bed_regions_no_overlap(self):
        """Test behavior when BED regions don't overlap with mutagenesis window."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 2
        seq_len = 100

        # BED region far from mutagenesis window (48-52)
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [10], "end": [20]})

        with self.assertWarns(UserWarning):
            ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
                chrom,
                anchor,
                anchor_radius,
                seq_len,
                self.reference,
                bed_regions=bed_df,
            )

        # Should return empty results
        self.assertEqual(len(alt_seqs), 0)
        self.assertEqual(len(metadata), 0)
        self.assertEqual(ref_1h.shape, (100, 4))  # Reference should still be returned

    def test_bed_regions_no_chromosome(self):
        """Test behavior when BED file has no regions for target chromosome."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 2
        seq_len = 100

        # BED region for different chromosome
        bed_df = pd.DataFrame({"chrom": ["chr2"], "start": [48], "end": [52]})

        # With auto_map_chromosomes=True, should warn but still work
        with self.assertWarns(UserWarning):
            ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
                chrom,
                anchor,
                anchor_radius,
                seq_len,
                self.reference,
                bed_regions=bed_df,
                auto_map_chromosomes=True,
            )

        # Should return empty results (no overlap after mapping)
        self.assertEqual(len(alt_seqs), 0)
        self.assertEqual(len(metadata), 0)

    def test_bed_regions_invalid_format(self):
        """Test error handling for invalid BED file formats."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 2
        seq_len = 100

        # Invalid BED DataFrame (missing required columns)
        invalid_bed_df = pd.DataFrame(
            {"chromosome": ["chr1"], "start": [48], "end": [52]}  # Wrong column name
        )

        with self.assertRaises(ValueError):
            sl.get_sm_subsequences(
                chrom,
                anchor,
                anchor_radius,
                seq_len,
                self.reference,
                bed_regions=invalid_bed_df,
            )

    def test_bed_regions_partial_overlap(self):
        """Test BED regions that partially overlap with mutagenesis window."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 5  # Window: 45-55
        seq_len = 100

        # BED region partially overlaps: 40-48 (should cover positions 45-47)
        bed_df = pd.DataFrame({"chrom": ["chr1"], "start": [40], "end": [48]})

        ref_1h, alt_seqs, metadata = sl.get_sm_subsequences(
            chrom, anchor, anchor_radius, seq_len, self.reference, bed_regions=bed_df
        )

        # Should only have mutations for positions 45, 46, 47 (3 positions * 3 alts = 9)
        positions = set(metadata["offset"].unique())
        expected_positions = {45, 46, 47}
        self.assertEqual(positions, expected_positions)
        self.assertEqual(len(metadata), 9)  # 3 positions * 3 alternatives each

    def test_backward_compatibility(self):
        """Test that function works exactly the same when bed_regions=None."""
        chrom = "chr1"
        anchor = 50
        anchor_radius = 3
        seq_len = 100

        # Test without BED regions
        ref_1h_no_bed, alt_seqs_no_bed, metadata_no_bed = sl.get_sm_subsequences(
            chrom, anchor, anchor_radius, seq_len, self.reference
        )

        # Test with bed_regions=None
        ref_1h_none, alt_seqs_none, metadata_none = sl.get_sm_subsequences(
            chrom, anchor, anchor_radius, seq_len, self.reference, bed_regions=None
        )

        # Results should be identical
        np.testing.assert_array_equal(ref_1h_no_bed, ref_1h_none)
        np.testing.assert_array_equal(alt_seqs_no_bed, alt_seqs_none)
        pd.testing.assert_frame_equal(metadata_no_bed, metadata_none)


if __name__ == "__main__":
    unittest.main()
