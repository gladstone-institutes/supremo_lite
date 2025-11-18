"""
Tests for the prediction alignment functions.

This file tests prediction alignment with the complete workflow:
sequences → model predictions → coordinate-based alignment

Uses built-in TestModel and TestModel2D from supremo_lite.mock_models
with test VCFs from tests/data directory.
"""

import os
import torch
import pytest
import supremo_lite as sl
from supremo_lite.mock_models import TestModel, TestModel2D
from supremo_lite.prediction_alignment import align_predictions_by_coordinate


class TestBasic1DPredictionAlignment:
    """Test 1D prediction alignment for basic variant types using TestModel."""

    def setup_method(self):
        """Set up test data paths and model."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # Initialize 1D model with basic parameters
        self.seq_len = 200
        self.bin_size = 32
        self.crop_length = 16
        self.model = TestModel(
            seq_length=self.seq_len,
            n_targets=2,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )

    def test_snv_1d_alignment(self):
        """Test 1D alignment for SNV variants."""
        snv_vcf = os.path.join(self.data_dir, "snp", "snp.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=snv_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Sequences are already in (batch, 4, seq_len) format - no permute needed!

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],  # First variant, first target
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="1D",
        )

        # For SNV, lengths should be identical
        assert len(aligned_ref) == len(aligned_alt), "SNV should preserve length"

        # Should have no NaN values for SNV
        assert not torch.isnan(aligned_ref).any(), "SNV should not introduce NaN in ref"
        assert not torch.isnan(aligned_alt).any(), "SNV should not introduce NaN in alt"

        # Outputs should be tensors
        assert isinstance(aligned_ref, torch.Tensor), "Output should be tensor"
        assert isinstance(aligned_alt, torch.Tensor), "Output should be tensor"

        print("✓ SNV 1D alignment test passed")

    def test_ins_1d_alignment(self):
        """Test 1D alignment for insertion variants."""
        ins_vcf = os.path.join(self.data_dir, "ins", "ins.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=ins_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Sequences are already in (batch, 4, seq_len) format - no permute needed!

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="1D",
        )

        # After alignment, should have same length
        assert len(aligned_ref) == len(aligned_alt), "Alignment should equalize lengths"

        # Insertion should introduce NaN values in aligned predictions
        # (either in ref or alt depending on svlen and bin alignment)
        has_nan = torch.isnan(aligned_ref).any() or torch.isnan(aligned_alt).any()

        # For insertions, if svlen spans bins, we expect NaN
        svlen = abs(var_metadata.get("svlen", 0))
        if svlen >= self.bin_size:
            assert has_nan, f"Large insertion (svlen={svlen}) should introduce NaN"

        print("✓ INS 1D alignment test passed")

    def test_del_1d_alignment(self):
        """Test 1D alignment for deletion variants."""
        del_vcf = os.path.join(self.data_dir, "del", "del.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=del_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Sequences are already in (batch, 4, seq_len) format - no permute needed!

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="1D",
        )

        # After alignment, should have same length
        assert len(aligned_ref) == len(aligned_alt), "Alignment should equalize lengths"

        # Deletion should introduce NaN values in aligned predictions
        has_nan = torch.isnan(aligned_ref).any() or torch.isnan(aligned_alt).any()

        # For deletions, if svlen spans bins, we expect NaN
        svlen = abs(var_metadata.get("svlen", 0))
        if svlen >= self.bin_size:
            assert has_nan, f"Large deletion (svlen={svlen}) should introduce NaN"

        print("✓ DEL 1D alignment test passed")


class TestBasic2DPredictionAlignment:
    """Test 2D prediction alignment for basic variant types using TestModel2D."""

    def setup_method(self):
        """Set up test data paths and model."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # Initialize 2D model with basic parameters
        # Use larger sequence to get reasonable matrix size after binning and cropping
        self.seq_len = 512
        self.bin_size = 32
        self.crop_length = 64  # 2 bins cropped from each side
        self.diag_offset = (
            0  # TestModel2D returns full matrices without diagonal masking
        )

        self.model = TestModel2D(
            seq_length=self.seq_len,
            n_targets=1,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )

        # Calculate matrix size for alignment
        effective_len = self.seq_len - 2 * self.crop_length
        self.matrix_size = effective_len // self.bin_size  # (512 - 128) / 32 = 12 bins

    def test_snv_2d_alignment(self):
        """Test 2D alignment for SNV variants."""
        snv_vcf = os.path.join(self.data_dir, "snp", "snp.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=snv_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],  # First variant, first target (flattened upper triangle)
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="2D",
            matrix_size=self.matrix_size,
            diag_offset=self.diag_offset,
        )

        # For SNV, shapes should be identical
        assert aligned_ref.shape == aligned_alt.shape, "SNV should preserve shape"

        # Should be 2D matrices (full contact map)
        assert aligned_ref.ndim == 2, "Output should be 2D matrix (full contact map)"
        assert aligned_alt.ndim == 2, "Output should be 2D matrix (full contact map)"

        # Expected shape: (matrix_size, matrix_size)
        assert aligned_ref.shape == (
            self.matrix_size,
            self.matrix_size,
        ), f"Expected ({self.matrix_size}, {self.matrix_size}), got {aligned_ref.shape}"

        # Outputs should be tensors
        assert isinstance(aligned_ref, torch.Tensor), "Output should be tensor"
        assert isinstance(aligned_alt, torch.Tensor), "Output should be tensor"

        print("✓ SNV 2D alignment test passed")

    def test_ins_2d_alignment(self):
        """Test 2D alignment for insertion variants."""
        ins_vcf = os.path.join(self.data_dir, "ins", "ins.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=ins_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Sequences are already in (batch, 4, seq_len) format - no permute needed!

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="2D",
            matrix_size=self.matrix_size,
            diag_offset=self.diag_offset,
        )

        # After alignment, shapes should match
        assert (
            aligned_ref.shape == aligned_alt.shape
        ), "Alignment should equalize shapes"

        # Should be 2D matrices (full contact map)
        assert aligned_ref.ndim == 2, "Output should be 2D matrix (full contact map)"
        assert aligned_alt.ndim == 2, "Output should be 2D matrix (full contact map)"

        # For 2D, insertions may introduce NaN elements
        has_nan = torch.isnan(aligned_ref).any() or torch.isnan(aligned_alt).any()

        svlen = abs(var_metadata.get("svlen", 0))
        if svlen >= self.bin_size:
            assert (
                has_nan
            ), f"Large insertion (svlen={svlen}) should introduce NaN in 2D"

        print("✓ INS 2D alignment test passed")

    def test_del_2d_alignment(self):
        """Test 2D alignment for deletion variants."""
        del_vcf = os.path.join(self.data_dir, "del", "del.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=del_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Sequences are already in (batch, 4, seq_len) format - no permute needed!

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="2D",
            matrix_size=self.matrix_size,
            diag_offset=self.diag_offset,
        )

        # After alignment, shapes should match
        assert (
            aligned_ref.shape == aligned_alt.shape
        ), "Alignment should equalize shapes"

        # Should be 2D matrices (full contact map)
        assert aligned_ref.ndim == 2, "Output should be 2D matrix (full contact map)"
        assert aligned_alt.ndim == 2, "Output should be 2D matrix (full contact map)"

        # For 2D, deletions may introduce NaN elements
        has_nan = torch.isnan(aligned_ref).any() or torch.isnan(aligned_alt).any()

        svlen = abs(var_metadata.get("svlen", 0))
        if svlen >= self.bin_size:
            assert has_nan, f"Large deletion (svlen={svlen}) should introduce NaN in 2D"

        print("✓ DEL 2D alignment test passed")


class TestCenteredMasking:
    """Test that masking is centered on variants (bug fix verification)."""

    def setup_method(self):
        """Set up test data paths and model."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # Use parameters from the prediction_alignment notebook examples
        # These match the test data structure (see docs/notebooks/03_prediction_alignment.ipynb)
        self.seq_len = 40
        self.bin_size = 2
        self.crop_length = 2
        self.model = TestModel(
            seq_length=self.seq_len,
            n_targets=1,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )
        # Expected bins: (40 - 2*2) / 2 = 18 bins

    def test_deletion_masking_centered(self):
        """
        Test that deletion masking is centered on the variant position.

        This test uses the multi.vcf file and DEL variant at index 2, which is properly
        centered in the window (as used in the prediction_alignment notebook example).

        The deletion is at position 10 in a 40bp window, which should result in
        centered masking in the prediction bins.
        """
        # Use multi.vcf which has variants properly centered in windows
        multi_vcf = os.path.join(self.data_dir, "multi", "multi.vcf")

        # Generate sequences
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=multi_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        ref_seqs, alt_seqs, metadata = results[0]

        # Run predictions
        ref_preds = self.model(ref_seqs)
        alt_preds = self.model(alt_seqs)

        # Test the deletion variant at index 2 (as in notebook example)
        var_idx = 2
        var_metadata = metadata.iloc[var_idx].to_dict()

        # Verify this is the deletion we expect
        assert var_metadata['variant_type'] == 'DEL', "Expected DEL variant at index 2"

        # Align predictions
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[var_idx, 0],
            alt_preds[var_idx, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="1D",
        )

        # Find NaN positions (should be in the alternate, as deletion removes sequence)
        nan_mask_alt = torch.isnan(aligned_alt)

        if nan_mask_alt.any():
            nan_positions = torch.where(nan_mask_alt)[0].cpu().numpy()

            # Calculate the center of NaN bins
            nan_center = (nan_positions[0] + nan_positions[-1]) / 2.0

            # Calculate expected bin position for the variant
            # Variant is at position 10 in window starting at 0
            # After crop (2bp from each side), position 10 becomes position 8 in cropped sequence
            # Bin index: 8 / 2 = 4
            variant_rel_pos = var_metadata['variant_pos0'] - var_metadata['window_start']
            variant_rel_pos_cropped = variant_rel_pos - self.crop_length
            expected_variant_bin = variant_rel_pos_cropped / self.bin_size

            # The NaN bins should be centered around the variant position
            # This is the key test: masking should be centered on the variant, not right-aligned
            tolerance = 1.0  # Allow for rounding in bin calculation
            assert abs(nan_center - expected_variant_bin) <= tolerance, (
                f"NaN masking not centered on variant position!\n"
                f"  NaN center: {nan_center:.2f}\n"
                f"  Expected variant bin: {expected_variant_bin:.2f}\n"
                f"  Difference: {abs(nan_center - expected_variant_bin):.2f}\n"
                f"  Variant pos: {var_metadata['variant_pos0']}, window_start: {var_metadata['window_start']}"
            )

            print(f"✓ Deletion masking centered on variant: NaN bins {nan_positions[0]}-{nan_positions[-1]}, "
                  f"center={nan_center:.2f}, expected_variant_bin={expected_variant_bin:.2f}")
            print(f"  Variant at position {var_metadata['variant_pos0']} in window starting at {var_metadata['window_start']}")
        else:
            # If no NaN, fail - we expect NaN for this deletion
            pytest.fail("Expected NaN masking for deletion variant")

    def test_window_relative_positioning(self):
        """
        Test that bin positions are calculated relative to window, not absolute genomic coords.

        This is the core bug that was fixed: using absolute genomic coordinates
        instead of window-relative coordinates for bin calculation.
        """
        from supremo_lite.prediction_alignment import VariantPosition

        # Test case: variant at absolute position 1000, window starts at 900
        # The variant is at position 100 within the 200bp window
        window_start = 900
        variant_pos = 1000
        bin_size = 32

        var_pos = VariantPosition(
            ref_pos=variant_pos,
            alt_pos=variant_pos,
            svlen=64,  # 2 bins worth
            variant_type="INS"
        )

        # Get bin positions with window_start
        ref_bin, alt_start_bin, alt_end_bin = var_pos.get_bin_positions(
            bin_size, window_start
        )

        # The variant is 100bp from window start
        # Expected bin (centered): ceil(100/32) = 4 (center of variant)
        # With 64bp insertion (2 bins), centered masking should place bins around position 4
        expected_center_bin = (100 / bin_size)  # ~3.125

        # Calculate actual center of masked region
        actual_center = (alt_start_bin + alt_end_bin) / 2.0

        # Verify the bin is calculated relative to window, not absolute position
        # If bug existed, ref_bin would be ~31 (1000/32), not ~4
        assert ref_bin < 10, (
            f"Bin position should be window-relative (~4), not absolute (~31). "
            f"Got ref_bin={ref_bin}"
        )

        # Verify centering (within reasonable tolerance)
        assert abs(actual_center - expected_center_bin) <= 2, (
            f"Masked region not centered around variant position. "
            f"Expected center ~{expected_center_bin:.2f}, got {actual_center:.2f}"
        )

        print(f"✓ Window-relative positioning correct: ref_bin={ref_bin}, "
              f"masked region center={actual_center:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
