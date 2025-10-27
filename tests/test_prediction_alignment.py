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

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

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

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

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

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

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

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

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

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

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

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
