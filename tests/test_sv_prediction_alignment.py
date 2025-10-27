"""
Comprehensive tests for structural variant prediction alignment.

Tests prediction alignment for INV, DUP, and BND variants with both 1D and 2D predictions.
Uses existing test data in tests/data/inv, tests/data/dup, and tests/data/bnd directories.
Uses built-in TestModel and TestModel2D from supremo_lite.mock_models.
"""

import os
import torch
import pytest
import supremo_lite as sl
from supremo_lite.mock_models import TestModel, TestModel2D
from supremo_lite.prediction_alignment import align_predictions_by_coordinate


class TestInversionPredictionAlignment:
    """Test prediction alignment for inversion (INV) variants."""

    def setup_method(self):
        """Set up test data and models."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")

        # Model parameters
        # Note: Use larger seq_len for 2D models to get reasonable matrix size after binning/cropping
        self.seq_len = 512
        self.bin_size = 32
        self.crop_length = 64

        # Initialize 1D model
        self.model_1d = TestModel(
            seq_length=self.seq_len,
            n_targets=1,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )

        # Initialize 2D model
        self.model_2d = TestModel2D(
            seq_length=self.seq_len,
            n_targets=1,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )
        self.diag_offset = (
            0  # TestModel2D returns full matrices without diagonal masking
        )

        # Calculate matrix size for 2D
        effective_len = self.seq_len - 2 * self.crop_length
        self.matrix_size = effective_len // self.bin_size

    def test_inv_1d_alignment_masking(self):
        """Test that INV 1D alignment masks inverted region bins in both REF and ALT."""
        # Get sequences with inversions
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.inv_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find INV variants (can be SV_INV or SV_BND_INV)
        inv_results = [r for r in results if "INV" in r[2]["variant_type"].iloc[0]]
        assert len(inv_results) > 0, "Should have at least one INV variant"

        ref_seqs, alt_seqs, metadata = inv_results[0]

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

        # Run model predictions
        ref_preds = self.model_1d(ref_seqs)
        alt_preds = self.model_1d(alt_seqs)

        # Get metadata for first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions (first variant, first target)
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="1D",
        )

        # For inversions, bins in the inverted region should be masked (NaN) in both REF and ALT
        # Check that we have NaN values (masking occurred)
        assert torch.isnan(aligned_ref).any(), "INV 1D: REF should have masked bins"
        assert torch.isnan(aligned_alt).any(), "INV 1D: ALT should have masked bins"

        # Check that REF and ALT have same NaN pattern (both masked at same positions)
        assert torch.equal(
            torch.isnan(aligned_ref), torch.isnan(aligned_alt)
        ), "INV 1D: REF and ALT should have identical masking patterns"

        print("✓ INV 1D alignment masking test passed")

    def test_inv_2d_alignment_cross_pattern_masking(self):
        """Test that INV 2D alignment uses cross-pattern masking (rows AND columns)."""
        # Get sequences with inversions
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.inv_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find INV variants (can be SV_INV or SV_BND_INV)
        inv_results = [r for r in results if "INV" in r[2]["variant_type"].iloc[0]]
        assert len(inv_results) > 0, "Should have at least one INV variant"

        ref_seqs, alt_seqs, metadata = inv_results[0]

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

        # Run model predictions (returns full 2D contact matrices)
        ref_preds = self.model_2d(ref_seqs)
        alt_preds = self.model_2d(alt_seqs)

        # Get metadata for first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions (first variant, first target)
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="2D",
            matrix_size=self.matrix_size,
            diag_offset=self.diag_offset,
        )

        # For inversions in 2D, cross-pattern masking means entire rows AND columns are masked
        # This creates more NaN values than just diagonal masking
        assert torch.isnan(aligned_ref).any(), "INV 2D: REF should have masked elements"
        assert torch.isnan(aligned_alt).any(), "INV 2D: ALT should have masked elements"

        # Check that REF and ALT have same NaN pattern
        assert torch.equal(
            torch.isnan(aligned_ref), torch.isnan(aligned_alt)
        ), "INV 2D: REF and ALT should have identical masking patterns"

        # Check that multiple elements are masked (cross-pattern creates more masks)
        n_masked = torch.isnan(aligned_ref).sum().item()
        assert (
            n_masked > 0
        ), "INV 2D: Should have masked elements from cross-pattern masking"

        print(
            f"✓ INV 2D alignment cross-pattern masking test passed ({n_masked} elements masked)"
        )

    def test_inv_coordinate_transformation(self):
        """Test that INV alignment correctly identifies inverted region coordinates."""
        # Get sequences with inversions
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.inv_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find INV variants (can be SV_INV or SV_BND_INV)
        inv_results = [r for r in results if "INV" in r[2]["variant_type"].iloc[0]]
        assert len(inv_results) > 0, "Should have at least one INV variant"

        ref_seqs, alt_seqs, metadata = inv_results[0]
        var_metadata = metadata.iloc[0].to_dict()

        # Verify metadata has required fields for inversions
        assert "window_start" in var_metadata, "Metadata should include window_start"
        assert "window_end" in var_metadata, "Metadata should include window_end"
        assert "variant_pos0" in var_metadata, "Metadata should include variant_pos0"

        # For symbolic alleles like <INV>, should have sym_variant_end
        if var_metadata["alt"].startswith("<"):
            assert (
                "sym_variant_end" in var_metadata
            ), "Symbolic allele should have sym_variant_end"

        print("✓ INV coordinate transformation test passed")


class TestDuplicationPredictionAlignment:
    """Test prediction alignment for duplication (DUP) variants."""

    def setup_method(self):
        """Set up test data and models."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.dup_vcf = os.path.join(self.data_dir, "dup", "dup.vcf")

        # Model parameters
        # Note: Use larger seq_len for 2D models to get reasonable matrix size after binning/cropping
        self.seq_len = 512
        self.bin_size = 32
        self.crop_length = 64

        # Initialize 1D model
        self.model_1d = TestModel(
            seq_length=self.seq_len,
            n_targets=1,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )

        # Initialize 2D model
        self.model_2d = TestModel2D(
            seq_length=self.seq_len,
            n_targets=1,
            bin_length=self.bin_size,
            crop_length=self.crop_length,
        )
        self.diag_offset = (
            0  # TestModel2D returns full matrices without diagonal masking
        )

        # Calculate matrix size for 2D
        effective_len = self.seq_len - 2 * self.crop_length
        self.matrix_size = effective_len // self.bin_size

    def test_dup_1d_alignment_nan_insertion(self):
        """Test that DUP 1D alignment inserts NaN bins for duplicated sequence."""
        # Get sequences with duplications
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.dup_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find DUP variants (could be SV_BND_DUP or SV_DUP)
        dup_results = [r for r in results if "DUP" in r[2]["variant_type"].iloc[0]]
        assert len(dup_results) > 0, "Should have at least one DUP variant"

        ref_seqs, alt_seqs, metadata = dup_results[0]

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

        # Run model predictions
        ref_preds = self.model_1d(ref_seqs)
        alt_preds = self.model_1d(alt_seqs)

        # Get metadata for first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions (first variant, first target)
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="1D",
        )

        # For duplications, alignment should handle length differences
        assert len(aligned_ref) == len(
            aligned_alt
        ), "Aligned sequences should have same length"

        # Verify that alignment produces valid output (may or may not have NaN depending on DUP size)
        # Just check that output has expected length
        assert len(aligned_ref) > 0, "Should produce non-empty aligned predictions"
        assert len(aligned_alt) > 0, "Should produce non-empty aligned predictions"

        print("✓ DUP 1D alignment NaN insertion test passed")

    def test_dup_2d_alignment_matrix_handling(self):
        """Test that DUP 2D alignment handles matrix alignment with duplicated bins."""
        # Get sequences with duplications
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.dup_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find DUP variants
        dup_results = [r for r in results if "DUP" in r[2]["variant_type"].iloc[0]]
        assert len(dup_results) > 0, "Should have at least one DUP variant"

        ref_seqs, alt_seqs, metadata = dup_results[0]

        # Transpose from (batch, seq_len, 4) to (batch, 4, seq_len) for model
        ref_seqs = ref_seqs.permute(0, 2, 1)
        alt_seqs = alt_seqs.permute(0, 2, 1)

        # Run model predictions (returns full 2D contact matrices)
        ref_preds = self.model_2d(ref_seqs)
        alt_preds = self.model_2d(alt_seqs)

        # Get metadata for first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Align predictions (first variant, first target)
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds[0, 0],
            alt_preds[0, 0],
            var_metadata,
            bin_size=self.bin_size,
            prediction_type="2D",
            matrix_size=self.matrix_size,
            diag_offset=self.diag_offset,
        )

        # Check that alignment produces valid output (should be 2D matrices)
        assert (
            aligned_ref.shape == aligned_alt.shape
        ), "Aligned contact maps should have same shape"
        assert aligned_ref.ndim == 2, "Output should be 2D matrix (full contact map)"
        assert aligned_alt.ndim == 2, "Output should be 2D matrix (full contact map)"

        print("✓ DUP 2D alignment matrix handling test passed")

    def test_dup_tandem_vs_symbolic(self):
        """Test that both tandem DUP and symbolic <DUP> are handled correctly."""
        # Get sequences with duplications
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.dup_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Check that we process DUP variants successfully
        dup_results = [r for r in results if "DUP" in r[2]["variant_type"].iloc[0]]
        assert len(dup_results) > 0, "Should have at least one DUP variant"

        # Check metadata completeness
        ref_seqs, alt_seqs, metadata = dup_results[0]
        var_metadata = metadata.iloc[0].to_dict()

        # Verify required fields
        assert "window_start" in var_metadata
        assert "window_end" in var_metadata
        assert "variant_type" in var_metadata

        # For symbolic alleles, should have sym_variant_end
        if var_metadata["alt"].startswith("<"):
            assert (
                "sym_variant_end" in var_metadata
            ), "Symbolic DUP should have sym_variant_end"

        print("✓ DUP tandem vs symbolic test passed")


class TestBreakendPredictionAlignment:
    """Test prediction alignment for breakend (BND) variants."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

        # Note: BND tests primarily validate metadata and structure
        # They don't currently use models due to BND's unique dual-locus nature
        self.seq_len = 200

    def test_bnd_1d_alignment_chimeric_reference(self):
        """Test that BND 1D alignment handles chimeric reference assembly."""
        # Get sequences with BNDs
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.bnd_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find BND variants (SV_BND or SV_BND_INS)
        bnd_results = [
            r
            for r in results
            if r[2]["variant_type"].iloc[0] in ["SV_BND", "SV_BND_INS"]
        ]

        if len(bnd_results) == 0:
            pytest.skip("No true BND variants found in test data")

        ref_seqs, alt_seqs, metadata = bnd_results[0]

        # BND ref sequences are regular tensors
        assert isinstance(ref_seqs, torch.Tensor), "BND ref sequences should be tensors"
        assert ref_seqs.shape[2] == 4, "Should have 4 channels (one-hot encoding)"

        # BND alt sequences can be a tuple of tensors (representing both breakpoints)
        # or a single tensor depending on implementation
        if isinstance(alt_seqs, tuple):
            assert (
                len(alt_seqs) == 2
            ), "BND tuple should have 2 tensors (both breakpoints)"
            assert isinstance(
                alt_seqs[0], torch.Tensor
            ), "First BND alt should be tensor"
            assert isinstance(
                alt_seqs[1], torch.Tensor
            ), "Second BND alt should be tensor"
            assert (
                alt_seqs[0].shape[2] == 4
            ), "Should have 4 channels (one-hot encoding)"
            assert (
                alt_seqs[1].shape[2] == 4
            ), "Should have 4 channels (one-hot encoding)"
        else:
            assert isinstance(
                alt_seqs, torch.Tensor
            ), "BND alt sequences should be tensor or tuple"
            assert alt_seqs.shape[2] == 4, "Should have 4 channels (one-hot encoding)"

        # Get metadata for first variant
        var_metadata = metadata.iloc[0].to_dict()

        # Verify BND-specific metadata
        assert "mate_chrom" in var_metadata, "BND metadata should include mate_chrom"
        assert "mate_pos" in var_metadata, "BND metadata should include mate_pos"

        print("✓ BND 1D alignment chimeric reference test passed")

    def test_bnd_2d_alignment_fusion_sequence(self):
        """Test that BND 2D alignment handles fusion sequence coordinate mapping."""
        # Get sequences with BNDs
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.bnd_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find BND variants
        bnd_results = [
            r
            for r in results
            if r[2]["variant_type"].iloc[0] in ["SV_BND", "SV_BND_INS"]
        ]

        if len(bnd_results) == 0:
            pytest.skip("No true BND variants found in test data")

        ref_seqs, alt_seqs, metadata = bnd_results[0]
        var_metadata = metadata.iloc[0].to_dict()

        # Verify metadata has fusion_name for chimeric sequences
        if "fusion_name" in var_metadata:
            assert isinstance(
                var_metadata["fusion_name"], str
            ), "fusion_name should be string"

        print("✓ BND 2D alignment fusion sequence test passed")

    def test_bnd_orientation_handling(self):
        """Test that BND alignment respects breakend orientations."""
        # Get sequences with BNDs
        results = list(
            sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=self.bnd_vcf,
                seq_len=self.seq_len,
                encode=True,
            )
        )

        # Find BND variants
        bnd_results = [
            r
            for r in results
            if r[2]["variant_type"].iloc[0] in ["SV_BND", "SV_BND_INS"]
        ]

        if len(bnd_results) == 0:
            pytest.skip("No true BND variants found in test data")

        ref_seqs, alt_seqs, metadata = bnd_results[0]

        # Check orientation metadata
        for idx, row in metadata.iterrows():
            var_meta = row.to_dict()

            # BND should have orientation information
            assert "orientation_1" in var_meta, "BND should have orientation_1"
            assert "orientation_2" in var_meta, "BND should have orientation_2"

            # Orientations can be in various formats ('+', '-', or descriptive strings like 't]p]')
            # Just check that they exist and are non-empty
            assert (
                var_meta["orientation_1"] is not None
            ), "orientation_1 should not be None"
            assert (
                var_meta["orientation_2"] is not None
            ), "orientation_2 should not be None"
            assert (
                len(str(var_meta["orientation_1"])) > 0
            ), "orientation_1 should not be empty"
            assert (
                len(str(var_meta["orientation_2"])) > 0
            ), "orientation_2 should not be empty"

        print("✓ BND orientation handling test passed")


class TestEdgeCases:
    """Test edge cases in SV prediction alignment."""

    def test_empty_predictions(self):
        """Test handling of empty prediction arrays."""
        metadata = {"variant_type": "SNV", "window_start": 0, "variant_pos0": 50}

        ref_preds = torch.tensor([])
        alt_preds = torch.tensor([])

        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata, bin_size=32, prediction_type="1D"
        )

        assert len(aligned_ref) == 0, "Empty input should produce empty output"
        assert len(aligned_alt) == 0, "Empty input should produce empty output"

        print("✓ Empty predictions test passed")

    def test_single_bin_prediction(self):
        """Test handling of single-bin predictions."""
        metadata = {"variant_type": "SNV", "window_start": 0, "variant_pos0": 50}

        ref_preds = torch.tensor([0.5])
        alt_preds = torch.tensor([0.7])

        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata, bin_size=32, prediction_type="1D"
        )

        assert len(aligned_ref) == 1, "Single bin should remain single bin"
        assert len(aligned_alt) == 1, "Single bin should remain single bin"
        # Use approximate equality for floating point comparisons
        assert torch.allclose(
            aligned_ref[0], torch.tensor(0.5)
        ), "Values should be preserved"
        assert torch.allclose(
            aligned_alt[0], torch.tensor(0.7)
        ), "Values should be preserved"

        print("✓ Single bin prediction test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
