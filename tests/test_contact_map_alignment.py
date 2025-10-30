#!/usr/bin/env python3
"""
Test script for contact map vs score alignment implementation.

Tests prediction alignment functionality with both 1D scores and 2D contact map predictions
using TestModel and TestModel2D from supremo_lite.mock_models.
"""

import os
import torch
import supremo_lite as sl
from supremo_lite.mock_models import TestModel, TestModel2D
from supremo_lite.prediction_alignment import (
    align_predictions_by_coordinate,
    vector_to_contact_matrix,
    contact_matrix_to_vector,
)


def test_1d_score_alignment():
    """Test alignment of 1D prediction scores using TestModel."""
    print("Testing 1D score alignment with TestModel...")

    # Setup paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    reference_fa = os.path.join(data_dir, "test_genome.fa")
    snp_vcf = os.path.join(data_dir, "snp", "snp.vcf")
    ins_vcf = os.path.join(data_dir, "ins", "ins.vcf")

    # Model parameters
    seq_len = 200
    bin_size = 32
    crop_length = 16

    # Initialize model
    model = TestModel(
        seq_length=seq_len, n_targets=1, bin_length=bin_size, crop_length=crop_length
    )

    # Test SNV alignment
    print("\n  Testing SNV alignment...")
    results = list(
        sl.get_alt_ref_sequences(
            reference_fn=reference_fa, variants_fn=snp_vcf, seq_len=seq_len, encode=True
        )
    )

    ref_seqs, alt_seqs, metadata = results[0]

    # Sequences are already in (batch, 4, seq_len) format - no permute needed!

    # Run predictions
    ref_preds = model(ref_seqs)
    alt_preds = model(alt_seqs)

    # Get metadata for first variant
    var_metadata = metadata.iloc[0].to_dict()

    # Align predictions
    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_preds[0, 0],
        alt_preds[0, 0],
        var_metadata,
        bin_size=bin_size,
        prediction_type="1D",
    )

    # Verify outputs are tensors
    assert isinstance(aligned_ref, torch.Tensor), "Output should be tensor"
    assert isinstance(aligned_alt, torch.Tensor), "Output should be tensor"

    # For SNV, aligned should be identical to original input (same length, no NaN)
    assert len(aligned_ref) == len(ref_preds[0, 0]), "SNV should preserve length"
    assert len(aligned_alt) == len(alt_preds[0, 0]), "SNV should preserve length"
    print("  ✓ SNV alignment test passed")

    # Test insertion alignment
    print("\n  Testing insertion alignment...")
    results = list(
        sl.get_alt_ref_sequences(
            reference_fn=reference_fa, variants_fn=ins_vcf, seq_len=seq_len, encode=True
        )
    )

    ref_seqs, alt_seqs, metadata = results[0]

    # Sequences are already in (batch, 4, seq_len) format - no permute needed!

    ref_preds = model(ref_seqs)
    alt_preds = model(alt_seqs)
    var_metadata = metadata.iloc[0].to_dict()

    aligned_ref_ins, aligned_alt_ins = align_predictions_by_coordinate(
        ref_preds[0, 0],
        alt_preds[0, 0],
        var_metadata,
        bin_size=bin_size,
        prediction_type="1D",
    )

    # Verify output types
    assert isinstance(aligned_ref_ins, torch.Tensor), "Output should be tensor"
    assert isinstance(aligned_alt_ins, torch.Tensor), "Output should be tensor"
    print("  ✓ Insertion alignment test passed")


def test_contact_map_utilities():
    """Test contact map utility functions."""
    print("\nTesting contact map utilities with tensors...")

    # Test vector to matrix conversion for small 3x3 matrix
    # Upper triangular: [M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]]
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix = vector_to_contact_matrix(vector, 3)

    # Verify output type
    assert isinstance(matrix, torch.Tensor), "Output should be tensor"

    expected_matrix = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])

    assert torch.allclose(matrix, expected_matrix), "Vector to matrix conversion failed"
    print("  ✓ Vector to matrix conversion test passed")

    # Test matrix to vector conversion
    recovered_vector = contact_matrix_to_vector(matrix)
    assert isinstance(recovered_vector, torch.Tensor), "Output should be tensor"

    assert torch.allclose(
        vector, recovered_vector
    ), "Matrix to vector conversion failed"
    print("  ✓ Matrix to vector conversion test passed")


def test_contact_map_alignment():
    """Test contact map alignment for different variant types using TestModel2D."""
    print("\nTesting contact map alignment with TestModel2D...")

    # Setup paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    reference_fa = os.path.join(data_dir, "test_genome.fa")
    snp_vcf = os.path.join(data_dir, "snp", "snp.vcf")
    ins_vcf = os.path.join(data_dir, "ins", "ins.vcf")
    del_vcf = os.path.join(data_dir, "del", "del.vcf")

    # Model parameters
    # Note: Use larger seq_len for 2D models to get reasonable matrix size
    seq_len = 512
    bin_size = 32
    crop_length = 64
    diag_offset = 0  # TestModel2D returns full matrices without diagonal masking

    # Initialize 2D model
    model = TestModel2D(
        seq_length=seq_len, n_targets=1, bin_length=bin_size, crop_length=crop_length
    )

    # Calculate matrix size
    effective_len = seq_len - 2 * crop_length
    matrix_size = effective_len // bin_size

    print(f"Contact map matrix size: {matrix_size}x{matrix_size}")

    # Test SNV alignment
    print("\n  Testing SNV contact map alignment...")
    results = list(
        sl.get_alt_ref_sequences(
            reference_fn=reference_fa, variants_fn=snp_vcf, seq_len=seq_len, encode=True
        )
    )

    ref_seqs, alt_seqs, metadata = results[0]

    # Sequences are already in (batch, 4, seq_len) format - no permute needed!

    ref_preds = model(ref_seqs)
    alt_preds = model(alt_seqs)
    var_metadata = metadata.iloc[0].to_dict()

    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_preds[0, 0],
        alt_preds[0, 0],
        var_metadata,
        bin_size=bin_size,
        prediction_type="2D",
        matrix_size=matrix_size,
        diag_offset=diag_offset,
    )

    # For SNV, shapes should match and be 2D (full contact matrix)
    assert aligned_ref.shape == aligned_alt.shape, "Shapes should match"
    assert aligned_ref.ndim == 2, "Should be 2D matrix (full contact map)"
    assert aligned_alt.ndim == 2, "Should be 2D matrix (full contact map)"
    assert aligned_ref.shape == (matrix_size, matrix_size), "Should match matrix_size"
    print("  ✓ Contact map SNV alignment test passed")

    # Test insertion alignment
    print("\n  Testing insertion contact map alignment...")
    results = list(
        sl.get_alt_ref_sequences(
            reference_fn=reference_fa, variants_fn=ins_vcf, seq_len=seq_len, encode=True
        )
    )

    ref_seqs, alt_seqs, metadata = results[0]

    # Sequences are already in (batch, 4, seq_len) format - no permute needed!

    ref_preds = model(ref_seqs)
    alt_preds = model(alt_seqs)
    var_metadata = metadata.iloc[0].to_dict()

    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_preds[0, 0],
        alt_preds[0, 0],
        var_metadata,
        bin_size=bin_size,
        prediction_type="2D",
        matrix_size=matrix_size,
        diag_offset=diag_offset,
    )

    print(
        f"  INS aligned contact map shapes: ref={aligned_ref.shape}, alt={aligned_alt.shape}"
    )
    assert aligned_ref.shape == aligned_alt.shape, "Shapes should match after alignment"
    assert aligned_ref.ndim == 2, "Should be 2D matrix (full contact map)"
    assert aligned_alt.ndim == 2, "Should be 2D matrix (full contact map)"
    print("  ✓ Contact map insertion alignment test passed")

    # Test deletion alignment
    print("\n  Testing deletion contact map alignment...")
    results = list(
        sl.get_alt_ref_sequences(
            reference_fn=reference_fa, variants_fn=del_vcf, seq_len=seq_len, encode=True
        )
    )

    ref_seqs, alt_seqs, metadata = results[0]

    # Sequences are already in (batch, 4, seq_len) format - no permute needed!

    ref_preds = model(ref_seqs)
    alt_preds = model(alt_seqs)
    var_metadata = metadata.iloc[0].to_dict()

    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_preds[0, 0],
        alt_preds[0, 0],
        var_metadata,
        bin_size=bin_size,
        prediction_type="2D",
        matrix_size=matrix_size,
        diag_offset=diag_offset,
    )

    print(
        f"  DEL aligned contact map shapes: ref={aligned_ref.shape}, alt={aligned_alt.shape}"
    )
    assert aligned_ref.shape == aligned_alt.shape, "Shapes should match after alignment"
    assert aligned_ref.ndim == 2, "Should be 2D matrix (full contact map)"
    assert aligned_alt.ndim == 2, "Should be 2D matrix (full contact map)"
    print("  ✓ Contact map deletion alignment test passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting error handling...")

    ref_preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    alt_preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    metadata = {"variant_type": "SNV"}

    # Test invalid prediction type
    try:
        align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata, bin_size=32, prediction_type="invalid"
        )
        assert False, "Should have raised ValueError for invalid prediction type"
    except ValueError as e:
        print(f"  ✓ Caught expected error for invalid prediction type: {e}")

    # Test missing matrix_size for 2D
    try:
        align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata, bin_size=32, prediction_type="2D"
        )
        assert False, "Should have raised ValueError for missing matrix_size"
    except ValueError as e:
        print(f"  ✓ Caught expected error for missing matrix_size: {e}")

    # Test 2D without diag_offset (should default to 0)
    # Create proper contact map data (3x3 matrix = 6 elements in upper triangular)
    ref_contact_test = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    alt_contact_test = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    metadata_contact = {
        "variant_type": "SNV",
        "window_start": 0,
        "variant_pos0": 50,
        "svlen": 0,
    }
    try:
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_contact_test,
            alt_contact_test,
            metadata_contact,
            bin_size=32,
            prediction_type="2D",
            matrix_size=3,
            # diag_offset not specified, should default to 0
        )
        print("  ✓ 2D alignment works without diag_offset (defaults to 0)")
    except Exception as e:
        print(f"  ✗ Unexpected error with default diag_offset: {e}")
        raise

    # Test unsupported variant type for 2D predictions
    # Create a 3x3 contact matrix
    ref_contact_test_matrix = torch.ones((3, 3))
    alt_contact_test_matrix = torch.ones((3, 3)) * 1.1
    metadata_unsupported = {
        "variant_type": "BND",
        "window_start": 0,
        "variant_pos0": 50,
        "svlen": 0,
    }
    try:
        align_predictions_by_coordinate(
            ref_contact_test_matrix,
            alt_contact_test_matrix,
            metadata_unsupported,
            bin_size=32,
            prediction_type="2D",
            matrix_size=3,
            diag_offset=0,
        )
        assert False, "Should have raised ValueError for unsupported variant type"
    except ValueError as e:
        print(f"  ✓ Caught expected error for unsupported variant: {e}")


if __name__ == "__main__":
    print("Contact Map vs Score Alignment Test Suite")
    print("=" * 50)

    try:
        test_1d_score_alignment()
        test_contact_map_utilities()
        test_contact_map_alignment()
        test_error_handling()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! Contact map alignment implementation is working.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
