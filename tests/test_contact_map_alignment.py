#!/usr/bin/env python3
"""
Simple test script for contact map vs score alignment implementation.

This script tests the new prediction alignment functionality with both
1D scores and 2D contact map predictions.
"""

import numpy as np

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch detected - will test both NumPy and PyTorch")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - testing NumPy only")

from supremo_lite.prediction_alignment import (
    align_predictions_by_coordinate,
    vector_to_contact_matrix,
    contact_matrix_to_vector
)


def create_test_data(data, use_torch=False):
    """Convert NumPy array to PyTorch tensor if requested and available."""
    if use_torch and TORCH_AVAILABLE:
        return torch.from_numpy(data).float()
    return data


def verify_output_type(output, input_was_torch):
    """Verify that output type matches input type."""
    if input_was_torch and TORCH_AVAILABLE:
        assert hasattr(output, 'device'), f"Expected PyTorch tensor, got {type(output)}"
        print("  ✓ PyTorch tensor output verified")
    else:
        assert isinstance(output, np.ndarray), f"Expected NumPy array, got {type(output)}"
        print("  ✓ NumPy array output verified")


def test_1d_score_alignment():
    """Test alignment of 1D prediction scores."""
    print("Testing 1D score alignment...")
    
    # Test with both NumPy and PyTorch (if available)
    for use_torch in [False, True]:
        if use_torch and not TORCH_AVAILABLE:
            continue
            
        data_type = "PyTorch" if use_torch else "NumPy"
        print(f"\n  Testing with {data_type} data...")
        
        # Create mock 1D prediction arrays
        ref_preds_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        alt_preds_np = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
        
        ref_preds = create_test_data(ref_preds_np, use_torch)
        alt_preds = create_test_data(alt_preds_np, use_torch)
        
        # Mock metadata for SNV (using new format)
        metadata_snv = {
            'variant_type': 'SNV',
            'window_start': 0,
            'effective_variant_start': 100,
            'svlen': 0
        }
        
        # Test SNV alignment (should be direct copy)
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata_snv,
            bin_size=32, prediction_type="1D"
        )
        
        # Verify output types match input types
        verify_output_type(aligned_ref, use_torch)
        verify_output_type(aligned_alt, use_torch)
        
        # Store original input for proper comparison
        original_ref = ref_preds
        original_alt = alt_preds
        
        # Convert both original and aligned to numpy for comparison
        if use_torch and TORCH_AVAILABLE:
            original_ref_np = original_ref.detach().cpu().numpy()
            original_alt_np = original_alt.detach().cpu().numpy()
            aligned_ref_np = aligned_ref.detach().cpu().numpy()
            aligned_alt_np = aligned_alt.detach().cpu().numpy()
        else:
            original_ref_np = original_ref
            original_alt_np = original_alt
            aligned_ref_np = aligned_ref
            aligned_alt_np = aligned_alt
        
        # For SNV, aligned should be identical to original input
        assert np.array_equal(original_ref_np, aligned_ref_np), f"SNV ref alignment failed for {data_type}"
        assert np.array_equal(original_alt_np, aligned_alt_np), f"SNV alt alignment failed for {data_type}"
        print(f"  ✓ {data_type} SNV alignment test passed")
        
        # Test insertion alignment
        metadata_ins = {
            'variant_type': 'INS',
            'window_start': 0,
            'effective_variant_start': 100,
            'svlen': 32  # 1 bin insertion
        }

        aligned_ref_ins, aligned_alt_ins = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata_ins,
            bin_size=32, prediction_type="1D"
        )
        
        # Verify output types for insertion test
        verify_output_type(aligned_ref_ins, use_torch)
        verify_output_type(aligned_alt_ins, use_torch)
        print(f"  ✓ {data_type} insertion alignment test passed")


def test_contact_map_utilities():
    """Test contact map utility functions."""
    print("\nTesting contact map utilities...")
    
    # Test with both NumPy and PyTorch (if available)
    for use_torch in [False, True]:
        if use_torch and not TORCH_AVAILABLE:
            continue
            
        data_type = "PyTorch" if use_torch else "NumPy"
        print(f"\n  Testing {data_type} contact map utilities...")
        
        # Test vector to matrix conversion for small 3x3 matrix
        # Upper triangular: [M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]]
        vector_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        vector = create_test_data(vector_np, use_torch)
        matrix = vector_to_contact_matrix(vector, 3)
        
        # Verify output type
        verify_output_type(matrix, use_torch)
        
        expected_matrix = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0]
        ])
        
        # Convert to numpy for comparison if needed
        if use_torch and TORCH_AVAILABLE:
            matrix_np = matrix.detach().cpu().numpy()
        else:
            matrix_np = matrix
        
        assert np.allclose(matrix_np, expected_matrix), f"Vector to matrix conversion failed for {data_type}"
        print(f"  ✓ {data_type} vector to matrix conversion test passed")
        
        # Test matrix to vector conversion
        recovered_vector = contact_matrix_to_vector(matrix)
        verify_output_type(recovered_vector, use_torch)
        
        # Store original input for proper comparison
        original_vector = vector
        
        # Convert both original and recovered to numpy for comparison
        if use_torch and TORCH_AVAILABLE:
            original_vector_np = original_vector.detach().cpu().numpy()
            recovered_vector_np = recovered_vector.detach().cpu().numpy()
        else:
            original_vector_np = original_vector
            recovered_vector_np = recovered_vector
        
        assert np.allclose(original_vector_np, recovered_vector_np), f"Matrix to vector conversion failed for {data_type}"
        print(f"  ✓ {data_type} matrix to vector conversion test passed")


def test_contact_map_alignment():
    """Test contact map alignment for different variant types."""
    print("\nTesting contact map alignment...")
    
    # Create mock contact map predictions (flattened upper triangular)
    matrix_size = 4
    n_elements = matrix_size * (matrix_size + 1) // 2  # 10 elements for 4x4 upper triangular
    
    ref_contact_preds = np.random.rand(n_elements)
    alt_contact_preds = np.random.rand(n_elements)
    
    print(f"Contact map vector length: {len(ref_contact_preds)} (for {matrix_size}x{matrix_size} matrix)")
    
    # Test SNV alignment
    metadata_snv = {
        'variant_type': 'SNV',
        'window_start': 0,
        'effective_variant_start': 50,
        'svlen': 0
    }

    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_contact_preds, alt_contact_preds, metadata_snv,
        bin_size=32,
        prediction_type="2D", matrix_size=matrix_size, diag_offset=2
    )
    
    # For SNV, should be identical to input
    assert np.array_equal(ref_contact_preds, aligned_ref), "Contact map SNV ref alignment failed"
    assert np.array_equal(alt_contact_preds, aligned_alt), "Contact map SNV alt alignment failed"
    print("✓ Contact map SNV alignment test passed")
    
    # Test insertion alignment
    metadata_ins = {
        'variant_type': 'INS',
        'window_start': 0,
        'effective_variant_start': 50,
        'svlen': 32
    }

    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_contact_preds, alt_contact_preds, metadata_ins,
        bin_size=32,
        prediction_type="2D", matrix_size=matrix_size, diag_offset=2
    )
    
    print(f"INS aligned contact map shapes: ref={aligned_ref.shape}, alt={aligned_alt.shape}")
    assert len(aligned_ref) == len(ref_contact_preds), "Contact map insertion ref length mismatch"
    assert len(aligned_alt) == len(alt_contact_preds), "Contact map insertion alt length mismatch"
    print("✓ Contact map insertion alignment test passed")
    
    # Test deletion alignment
    metadata_del = {
        'variant_type': 'DEL',
        'window_start': 0,
        'effective_variant_start': 50,
        'svlen': -32  # Negative for deletion
    }

    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_contact_preds, alt_contact_preds, metadata_del,
        bin_size=32,
        prediction_type="2D", matrix_size=matrix_size, diag_offset=2
    )
    
    print(f"DEL aligned contact map shapes: ref={aligned_ref.shape}, alt={aligned_alt.shape}")
    assert len(aligned_ref) == len(ref_contact_preds), "Contact map deletion ref length mismatch"
    assert len(aligned_alt) == len(alt_contact_preds), "Contact map deletion alt length mismatch"
    print("✓ Contact map deletion alignment test passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting error handling...")
    
    ref_preds = np.array([1, 2, 3, 4])
    alt_preds = np.array([1, 2, 3, 4])
    metadata = {'variant_type': 'SNV'}
    
    # Test invalid prediction type
    try:
        align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata,
            bin_size=32, prediction_type="invalid"
        )
        assert False, "Should have raised ValueError for invalid prediction type"
    except ValueError as e:
        print(f"✓ Caught expected error for invalid prediction type: {e}")

    # Test missing matrix_size for 2D
    try:
        align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata,
            bin_size=32, prediction_type="2D"
        )
        assert False, "Should have raised ValueError for missing matrix_size"
    except ValueError as e:
        print(f"✓ Caught expected error for missing matrix_size: {e}")

    # Test 2D without diag_offset (should default to 0)
    # Create proper contact map data (3x3 matrix = 6 elements in upper triangular)
    ref_contact_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    alt_contact_test = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    metadata_contact = {
        'variant_type': 'SNV',
        'window_start': 0,
        'effective_variant_start': 50,
        'svlen': 0
    }
    try:
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_contact_test, alt_contact_test, metadata_contact,
            bin_size=32, prediction_type="2D", matrix_size=3
            # diag_offset not specified, should default to 0
        )
        print(f"✓ 2D alignment works without diag_offset (defaults to 0)")
    except Exception as e:
        print(f"✗ Unexpected error with default diag_offset: {e}")
        raise

    # Test unsupported variant type for 2D predictions
    metadata_unsupported = {
        'variant_type': 'BND',
        'window_start': 0,
        'effective_variant_start': 50,
        'svlen': 0
    }
    try:
        align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata_unsupported,
            bin_size=32,
            prediction_type="2D", matrix_size=4, diag_offset=2
        )
        assert False, "Should have raised ValueError for unsupported variant type"
    except ValueError as e:
        print(f"✓ Caught expected error for unsupported variant: {e}")


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