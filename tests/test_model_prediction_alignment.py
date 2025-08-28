"""
Tests for model-based prediction alignment functionality.

Tests the integration between supremo_lite's variant processing and 
model prediction alignment using the TestModel.
"""

import pytest
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Import the modules we're testing
from supremo_lite.model_prediction_alignment import (
    generate_predictions_for_variants,
    align_variant_predictions,
    predict_and_align_variants
)
from supremo_lite.prediction_alignment import (
    coord_to_bin_offset,
    bin_offset_to_coord,
    classify_variant_type,
    align_predictions_by_coordinate,
    calculate_variant_effect_scores
)
from supremo_lite.variant_utils import read_vcf
from supremo_lite.sequence_utils import encode_seq

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_genome():
    """Simple test genome for predictions."""
    return {
        'chr1': 'ATGAATATAATATTTTCGAGAATTACTCCTTTTGGAAATGGAACATTATGCGTTTTAAGAGTTTCTGGTAACAATATATT',
        'chr2': 'TATTTCTTATTCGTTTAAAAAAATTAATTTATTTATTTCTAAATTAAAAACGGGAACACCTCCCAGATTGGATTTAAATT',
        'chr3': 'TTCTTTAATAAATACTATTAATAAATTAAAAAAATTTAATTGTTTAGATTATACAATTATTATCGCTTCGACTGCTTCAG'
    }


@pytest.fixture
def simple_vcf_data():
    """Simple VCF data for testing."""
    return pd.DataFrame({
        'chrom': ['chr1', 'chr1', 'chr2'],
        'pos1': [2, 31, 19], 
        'ref': ['T', 'T', 'A'],
        'alt': ['G', 'A', 'G'],
        'id': ['.', '.', '.'],
        'qual': [6.0, 6.0, 6.0],
        'filter': ['PASS', 'PASS', 'PASS'],
        'info': ['DP=100', 'DP=100', 'DP=100']
    })


@pytest.fixture 
def indel_vcf_data():
    """VCF data with insertions and deletions."""
    return pd.DataFrame({
        'chrom': ['chr1', 'chr2'],
        'pos1': [8, 17],
        'ref': ['T', 'CGAGAA'],
        'alt': ['TGGG', 'C'],
        'id': ['.', '.'],
        'qual': [6.0, 6.0],
        'filter': ['PASS', 'PASS'],
        'info': ['DP=100', 'DP=100']
    })


class TestCoordinateConversion:
    """Test coordinate to bin conversion functions."""
    
    def test_coord_to_bin_offset(self):
        """Test genomic coordinate to bin offset conversion."""
        # Test with default parameters (32bp bins, 16 bin crop)
        assert coord_to_bin_offset(512, 0, 32, 16) == 0  # First non-cropped bin
        assert coord_to_bin_offset(544, 0, 32, 16) == 1  # Second bin
        assert coord_to_bin_offset(256, 0, 32, 16) == -8  # In cropped region
        
    def test_bin_offset_to_coord(self):
        """Test bin offset to genomic coordinate conversion."""
        assert bin_offset_to_coord(0, 0, 32, 16) == 512  # First non-cropped bin start
        assert bin_offset_to_coord(1, 0, 32, 16) == 544  # Second bin start
        assert bin_offset_to_coord(-1, 0, 32, 16) == 480  # Last cropped bin
        

class TestVariantClassification:
    """Test variant type classification."""
    
    def test_classify_snv(self):
        """Test SNV classification."""
        assert classify_variant_type('A', 'G') == 'SNV'
        assert classify_variant_type('T', 'C') == 'SNV'
        
    def test_classify_insertion(self):
        """Test insertion classification.""" 
        assert classify_variant_type('T', 'TGGG') == 'insertion'
        assert classify_variant_type('A', 'ATCG') == 'insertion'
        
    def test_classify_deletion(self):
        """Test deletion classification."""
        assert classify_variant_type('CGAGAA', 'C') == 'deletion'
        assert classify_variant_type('ATCG', 'A') == 'deletion'
        
    def test_classify_complex(self):
        """Test complex variant classification."""
        assert classify_variant_type('ATCG', 'GCTA') == 'complex'
        assert classify_variant_type('AT', 'GC') == 'complex'


class TestModelPredictions:
    """Test model-based prediction generation."""
    
    def test_mock_model_creation(self):
        """Test MockModel can be created and run."""
        from tests.test_utils import MockModel
        model = MockModel(seq_length=1024, bin_length=32, crop_length=512, n_targets=1)
        
        # Test forward pass
        x = torch.rand(2, 4, 1024)  # Batch of 2 sequences
        y = model(x)
        
        expected_bins = (1024 // 32) - 2 * (512 // 32)  # 32 - 32 = 0 bins after crop
        assert y.shape == (2, expected_bins, 1)
        
    def test_generate_predictions_for_variants(self, test_genome, simple_vcf_data):
        """Test generating predictions for variants."""
        from tests.test_utils import MockModel
        model = MockModel(seq_length=256, bin_length=32, crop_length=64, n_targets=1)
        
        predictions_data = generate_predictions_for_variants(
            model, test_genome, simple_vcf_data, 
            seq_len=256, bin_length=32, crop_length=64
        )
        
        # Check structure
        assert 'ref_predictions' in predictions_data
        assert 'alt_predictions' in predictions_data
        assert 'metadata' in predictions_data
        
        # Check shapes
        n_variants = len(simple_vcf_data)
        expected_bins = (256 // 32) - 2 * (64 // 32)  # 8 - 4 = 4 bins
        
        assert predictions_data['ref_predictions'].shape == (n_variants, expected_bins, 1)
        assert predictions_data['alt_predictions'].shape == (n_variants, expected_bins, 1)
        assert len(predictions_data['metadata']) == n_variants
        
        # Check metadata has variant type
        assert 'variant_type' in predictions_data['metadata'].columns
        assert all(predictions_data['metadata']['variant_type'] == 'SNV')
        
    def test_create_test_predictions_dataset(self, test_genome, simple_vcf_data):
        """Test creating test predictions dataset."""
        from tests.test_utils import create_test_predictions_dataset
        dataset = create_test_predictions_dataset(
            test_genome, simple_vcf_data,
            seq_len=256, bin_length=32, crop_length=64, n_targets=1
        )
        
        # Check structure
        assert 'model' in dataset
        assert 'results' in dataset
        assert 'model_params' in dataset
        
        # Check results structure
        results = dataset['results']
        assert len(results) == len(simple_vcf_data)
        
        for result in results:
            assert 'variant_id' in result
            assert 'ref_predictions' in result
            assert 'alt_predictions' in result
            assert 'aligned_ref' in result
            assert 'aligned_alt' in result
            assert 'effect_scores' in result


class TestPredictionAlignment:
    """Test prediction alignment functionality."""
    
    def test_align_snv_predictions(self):
        """Test aligning SNV predictions (should be identical)."""
        # Create mock predictions
        ref_preds = np.array([1.0, 2.0, 3.0, 4.0])
        alt_preds = np.array([1.1, 2.1, 3.1, 4.1])
        
        # SNV metadata
        metadata_row = {
            'variant_type': 'SNV',
            'variant_pos0_in_window': 128,
            'position_offset_downstream': 0
        }
        
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata_row, bin_length=32, crop_length=16
        )
        
        # SNVs should have no alignment shift
        np.testing.assert_array_equal(aligned_ref, ref_preds)
        np.testing.assert_array_equal(aligned_alt, alt_preds)
        
    def test_align_insertion_predictions(self):
        """Test aligning insertion predictions."""
        ref_preds = np.array([1.0, 2.0, 3.0, 4.0])
        alt_preds = np.array([1.1, 2.1, 3.1, 4.1])
        
        # Insertion metadata (3bp insertion at position 128)
        metadata_row = {
            'variant_type': 'insertion',
            'variant_pos0_in_window': 128,
            'position_offset_downstream': 3  # 3bp insertion
        }
        
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata_row, bin_length=32, crop_length=16
        )
        
        # Reference should be unchanged
        np.testing.assert_array_equal(aligned_ref, ref_preds)
        
        # Alt should be aligned (small insertion shouldn't shift bins significantly)
        assert len(aligned_alt) == len(alt_preds)
        
    def test_align_deletion_predictions(self):
        """Test aligning deletion predictions.""" 
        ref_preds = np.array([1.0, 2.0, 3.0, 4.0])
        alt_preds = np.array([1.1, 2.1, 3.1, 4.1])
        
        # Deletion metadata (5bp deletion at position 128)
        metadata_row = {
            'variant_type': 'deletion',
            'variant_pos0_in_window': 128,
            'position_offset_downstream': -5  # 5bp deletion
        }
        
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_preds, alt_preds, metadata_row, bin_length=32, crop_length=16
        )
        
        # Reference should be unchanged
        np.testing.assert_array_equal(aligned_ref, ref_preds)
        
        # Alt should be aligned
        assert len(aligned_alt) == len(alt_preds)
        
    def test_calculate_variant_effect_scores(self):
        """Test calculating variant effect scores."""
        ref_preds = np.array([1.0, 2.0, 3.0, 4.0])
        alt_preds = np.array([2.0, 4.0, 6.0, 8.0])  # 2x ref values
        
        # Test log_ratio method
        log_effects = calculate_variant_effect_scores(ref_preds, alt_preds, 'log_ratio')
        expected_log = np.log2(2.0)  # log2(alt/ref) = log2(2) = 1
        np.testing.assert_allclose(log_effects, expected_log, rtol=1e-6)
        
        # Test difference method
        diff_effects = calculate_variant_effect_scores(ref_preds, alt_preds, 'difference')
        expected_diff = alt_preds - ref_preds
        np.testing.assert_array_equal(diff_effects, expected_diff)
        
        # Test fold_change method
        fold_effects = calculate_variant_effect_scores(ref_preds, alt_preds, 'fold_change')
        expected_fold = alt_preds / ref_preds
        np.testing.assert_allclose(fold_effects, expected_fold, rtol=1e-6)


class TestIntegration:
    """Integration tests using real test data."""
    
    def test_snv_integration(self, test_data_dir, test_genome):
        """Test complete SNV processing pipeline."""
        from tests.test_utils import create_test_predictions_dataset
        snv_vcf_path = test_data_dir / "snp" / "snp.vcf"
        assert snv_vcf_path.exists(), f"Test VCF file not found: {snv_vcf_path}"
            
        # Read VCF
        vcf_df = read_vcf(str(snv_vcf_path))
        
        # Create test dataset
        dataset = create_test_predictions_dataset(
            test_genome, vcf_df, 
            seq_len=256, bin_length=32, crop_length=64
        )
        
        results = dataset['results']
        assert len(results) == len(vcf_df)
        
        # Check each result
        for i, result in enumerate(results):
            # Check that we have a variant ID (format will depend on internal column names)
            assert 'variant_id' in result
            assert ':' in result['variant_id']  # Should have chrom:pos:ref>alt format
            
            # Check that we have predictions
            assert result['ref_predictions'].size > 0
            assert result['alt_predictions'].size > 0
            assert result['effect_scores'].size > 0
            
            # For SNVs, aligned predictions should match original
            np.testing.assert_array_equal(result['aligned_ref'], result['ref_predictions'])
            np.testing.assert_array_equal(result['aligned_alt'], result['alt_predictions'])
            
    def test_indel_integration(self, test_data_dir, test_genome):
        """Test integration with insertion/deletion variants."""
        from tests.test_utils import create_test_predictions_dataset
        ins_vcf_path = test_data_dir / "ins" / "ins.vcf"
        assert ins_vcf_path.exists(), f"Test VCF file not found: {ins_vcf_path}"
            
        # Read VCF
        vcf_df = read_vcf(str(ins_vcf_path))
        
        # Create test dataset
        dataset = create_test_predictions_dataset(
            test_genome, vcf_df,
            seq_len=256, bin_length=32, crop_length=64
        )
        
        results = dataset['results']
        assert len(results) == len(vcf_df)
        
        # Check results
        for result in results:
            assert result['metadata']['variant_type'] == 'insertion'
            assert result['metadata']['position_offset_downstream'] > 0
            
            # Should have alignment results
            assert 'aligned_ref' in result
            assert 'aligned_alt' in result
            assert 'effect_scores' in result
            
    def test_deletion_integration(self, test_data_dir, test_genome):
        """Test integration with deletion variants."""
        from tests.test_utils import create_test_predictions_dataset
        del_vcf_path = test_data_dir / "del" / "del.vcf"
        assert del_vcf_path.exists(), f"Test VCF file not found: {del_vcf_path}"
            
        # Read VCF  
        vcf_df = read_vcf(str(del_vcf_path))
        
        # Create test dataset
        dataset = create_test_predictions_dataset(
            test_genome, vcf_df,
            seq_len=256, bin_length=32, crop_length=64
        )
        
        results = dataset['results']
        assert len(results) == len(vcf_df)
        
        # Check results
        for result in results:
            assert result['metadata']['variant_type'] == 'deletion'
            assert result['metadata']['position_offset_downstream'] < 0
            
            # Should have alignment results
            assert 'aligned_ref' in result
            assert 'aligned_alt' in result
            assert 'effect_scores' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_predictions(self):
        """Test handling of empty prediction arrays."""
        with pytest.raises(ValueError):
            calculate_variant_effect_scores(np.array([]), np.array([1.0]))
            
    def test_mismatched_prediction_lengths(self):
        """Test handling of mismatched prediction array lengths."""
        with pytest.raises(ValueError):
            calculate_variant_effect_scores(
                np.array([1.0, 2.0]), np.array([1.0])
            )
            
    def test_invalid_effect_method(self):
        """Test handling of invalid effect calculation method."""
        with pytest.raises(ValueError):
            calculate_variant_effect_scores(
                np.array([1.0]), np.array([2.0]), method='invalid'
            )