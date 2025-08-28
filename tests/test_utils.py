"""
Test utilities for supremo_lite model prediction alignment tests.

This module contains mock models and test-specific functions that should not
be part of the main package.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class MockModel(nn.Module):
    """
    Mock PyTorch model for testing prediction alignment functionality.
    
    This model generates random predictions for testing purposes and should
    not be used for actual genomics analysis.
    """
    
    def __init__(self, seq_length, bin_length, crop_length=0, n_targets=1, seed=0):
        super().__init__()

        self.seq_length = seq_length
        self.bin_length = bin_length
        self.crop_length = crop_length
        self.n_targets = n_targets

        self.crop_bins = crop_length // bin_length
        self.n_initial_bins = seq_length // bin_length
        self.n_final_bins = self.n_initial_bins - 2 * self.crop_bins

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        assert x.shape[1] == 4
        assert x.shape[2] == self.seq_length

        return torch.rand([x.shape[0], self.n_final_bins, self.n_targets])

    # For intuition only
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.crop_bins:-self.crop_bins, :]
        assert y.shape[1] == self.n_final_bins
        assert y.shape[2] == self.n_targets

        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        return {'loss': loss}


def create_test_predictions_dataset(
    genome_dict: Dict[str, str],
    vcf_df: pd.DataFrame,
    seq_len: int = 1024,
    bin_length: int = 32,
    crop_length: int = 16,
    n_targets: int = 1,
    seed: int = 42
) -> Dict:
    """
    Create a test dataset with predictions using the MockModel.
    
    This function is useful for testing and development without requiring
    a pre-trained model.
    
    Args:
        genome_dict: Dictionary mapping chromosome names to sequences
        vcf_df: DataFrame with variant information
        seq_len: Length of sequence windows
        bin_length: Base pairs per prediction bin
        crop_length: Crop length in base pairs
        n_targets: Number of prediction targets
        seed: Random seed for reproducible results
        
    Returns:
        Dictionary containing predictions and aligned results
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for MockModel")
    
    # Import here to avoid circular imports
    from supremo_lite.model_prediction_alignment import predict_and_align_variants
    
    # Create test model
    model = MockModel(
        seq_length=seq_len,
        bin_length=bin_length, 
        crop_length=crop_length,
        n_targets=n_targets,
        seed=seed
    )
    
    # Generate predictions and align
    results = predict_and_align_variants(
        model, genome_dict, vcf_df,
        seq_len, bin_length, crop_length
    )
    
    return {
        'model': model,
        'results': results,
        'model_params': {
            'seq_length': seq_len,
            'bin_length': bin_length,
            'crop_length': crop_length,
            'n_targets': n_targets,
            'seed': seed
        }
    }


# Test runner for development
if __name__ == '__main__':
    if TORCH_AVAILABLE:
        batch_size = 8
        seq_length = 512 * 1024
        bin_length = 32
        crop_length = 5120 * bin_length
        n_targets = 2

        crop_bins = crop_length // bin_length
        n_initial_bins = seq_length // bin_length
        n_final_bins = n_initial_bins - 2 * crop_bins

        m = MockModel(seq_length, bin_length, crop_length, n_targets)
        x = torch.rand([batch_size, 4, seq_length])

        y_hat = m(x)
        assert y_hat.shape[0] == batch_size
        assert y_hat.shape[1] == n_final_bins
        assert y_hat.shape[2] == n_targets
        print(f"MockModel test passed: {y_hat.shape}")
    else:
        print("PyTorch not available - skipping MockModel test")