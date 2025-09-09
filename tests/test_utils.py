"""
Test utilities for supremo_lite model prediction alignment tests.

This module contains mock models and test-specific functions that should not
be part of the main package.
"""

import numpy as np
import pandas as pd
from typing import Dict

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
    
    This demonstrates the proper user workflow:
    1. Get sequences with get_alt_ref_sequences
    2. Run model to get predictions
    3. Use alignment functions to align predictions
    
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
    from supremo_lite.personalize import get_alt_ref_sequences
    from supremo_lite.model_prediction_alignment import align_variant_predictions
    from supremo_lite.variant_utils import classify_variant_type
    
    # Step 1: Create test model
    model = MockModel(
        seq_length=seq_len,
        bin_length=bin_length, 
        crop_length=crop_length,
        n_targets=n_targets,
        seed=seed
    )
    
    # Step 2: Generate sequences (this is what users do)
    sequences_generator = get_alt_ref_sequences(
        genome_dict, vcf_df, 
        seq_len=seq_len,
        encode=True,
        n_chunks=1
    )
    
    alt_sequences, ref_sequences, metadata = next(sequences_generator)
    
    # Add variant type classification to metadata
    metadata['variant_type'] = metadata.apply(
        lambda row: classify_variant_type(row['ref'], row['alt']), axis=1
    )
    
    # Step 3: Run model to get predictions (users do this with their own model)
    model.eval()
    ref_predictions = []
    alt_predictions = []
    
    with torch.no_grad():
        for i in range(len(ref_sequences)):
            # Convert to tensors and add batch dimension
            if isinstance(ref_sequences[i], torch.Tensor):
                ref_tensor = ref_sequences[i].unsqueeze(0).float()
                alt_tensor = alt_sequences[i].unsqueeze(0).float()
            else:
                ref_tensor = torch.from_numpy(ref_sequences[i]).unsqueeze(0).float()
                alt_tensor = torch.from_numpy(alt_sequences[i]).unsqueeze(0).float()
            
            # Ensure correct tensor shape
            if ref_tensor.shape[-1] == 4 and ref_tensor.shape[1] != 4:
                ref_tensor = ref_tensor.transpose(1, 2)
                alt_tensor = alt_tensor.transpose(1, 2)
            
            ref_pred = model(ref_tensor).cpu().numpy().squeeze(0)
            alt_pred = model(alt_tensor).cpu().numpy().squeeze(0)
            
            ref_predictions.append(ref_pred)
            alt_predictions.append(alt_pred)
    
    # Step 4: Use alignment functions (this is what users should do)
    results = align_variant_predictions(
        np.array(ref_predictions),
        np.array(alt_predictions), 
        metadata,
        bin_length, crop_length
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
