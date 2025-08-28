"""
Model-based prediction alignment utilities for supremo_lite.

This module provides functions to generate model predictions for reference and variant
sequences, then align them for variant effect analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import warnings

from .prediction_alignment import (
    align_predictions_by_coordinate, 
    calculate_variant_effect_scores,
    classify_variant_type,
    parse_vcf_info
)
from .personalize import get_alt_ref_sequences
from .sequence_utils import encode_seq

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def generate_predictions_for_variants(
    model,
    genome_dict: Dict[str, str],
    vcf_df: pd.DataFrame,
    seq_len: int = 1024,
    bin_length: int = 32,
    crop_length: int = 16,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Generate model predictions for reference and variant sequences.
    
    Args:
        model: PyTorch model for generating predictions
        genome_dict: Dictionary mapping chromosome names to sequences
        vcf_df: DataFrame with variant information (CHROM, POS, REF, ALT)
        seq_len: Length of sequence windows for model input
        bin_length: Base pairs per prediction bin
        crop_length: Number of bins cropped from each edge during prediction
        device: Device to run model on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with keys:
        - 'ref_predictions': Reference sequence predictions
        - 'alt_predictions': Variant sequence predictions  
        - 'metadata': Metadata DataFrame with variant information
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model predictions")
    
    # Generate reference and variant sequences
    sequences_generator = get_alt_ref_sequences(
        genome_dict, vcf_df, 
        seq_len=seq_len,
        encode=True,
        chunk_size=len(vcf_df)
    )
    
    # Get all sequences at once
    alt_sequences, ref_sequences, metadata = next(sequences_generator)
    
    # Add variant type classification to metadata
    metadata['variant_type'] = metadata.apply(
        lambda row: classify_variant_type(row['ref'], row['alt']), axis=1
    )
    
    # Add downstream position offset calculation
    metadata['ref_length'] = metadata['ref'].str.len()
    metadata['alt_length'] = metadata['alt'].str.len()
    metadata['position_offset_downstream'] = metadata['alt_length'] - metadata['ref_length']
    
    model.eval()
    model.to(device)
    
    ref_predictions = []
    alt_predictions = []
    
    with torch.no_grad():
        for i in range(len(ref_sequences)):
            # Convert to tensors and add batch dimension
            if isinstance(ref_sequences[i], torch.Tensor):
                ref_tensor = ref_sequences[i].unsqueeze(0).float().to(device)
                alt_tensor = alt_sequences[i].unsqueeze(0).float().to(device)
            else:
                ref_tensor = torch.from_numpy(ref_sequences[i]).unsqueeze(0).float().to(device)
                alt_tensor = torch.from_numpy(alt_sequences[i]).unsqueeze(0).float().to(device)
            
            # Ensure tensor is in correct shape: (batch, 4, seq_len)
            # If shape is (batch, seq_len, 4), transpose to (batch, 4, seq_len)
            if ref_tensor.shape[-1] == 4 and ref_tensor.shape[1] != 4:
                ref_tensor = ref_tensor.transpose(1, 2)
                alt_tensor = alt_tensor.transpose(1, 2)
            
            # Generate predictions - only squeeze batch dimension to preserve target dimension
            ref_pred = model(ref_tensor).cpu().numpy().squeeze(0)
            alt_pred = model(alt_tensor).cpu().numpy().squeeze(0)
            
            ref_predictions.append(ref_pred)
            alt_predictions.append(alt_pred)
    
    return {
        'ref_predictions': np.array(ref_predictions),
        'alt_predictions': np.array(alt_predictions),
        'metadata': metadata
    }


def align_variant_predictions(
    ref_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    metadata: pd.DataFrame,
    bin_length: int = 32,
    crop_length: int = 16,
    effect_method: str = 'log_ratio'
) -> List[Dict]:
    """
    Align predictions for multiple variants and calculate effect scores.
    
    Args:
        ref_predictions: Array of reference predictions (n_variants, n_bins, n_targets)
        alt_predictions: Array of variant predictions (n_variants, n_bins, n_targets) 
        metadata: DataFrame with variant metadata
        bin_length: Base pairs per prediction bin
        crop_length: Number of bins cropped from each edge
        effect_method: Method for calculating variant effects
        
    Returns:
        List of dictionaries containing aligned predictions and effect scores for each variant
    """
    results = []
    
    # Debug: Check array lengths match
    if len(ref_predictions) != len(metadata):
        raise ValueError(f"Length mismatch: {len(ref_predictions)} predictions vs {len(metadata)} metadata rows")
    
    for i in range(len(ref_predictions)):
        metadata_row = metadata.iloc[i].to_dict()
        
        # Align predictions for this variant
        aligned_ref, aligned_alt = align_predictions_by_coordinate(
            ref_predictions[i], alt_predictions[i], 
            metadata_row, bin_length, crop_length
        )
        
        # Calculate effect scores
        effect_scores = calculate_variant_effect_scores(
            aligned_ref, aligned_alt, effect_method
        )
        
        # Store results
        variant_result = {
            'variant_id': f"{metadata_row['chrom']}:{metadata_row.get('variant_pos', metadata_row.get('pos1', 'unknown'))}:{metadata_row['ref']}>{metadata_row['alt']}",
            'ref_predictions': ref_predictions[i],
            'alt_predictions': alt_predictions[i],
            'aligned_ref': aligned_ref,
            'aligned_alt': aligned_alt,
            'effect_scores': effect_scores,
            'metadata': metadata_row
        }
        
        results.append(variant_result)
    
    return results


def predict_and_align_variants(
    model,
    genome_dict: Dict[str, str],
    vcf_df: pd.DataFrame,
    seq_len: int = 1024,
    bin_length: int = 32, 
    crop_length: int = 16,
    effect_method: str = 'log_ratio',
    device: str = 'cpu'
) -> List[Dict]:
    """
    Complete workflow: generate predictions and align them for variant effect analysis.
    
    Args:
        model: PyTorch model for predictions
        genome_dict: Dictionary mapping chromosome names to sequences
        vcf_df: DataFrame with variant information
        seq_len: Length of sequence windows
        bin_length: Base pairs per prediction bin
        crop_length: Number of bins cropped from edges
        effect_method: Method for calculating variant effects
        device: Device to run model on
        
    Returns:
        List of dictionaries with aligned predictions and effect scores for each variant
    """
    # Generate predictions
    predictions_data = generate_predictions_for_variants(
        model, genome_dict, vcf_df, seq_len, bin_length, crop_length, device
    )
    
    # Align predictions and calculate effects
    results = align_variant_predictions(
        predictions_data['ref_predictions'],
        predictions_data['alt_predictions'], 
        predictions_data['metadata'],
        bin_length, crop_length, effect_method
    )
    
    return results


