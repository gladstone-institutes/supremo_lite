"""
Utilities for aligning model predictions between reference and variant sequences.

This module provides functions to handle the alignment of ML model predictions
when reference and variant sequences have position offsets due to indels.
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional


def coord_to_bin_offset(coord, window_start, bin_length=32, crop_length=16):
    """
    Convert genomic coordinate to prediction bin offset.
    
    This function maps a genomic coordinate within a sequence window to the 
    corresponding bin index in model predictions, accounting for binning and 
    edge cropping.
    
    Args:
        coord: Genomic coordinate within the window (0-based)
        window_start: Start coordinate of the sequence window (0-based)
        bin_length: Number of base pairs per prediction bin (default: 32)
        crop_length: Number of bins cropped from each edge during prediction (default: 16)
        
    Returns:
        int: Bin offset in the prediction array (can be negative if coord is in cropped region)
        
    Example:
        # For a 1000bp window with 32bp bins and 16-bin cropping:
        # Bin 0 covers positions 512-543 (after cropping first 16 bins = 512bp)
        coord_to_bin_offset(520, 0, 32, 16)  # Returns 0 (first non-cropped bin)
    """
    return (coord - window_start) // bin_length - crop_length


def bin_offset_to_coord(bin_offset, window_start, bin_length=32, crop_length=16):
    """
    Convert prediction bin offset to genomic coordinate.
    
    This function maps a bin index in model predictions back to the corresponding
    genomic coordinate, accounting for binning and edge cropping.
    
    Args:
        bin_offset: Bin index in the prediction array
        window_start: Start coordinate of the sequence window (0-based) 
        bin_length: Number of base pairs per prediction bin (default: 32)
        crop_length: Number of bins cropped from each edge during prediction (default: 16)
        
    Returns:
        int: Genomic coordinate at the start of the bin
        
    Example:
        # For a 1000bp window with 32bp bins and 16-bin cropping:
        bin_offset_to_coord(0, 0, 32, 16)  # Returns 512 (start of first non-cropped bin)
    """
    return (bin_offset + crop_length) * bin_length + window_start


def align_predictions_by_coordinate(ref_preds: np.ndarray, 
                                   alt_preds: np.ndarray,
                                   metadata_row,
                                   bin_length: int = 32,
                                   crop_length: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align reference and alt predictions using coordinate transformation and variant type awareness.
    
    This enhanced function uses precise coordinate mapping to align predictions,
    accounting for different variant types and their effects on sequence coordinates.
    
    Args:
        ref_preds: Reference predictions array (from model with edge cropping)
        alt_preds: Alt predictions array (same length as ref_preds)
        metadata_row: Row from metadata DataFrame with variant information
        bin_length: Number of base pairs per prediction bin (default: 32)
        crop_length: Number of bins cropped from each edge during prediction (default: 16)
        
    Returns:
        Tuple of (aligned_ref_preds, aligned_alt_preds) aligned at bin level
        
    Note:
        This function now works at the prediction bin level rather than expanding
        to base-pair resolution, providing more accurate alignment for model predictions.
    """
    # Extract variant information from metadata
    variant_type = metadata_row.get('variant_type', 'unknown')
    variant_pos_in_window = metadata_row.get('variant_pos0_in_window', metadata_row.get('effective_variant_start', 0))
    downstream_offset = metadata_row.get('position_offset_downstream', 0)
    
    # Convert variant position to bin coordinates
    variant_bin = coord_to_bin_offset(variant_pos_in_window, 
                                     window_start=0,
                                     bin_length=bin_length, 
                                     crop_length=crop_length)
    
    # Apply alignment strategy based on variant type
    if variant_type == 'SNV':
        # SNVs don't change coordinates, direct alignment
        return ref_preds.copy(), alt_preds.copy()
        
    elif variant_type in ['insertion', 'deletion', 'complex']:
        # Handle indel-type variants with coordinate shifts
        return align_indel_variants(ref_preds, alt_preds, variant_bin, downstream_offset, bin_length)
        
    elif variant_type == 'INV':
        # Inversions: sequence is reversed but coordinates maintained
        return align_inversion_variant(ref_preds, alt_preds, variant_bin, metadata_row)
        
    elif variant_type == 'DUP':
        # Duplications: sequence is duplicated, extending downstream
        return align_duplication_variant(ref_preds, alt_preds, variant_bin, metadata_row)
        
    elif variant_type == 'BND':
        # Breakends: complex rearrangements, may need special handling
        return align_breakend_variant(ref_preds, alt_preds, variant_bin, metadata_row)
        
    else:
        # Unknown variant type, fall back to simple alignment
        return ref_preds.copy(), alt_preds.copy()


def align_indel_variants(ref_preds: np.ndarray, alt_preds: np.ndarray, 
                        variant_bin: int, downstream_offset: int,
                        bin_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align predictions for indel variants (insertions, deletions, complex).
    
    Args:
        ref_preds: Reference predictions
        alt_preds: Alt predictions  
        variant_bin: Bin index where variant occurs
        downstream_offset: Coordinate offset for downstream positions
        bin_length: Base pairs per bin
        
    Returns:
        Aligned (ref_preds, alt_preds) accounting for coordinate shifts
    """
    if downstream_offset == 0:
        # No coordinate shift needed
        return ref_preds.copy(), alt_preds.copy()
    
    # Create aligned alt predictions array
    aligned_alt_preds = np.zeros_like(alt_preds)
    
    # Calculate bin shift from coordinate offset
    bin_shift = downstream_offset // bin_length
    
    for i in range(len(alt_preds)):
        if i <= variant_bin:
            # Upstream and at variant: no shift needed
            aligned_alt_preds[i] = alt_preds[i]
        else:
            # Downstream: apply bin shift
            shifted_idx = i - bin_shift
            if 0 <= shifted_idx < len(alt_preds):
                aligned_alt_preds[i] = alt_preds[shifted_idx]
            else:
                # Shifted outside array bounds
                aligned_alt_preds[i] = 0.0
    
    return ref_preds.copy(), aligned_alt_preds


def align_inversion_variant(ref_preds: np.ndarray, alt_preds: np.ndarray,
                           variant_bin: int, metadata_row) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align predictions for inversion variants.
    
    For inversions, the sequence is reversed but coordinates are maintained.
    This may require special handling depending on the model's behavior.
    
    Args:
        ref_preds: Reference predictions
        alt_preds: Alt predictions
        variant_bin: Bin index where inversion starts
        metadata_row: Metadata with inversion details
        
    Returns:
        Aligned predictions for inversion variant
    """
    # For now, treat as no coordinate shift (sequence reversal may not affect predictions)
    # Future enhancement: could reverse prediction values within inverted region
    return ref_preds.copy(), alt_preds.copy()


def align_duplication_variant(ref_preds: np.ndarray, alt_preds: np.ndarray,
                             variant_bin: int, metadata_row) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align predictions for duplication variants.
    
    Duplications extend the sequence, shifting downstream coordinates.
    
    Args:
        ref_preds: Reference predictions
        alt_preds: Alt predictions
        variant_bin: Bin index where duplication occurs
        metadata_row: Metadata with duplication details
        
    Returns:
        Aligned predictions for duplication variant
    """
    # Extract duplication length from metadata
    dup_length = metadata_row.get('alt_length', 0) - metadata_row.get('ref_length', 0)
    
    # Use indel alignment logic with duplication offset
    return align_indel_variants(ref_preds, alt_preds, variant_bin, dup_length, 32)


def align_breakend_variant(ref_preds: np.ndarray, alt_preds: np.ndarray,
                          variant_bin: int, metadata_row) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align predictions for breakend/translocation variants.
    
    Breakends involve complex rearrangements that may be difficult to align precisely.
    
    Args:
        ref_preds: Reference predictions
        alt_preds: Alt predictions
        variant_bin: Bin index where breakend occurs
        metadata_row: Metadata with breakend details
        
    Returns:
        Aligned predictions for breakend variant (may be approximate)
    """
    # For complex breakends, alignment may not be straightforward
    # Fall back to simple alignment for now
    return ref_preds.copy(), alt_preds.copy()


# Legacy function for backward compatibility
def align_predictions_by_coordinate_legacy(ref_preds: np.ndarray, 
                                          alt_preds: np.ndarray,
                                          indel_offset: int,
                                          seq_len: int,
                                          bin_size: int = 10,
                                          edge_trim: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy alignment function for backward compatibility.
    
    DEPRECATED: Use align_predictions_by_coordinate with metadata_row instead.
    
    This function maintains the old interface but with a deprecation warning.
    """
    import warnings
    warnings.warn(
        "align_predictions_by_coordinate with indel_offset parameter is deprecated. "
        "Use the new version with metadata_row parameter for enhanced variant type support.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create minimal metadata for legacy compatibility
    metadata_row = {
        'variant_type': 'insertion' if indel_offset > 0 else ('deletion' if indel_offset < 0 else 'SNV'),
        'variant_pos0_in_window': seq_len // 2,
        'position_offset_downstream': indel_offset,
    }
    
    # Use new function with legacy parameter mapping
    return align_predictions_by_coordinate(
        ref_preds, alt_preds, metadata_row,
        bin_length=bin_size, crop_length=edge_trim
    )



