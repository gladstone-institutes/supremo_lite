"""
Utilities for aligning model predictions between reference and variant sequences.

This module provides functions to handle the alignment of ML model predictions
when reference and variant sequences have position offsets due to indels.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


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


def classify_variant_type(ref_allele, alt_allele, variant_info=None):
    """
    Classify variant type including structural variants.
    
    Args:
        ref_allele: Reference allele sequence
        alt_allele: Alternate allele sequence
        variant_info: Additional VCF INFO field data for structural variants (dict)
        
    Returns:
        str: Variant type - one of 'SNV', 'insertion', 'deletion', 'complex', 'INV', 'DUP', 'BND'
        
    Examples:
        classify_variant_type('A', 'G') # Returns 'SNV'
        classify_variant_type('T', 'TGGG') # Returns 'insertion'  
        classify_variant_type('CGAGAA', 'C') # Returns 'deletion'
        classify_variant_type('ATATT', 'AGATT') # Returns 'complex'
        classify_variant_type('.', '<INV>') # Returns 'INV'
    """
    # Handle structural variants with symbolic alleles
    if alt_allele.startswith('<') and alt_allele.endswith('>'):
        sv_type = alt_allele[1:-1]  # Extract type from <INV>, <DUP>, etc.
        if sv_type == 'INV':
            return 'INV'
        elif sv_type == 'DUP':
            return 'DUP'
        elif sv_type in ['BND', 'TRA']:  # Breakend/translocation
            return 'BND'
        else:
            return 'complex'
    
    # Handle breakend notation (e.g., 'A[chr2:1000[')
    if '[' in alt_allele or ']' in alt_allele:
        return 'BND'
    
    # Handle standard sequence variants based on allele lengths
    ref_len = len(ref_allele)
    alt_len = len(alt_allele)
    
    if ref_len == 1 and alt_len == 1:
        return 'SNV'
    elif ref_len == 1 and alt_len > 1:
        return 'insertion'
    elif ref_len > 1 and alt_len == 1:
        return 'deletion'
    else:
        return 'complex'


def parse_vcf_info(info_string):
    """
    Parse VCF INFO field to extract structural variant information.
    
    Args:
        info_string: VCF INFO field string (e.g., "SVTYPE=INV;END=1234;SVLEN=100")
        
    Returns:
        dict: Parsed INFO field values
        
    Common SV INFO fields:
        - END: End position of variant
        - SVTYPE: Type of structural variant
        - SVLEN: Length of structural variant  
        - CHR2: Chromosome for BND mate
        - POS2: Position for BND mate
    """
    info_dict = {}
    if not info_string or info_string == '.':
        return info_dict
        
    for field in info_string.split(';'):
        if '=' in field:
            key, value = field.split('=', 1)
            # Try to convert numeric values
            try:
                if '.' in value:
                    info_dict[key] = float(value)
                else:
                    info_dict[key] = int(value)
            except ValueError:
                info_dict[key] = value
        else:
            # Boolean flag (present = True)
            info_dict[field] = True
            
    return info_dict


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


def calculate_variant_effect_scores(aligned_ref_preds: np.ndarray,
                                  aligned_alt_preds: np.ndarray,
                                  method: str = 'log_ratio') -> np.ndarray:
    """
    Calculate variant effect scores from aligned predictions.
    
    Args:
        aligned_ref_preds: Aligned reference predictions
        aligned_alt_preds: Aligned alt predictions
        method: Method for calculating effect ('log_ratio', 'difference', 'fold_change')
    
    Returns:
        Array of effect scores for each aligned bin
    """
    if len(aligned_ref_preds) != len(aligned_alt_preds):
        raise ValueError("Ref and alt predictions must have same length")
    
    if method == 'log_ratio':
        # log2(alt/ref), handle zeros by adding small pseudocount
        pseudocount = 1e-8
        effect = np.log2((aligned_alt_preds + pseudocount) / (aligned_ref_preds + pseudocount))
    elif method == 'difference':
        effect = aligned_alt_preds - aligned_ref_preds
    elif method == 'fold_change':
        # alt/ref ratio
        effect = aligned_alt_preds / (aligned_ref_preds + 1e-8)  # Avoid division by zero
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return effect


def process_variant_predictions(ref_preds: np.ndarray,
                               alt_preds: np.ndarray,
                               metadata_df_row,
                               bin_length: int = 32,
                               crop_length: int = 16,
                               effect_method: str = 'log_ratio') -> Dict[str, np.ndarray]:
    """
    Complete workflow for processing variant predictions with coordinate alignment.
    
    This enhanced function works with the new DataFrame metadata format and 
    comprehensive variant type support.
    
    Args:
        ref_preds: Reference predictions array
        alt_preds: Alt predictions array
        metadata_df_row: Row from metadata DataFrame (dict-like) with variant information
        bin_length: Base pairs per prediction bin (default: 32)
        crop_length: Number of bins cropped from each edge during prediction (default: 16)
        effect_method: Method for calculating variant effects ('log_ratio', 'difference', 'fold_change')
    
    Returns:
        Dictionary containing:
        - 'ref_predictions': Original reference predictions
        - 'alt_predictions': Original alt predictions  
        - 'aligned_ref': Aligned reference predictions
        - 'aligned_alt': Aligned alt predictions
        - 'effect_scores': Variant effect scores
        - 'variant_info': Enhanced variant information
    """
    # Align predictions using coordinate-based approach with variant type awareness
    aligned_ref, aligned_alt = align_predictions_by_coordinate(
        ref_preds, alt_preds, metadata_df_row, bin_length, crop_length
    )
    
    # Calculate effect scores
    effect_scores = calculate_variant_effect_scores(aligned_ref, aligned_alt, effect_method)
    
    return {
        'ref_predictions': ref_preds,
        'alt_predictions': alt_preds,
        'aligned_ref': aligned_ref,
        'aligned_alt': aligned_alt,
        'effect_scores': effect_scores,
        'variant_info': {
            'variant_type': metadata_df_row.get('variant_type'),
            'ref_length': metadata_df_row.get('ref_length'),
            'alt_length': metadata_df_row.get('alt_length'),
            'position_offset_downstream': metadata_df_row.get('position_offset_downstream'),
            'structural_variant_info': metadata_df_row.get('structural_variant_info'),
        }
    }


# Legacy function for backward compatibility
def process_variant_predictions_legacy(ref_preds: np.ndarray,
                                      alt_preds: np.ndarray,
                                      metadata: Dict,
                                      seq_len: int,
                                      bin_size: int = 10,
                                      edge_trim: int = 5,
                                      effect_method: str = 'log_ratio') -> Dict[str, np.ndarray]:
    """
    Legacy processing function for backward compatibility.
    
    DEPRECATED: Use process_variant_predictions with metadata_df_row instead.
    """
    import warnings
    warnings.warn(
        "process_variant_predictions with metadata dict is deprecated. "
        "Use the new version with metadata_df_row parameter for enhanced variant type support.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert legacy metadata to new format
    metadata_df_row = {
        'variant_type': 'insertion' if metadata.get('indel_offset', 0) > 0 else ('deletion' if metadata.get('indel_offset', 0) < 0 else 'SNV'),
        'variant_pos0_in_window': seq_len // 2,
        'position_offset_downstream': metadata.get('indel_offset', 0),
        'ref_length': len(metadata.get('ref', '')),
        'alt_length': len(metadata.get('alt', '')),
        'structural_variant_info': None,
    }
    
    return process_variant_predictions(
        ref_preds, alt_preds, metadata_df_row,
        bin_length=bin_size, crop_length=edge_trim,
        effect_method=effect_method
    )