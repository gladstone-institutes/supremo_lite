"""
Utilities for aligning model predictions between reference and variant sequences.

This module provides functions to handle the alignment of ML model predictions
when reference and variant sequences have position offsets due to indels. 

TODO: The user must specify what type of predictions are being input to these functions

The inputs will be 2 tensors for each window of dims: batch : bin : data
Where data is either a 1D set of prediction scores or a 1D representation of an upper 
triangular matrix (predicted contact map). For the contact map case we need to treat
the alignment differently so the user has to specify which of these it is

"""

import numpy as np
import warnings
from typing import Tuple, List, Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def coord_to_bin_offset(coord, window_start, bin_length:int, crop_length:int):
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


def bin_offset_to_coord(bin_offset, window_start, bin_length:int, crop_length:int):
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


def vector_to_contact_matrix(vector: Union[np.ndarray, 'torch.Tensor'], matrix_size: int) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Convert flattened upper triangular vector to full contact matrix.
    
    This function reconstructs a full symmetric contact matrix from its upper 
    triangular representation, following the pattern used in genomic contact map models.
    
    Args:
        vector: Flattened upper triangular matrix (length = matrix_size * (matrix_size + 1) / 2)
        matrix_size: Dimension of the output square matrix
        
    Returns:
        Full symmetric contact matrix of shape (matrix_size, matrix_size)
        
    Example:
        # For a 3x3 matrix, vector contains [M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]]
        vector = np.array([1, 2, 3, 4, 5, 6])
        matrix = vector_to_contact_matrix(vector, 3)
        # Result: [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    """
    # Handle both PyTorch tensors and NumPy arrays
    is_torch = TORCH_AVAILABLE and hasattr(vector, 'device')
    
    if is_torch:
        matrix = torch.zeros((matrix_size, matrix_size), dtype=vector.dtype, device=vector.device)
        triu_indices = torch.triu_indices(matrix_size, matrix_size)
        matrix[triu_indices[0], triu_indices[1]] = vector
        # Make symmetric by copying upper triangle to lower
        matrix = matrix + matrix.T - torch.diag(torch.diag(matrix))
    else:
        matrix = np.zeros((matrix_size, matrix_size), dtype=vector.dtype)
        triu_indices = np.triu_indices(matrix_size)
        matrix[triu_indices] = vector
        # Make symmetric by copying upper triangle to lower
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    
    return matrix


def contact_matrix_to_vector(matrix: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Convert full contact matrix to flattened upper triangular vector.
    
    This function extracts the upper triangular portion of a contact matrix,
    which is the standard representation for genomic contact maps.
    
    Args:
        matrix: Full symmetric contact matrix of shape (N, N)
        
    Returns:
        Flattened upper triangular vector of length N * (N + 1) / 2
        
    Example:
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        vector = contact_matrix_to_vector(matrix)
        # Result: [1, 2, 3, 4, 5, 6]
    """
    # Handle both PyTorch tensors and NumPy arrays
    is_torch = TORCH_AVAILABLE and hasattr(matrix, 'device')
    
    if is_torch:
        triu_indices = torch.triu_indices(matrix.shape[0], matrix.shape[1])
        return matrix[triu_indices[0], triu_indices[1]]
    else:
        triu_indices = np.triu_indices(matrix.shape[0])
        return matrix[triu_indices]


def coord_to_matrix_bin(coord: int, window_start: int, bin_length: int, crop_length: int) -> int:
    """
    Convert genomic coordinate to contact matrix bin index.
    
    This function maps a genomic coordinate to the corresponding bin index
    in a contact matrix, accounting for binning and edge cropping.
    
    Args:
        coord: Genomic coordinate within the window (0-based)
        window_start: Start coordinate of the sequence window (0-based)
        bin_length: Number of base pairs per bin
        crop_length: Number of bins cropped from each edge
        
    Returns:
        Matrix bin index (0-based within the contact matrix)
    """
    return coord_to_bin_offset(coord, window_start, bin_length, crop_length)


def mask_contact_matrix_for_variant(matrix: Union[np.ndarray, 'torch.Tensor'], 
                                   variant_bin: int, 
                                   variant_type: str,
                                   mask_size: int = 1) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Apply variant-specific masking to contact matrix.
    
    This function inserts NaN values in the contact matrix to represent
    structural changes caused by genomic variants.
    
    Args:
        matrix: Contact matrix to mask
        variant_bin: Bin index where variant occurs
        variant_type: Type of variant ('INS', 'DEL', etc.)
        mask_size: Number of bins to mask (default: 1)
        
    Returns:
        Masked contact matrix with NaN values at affected positions
    """
    # Handle both PyTorch tensors and NumPy arrays
    is_torch = TORCH_AVAILABLE and hasattr(matrix, 'device')
    
    masked_matrix = matrix.copy() if not is_torch else matrix.clone()
    
    if variant_type in ['INS', 'DEL']:
        # Mask both row and column for the affected bin(s)
        end_bin = variant_bin + mask_size
        if is_torch:
            masked_matrix[variant_bin:end_bin, :] = float('nan')
            masked_matrix[:, variant_bin:end_bin] = float('nan')
        else:
            masked_matrix[variant_bin:end_bin, :] = np.nan
            masked_matrix[:, variant_bin:end_bin] = np.nan
    
    return masked_matrix


def align_predictions_by_coordinate(ref_preds: Union[np.ndarray, 'torch.Tensor'], 
                                   alt_preds: Union[np.ndarray, 'torch.Tensor'],
                                   metadata_row,
                                   bin_length: int,
                                   crop_length: int,
                                   prediction_type: str,
                                   matrix_size: Optional[int] = None) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align reference and alt predictions using coordinate transformation and variant type awareness.
    
    This enhanced function uses precise coordinate mapping to align predictions,
    accounting for different variant types and their effects on sequence coordinates.
    Supports both 1D prediction scores and 2D contact map predictions.
    
    Args:
        ref_preds: Reference predictions array (from model with edge cropping)
        alt_preds: Alt predictions array (same length as ref_preds)
        metadata_row: Row from metadata DataFrame with variant information
        bin_length: Number of base pairs per prediction bin (default: 32)
        crop_length: Number of bins cropped from each edge during prediction (default: 16)
        prediction_type: Type of predictions ("scores" for 1D, "contact_map" for 2D)
        matrix_size: Size of contact matrix (required for contact_map type)
        
    Returns:
        Tuple of (aligned_ref_preds, aligned_alt_preds) aligned at bin level
        
    Note:
        This function now works at the prediction bin level rather than expanding
        to base-pair resolution, providing more accurate alignment for model predictions.
    """
    # Validate prediction type and parameters
    if prediction_type not in ["scores", "contact_map"]:
        raise ValueError(f"prediction_type must be 'scores' or 'contact_map', got '{prediction_type}'")
    
    if prediction_type == "contact_map" and matrix_size is None:
        raise ValueError("matrix_size must be provided for contact_map prediction type")
    
    # Route to appropriate alignment function based on prediction type
    if prediction_type == "scores":
        return align_1d_predictions(ref_preds, alt_preds, metadata_row, bin_length, crop_length)
    else:  # contact_map
        return align_contact_map_predictions(ref_preds, alt_preds, metadata_row, bin_length, crop_length, matrix_size)


def align_1d_predictions(ref_preds: Union[np.ndarray, 'torch.Tensor'], 
                        alt_preds: Union[np.ndarray, 'torch.Tensor'],
                        metadata_row,
                        bin_length: int,
                        crop_length: int) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align 1D prediction scores between reference and alt sequences.
    
    This function handles the original prediction alignment logic for 1D prediction arrays.
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
    if variant_type in ['SNV', 'MNV']:
        # SNVs don't change coordinates, direct alignment
        is_torch = TORCH_AVAILABLE and hasattr(ref_preds, 'device')
        if is_torch:
            return ref_preds.clone(), alt_preds.clone()
        else:
            return ref_preds.copy(), alt_preds.copy()
        
    elif variant_type in ['INS', 'DEL']:
        # Handle indel-type variants with coordinate shifts
        return align_indel_variants(ref_preds, alt_preds, variant_bin, downstream_offset, bin_length)
        
    elif variant_type == 'INV':
        # Inversions: sequence is reversed but coordinates maintained
        # TODO: Implement
        # Unknown variant type, raise error to prevent unsupported usage
        raise ValueError(f"Unsupported variant type: '{variant_type}'. "
                        f"Supported types are: SNV, INS, DEL")

        
    elif variant_type == 'DUP':
        # Duplications: sequence is duplicated, extending downstream
        # TODO: Implement
        # Unknown variant type, raise error to prevent unsupported usage
        raise ValueError(f"Unsupported variant type: '{variant_type}'. "
                        f"Supported types are: SNV, INS, DEL")

        
    elif variant_type == 'BND':
        # Breakends: complex rearrangements, may need special handling
        # TODO: Implement
        # Unknown variant type, raise error to prevent unsupported usage
        raise ValueError(f"Unsupported variant type: '{variant_type}'. "
                        f"Supported types are: SNV, INS, DEL")

        
    else:
        # Unknown variant type, raise error to prevent unsupported usage
        raise ValueError(f"Unsupported variant type: '{variant_type}'. "
                        f"Supported types are: SNV, INS, DEL")


def align_contact_map_predictions(ref_preds: Union[np.ndarray, 'torch.Tensor'], 
                                 alt_preds: Union[np.ndarray, 'torch.Tensor'],
                                 metadata_row,
                                 bin_length: int,
                                 crop_length: int,
                                 matrix_size: int) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align contact map predictions between reference and alt sequences.
    
    This function handles alignment of 2D contact map predictions, which are typically
    represented as flattened upper triangular matrices.
    
    Args:
        ref_preds: Reference contact map predictions (flattened upper triangular)
        alt_preds: Alt contact map predictions (same format as ref_preds)
        metadata_row: Row from metadata DataFrame with variant information
        bin_length: Number of base pairs per bin
        crop_length: Number of bins cropped from each edge
        matrix_size: Size of the contact matrix (e.g., 448 for Akita)
        
    Returns:
        Tuple of (aligned_ref_preds, aligned_alt_preds) for contact map predictions
    """
    # Extract variant information from metadata
    variant_type = metadata_row.get('variant_type', 'unknown')
    variant_pos_in_window = metadata_row.get('variant_pos0_in_window', metadata_row.get('effective_variant_start', 0))
    
    # Convert variant position to matrix bin coordinates
    variant_bin = coord_to_matrix_bin(variant_pos_in_window, 
                                     window_start=0,
                                     bin_length=bin_length, 
                                     crop_length=crop_length)
    
    # Apply alignment strategy based on variant type
    if variant_type in ['SNV', 'MNV']:
        return align_contact_map_snv(ref_preds, alt_preds, variant_bin, matrix_size)
        
    elif variant_type == 'INS':
        return align_contact_map_insertion(ref_preds, alt_preds, variant_bin, matrix_size, metadata_row)
        
    elif variant_type == 'DEL':
        return align_contact_map_deletion(ref_preds, alt_preds, variant_bin, matrix_size, metadata_row)
        
    else:
        # Unknown variant type, raise error to prevent unsupported usage
        raise ValueError(f"Unsupported variant type for contact maps: '{variant_type}'. "
                        f"Supported types are: SNV, MNV, INS, DEL")


def align_contact_map_snv(ref_preds: Union[np.ndarray, 'torch.Tensor'], 
                         alt_preds: Union[np.ndarray, 'torch.Tensor'],
                         variant_bin: int,
                         matrix_size: int) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align contact map predictions for SNV/MNV variants.
    
    For SNVs and MNVs, there are no coordinate changes, so the contact maps
    can be directly aligned without structural modifications.
    
    Args:
        ref_preds: Reference contact map predictions (flattened)
        alt_preds: Alt contact map predictions (flattened)
        variant_bin: Bin index where variant occurs (for future use)
        matrix_size: Size of the contact matrix
        
    Returns:
        Directly aligned contact map predictions
    """
    # SNVs don't change genomic coordinates, so direct alignment is appropriate
    is_torch = TORCH_AVAILABLE and hasattr(ref_preds, 'device')
    
    if is_torch:
        return ref_preds.clone(), alt_preds.clone()
    else:
        return ref_preds.copy(), alt_preds.copy()


def align_contact_map_insertion(ref_preds: Union[np.ndarray, 'torch.Tensor'], 
                               alt_preds: Union[np.ndarray, 'torch.Tensor'],
                               variant_bin: int,
                               matrix_size: int,
                               metadata_row) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align contact map predictions for insertion variants.
    
    For insertions, the contact map matrix needs to be modified to account for
    the new interaction space created by the inserted sequence.
    
    Args:
        ref_preds: Reference contact map predictions (flattened)
        alt_preds: Alt contact map predictions (flattened)
        variant_bin: Bin index where insertion occurs
        matrix_size: Size of the contact matrix
        metadata_row: Metadata with insertion details
        
    Returns:
        Aligned contact map predictions with insertion effects
    """
    # Convert to matrices for easier manipulation
    ref_matrix = vector_to_contact_matrix(ref_preds, matrix_size)
    alt_matrix = vector_to_contact_matrix(alt_preds, matrix_size)
    
    # For insertions, mask the alt matrix at the insertion site
    # This represents the fact that new interactions are created
    masked_alt_matrix = mask_contact_matrix_for_variant(alt_matrix, variant_bin, 'INS')
    
    # Convert back to flattened format
    is_torch = TORCH_AVAILABLE and hasattr(ref_preds, 'device')
    
    if is_torch:
        aligned_ref = contact_matrix_to_vector(ref_matrix)
        aligned_alt = contact_matrix_to_vector(masked_alt_matrix)
    else:
        aligned_ref = contact_matrix_to_vector(ref_matrix)
        aligned_alt = contact_matrix_to_vector(masked_alt_matrix)
    
    return aligned_ref, aligned_alt


def align_contact_map_deletion(ref_preds: Union[np.ndarray, 'torch.Tensor'], 
                              alt_preds: Union[np.ndarray, 'torch.Tensor'],
                              variant_bin: int,
                              matrix_size: int,
                              metadata_row) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align contact map predictions for deletion variants.
    
    For deletions, the contact map matrix needs to be modified to account for
    the loss of interaction space due to the deleted sequence.
    
    Args:
        ref_preds: Reference contact map predictions (flattened)
        alt_preds: Alt contact map predictions (flattened)
        variant_bin: Bin index where deletion occurs
        matrix_size: Size of the contact matrix
        metadata_row: Metadata with deletion details
        
    Returns:
        Aligned contact map predictions with deletion effects
    """
    # Convert to matrices for easier manipulation
    ref_matrix = vector_to_contact_matrix(ref_preds, matrix_size)
    alt_matrix = vector_to_contact_matrix(alt_preds, matrix_size)
    
    # For deletions, mask the reference matrix at the deletion site
    # This represents the loss of interactions in the deleted region
    masked_ref_matrix = mask_contact_matrix_for_variant(ref_matrix, variant_bin, 'DEL')
    
    # Convert back to flattened format
    aligned_ref = contact_matrix_to_vector(masked_ref_matrix)
    aligned_alt = contact_matrix_to_vector(alt_matrix)
    
    return aligned_ref, aligned_alt


def align_indel_variants(ref_preds: Union[np.ndarray, 'torch.Tensor'], alt_preds: Union[np.ndarray, 'torch.Tensor'], 
                        variant_bin: int, downstream_offset: int,
                        bin_length: int) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
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
    # Handle PyTorch tensors and NumPy arrays
    is_torch = TORCH_AVAILABLE and hasattr(ref_preds, 'device')
    
    if downstream_offset == 0:
        # No coordinate shift needed
        if is_torch:
            return ref_preds.clone(), alt_preds.clone()
        else:
            return ref_preds.copy(), alt_preds.copy()
    
    # Create aligned alt predictions array
    if is_torch:
        aligned_alt_preds = torch.zeros_like(alt_preds)
    else:
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
    
    if is_torch:
        return ref_preds.clone(), aligned_alt_preds
    else:
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
    # TODO: Implement BND alignment
    return ref_preds.copy(), alt_preds.copy()

