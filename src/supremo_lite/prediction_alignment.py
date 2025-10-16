"""
Utilities for aligning model predictions between reference and variant sequences.

This module provides functions to handle the alignment of ML model predictions
when reference and variant sequences have position offsets due to structural variants.

The alignment logic properly handles:
- 1D predictions (chromatin accessibility, TF binding, etc.)
- 2D contact maps (Hi-C, Micro-C predictions)
- All variant types: SNV, INS, DEL, DUP, INV, BND

Key principle: Users must specify all model-specific parameters (bin_size, diag_offset)
as these vary across different prediction models.
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class VariantPosition:
    """
    Container for variant position information in both REF and ALT sequences.

    This class encapsulates the essential positional information needed to align
    predictions across reference and alternate sequences that may differ in length.

    Attributes:
        ref_pos: Position in reference sequence (base pairs, 0-based)
        alt_pos: Position in alternate sequence (base pairs, 0-based)
        svlen: Length of structural variant (base pairs, signed for DEL/INS)
        variant_type: Type of variant ('SNV', 'INS', 'DEL', 'DUP', 'INV', 'BND')
    """
    ref_pos: int
    alt_pos: int
    svlen: int
    variant_type: str

    def get_bin_positions(self, bin_size: int) -> Tuple[int, int, int]:
        """
        Convert base pair positions to bin indices.

        Args:
            bin_size: Number of base pairs per prediction bin

        Returns:
            Tuple of (ref_bin, alt_start_bin, alt_end_bin)
        """
        ref_bin = int(np.ceil(self.ref_pos / bin_size))
        alt_start_bin = int(np.ceil(self.alt_pos / bin_size))
        alt_end_bin = int(np.ceil((self.alt_pos + abs(self.svlen)) / bin_size))
        return ref_bin, alt_start_bin, alt_end_bin


class PredictionAligner1D:
    """
    Aligns reference and alternate 1D prediction vectors for variant comparison.

    Handles alignment of 1D genomic predictions (e.g., chromatin accessibility,
    transcription factor binding, epigenetic marks) between reference and variant
    sequences that may differ in length due to structural variants.

    The aligner uses a masking strategy where positions that exist in one sequence
    but not the other are marked with NaN values, enabling direct comparison of
    corresponding genomic positions.

    Args:
        target_size: Expected number of bins in the prediction output
        bin_size: Number of base pairs per prediction bin (model-specific)

    Example:
        >>> aligner = PredictionAligner1D(target_size=896, bin_size=128)
        >>> ref_aligned, alt_aligned = aligner.align_predictions(
        ...     ref_pred, alt_pred, 'INS', variant_position
        ... )
    """

    def __init__(self, target_size: int, bin_size: int):
        """
        Initialize the 1D prediction aligner.

        Args:
            target_size: Expected number of bins in prediction (e.g., 896 for Enformer)
            bin_size: Base pairs per bin (e.g., 128 for Enformer)
        """
        self.target_size = target_size
        self.bin_size = bin_size

    def align_predictions(
        self,
        ref_pred: Union[np.ndarray, 'torch.Tensor'],
        alt_pred: Union[np.ndarray, 'torch.Tensor'],
        svtype: str,
        var_pos: VariantPosition
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Main entry point for 1D prediction alignment.

        Args:
            ref_pred: Reference prediction vector (length N)
            alt_pred: Alternate prediction vector (length N)
            svtype: Variant type ('DEL', 'DUP', 'INS', 'INV', 'SNV')
            var_pos: Variant position information

        Returns:
            Tuple of (aligned_ref, aligned_alt) vectors with NaN masking applied

        Raises:
            ValueError: For unsupported variant types or if using BND (use align_bnd_predictions)
        """
        if svtype == 'BND' or svtype == 'SV_BND':
            raise ValueError("Use align_bnd_predictions() for breakends")

        # Normalize variant type names
        svtype_normalized = svtype.replace('SV_', '')

        if svtype_normalized in ['DEL', 'DUP', 'INS']:
            return self._align_indel_predictions(ref_pred, alt_pred, svtype_normalized, var_pos)
        elif svtype_normalized == 'INV':
            return self._align_inversion_predictions(ref_pred, alt_pred, var_pos)
        elif svtype_normalized in ['SNV', 'MNV']:
            # SNVs don't change coordinates, direct alignment
            is_torch = TORCH_AVAILABLE and torch.is_tensor(ref_pred)
            if is_torch:
                return ref_pred.clone(), alt_pred.clone()
            else:
                return ref_pred.copy(), alt_pred.copy()
        else:
            raise ValueError(f"Unknown variant type: {svtype}")

    def _align_indel_predictions(
        self,
        ref_pred: Union[np.ndarray, 'torch.Tensor'],
        alt_pred: Union[np.ndarray, 'torch.Tensor'],
        svtype: str,
        var_pos: VariantPosition
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Align predictions for insertions, deletions, and duplications.

        Strategy:
        1. For DEL: Swap REF/ALT (deletion removes from REF)
        2. Insert NaN bins in shorter sequence
        3. Crop edges to maintain target size
        4. For DEL: Swap back

        This ensures that positions present in one sequence but not the other
        are marked with NaN, enabling fair comparison of overlapping regions.
        """
        is_torch = TORCH_AVAILABLE and torch.is_tensor(ref_pred)

        # Convert to numpy for manipulation
        if is_torch:
            ref_np = ref_pred.detach().cpu().numpy()
            alt_np = alt_pred.detach().cpu().numpy()
        else:
            ref_np = ref_pred
            alt_np = alt_pred

        # Swap for deletions (treat as insertion in reverse)
        if svtype == 'DEL':
            ref_np, alt_np = alt_np, ref_np
            var_pos = VariantPosition(var_pos.alt_pos, var_pos.ref_pos, var_pos.svlen, svtype)

        # Get bin positions
        ref_bin, alt_start_bin, alt_end_bin = var_pos.get_bin_positions(self.bin_size)
        bins_to_add = alt_end_bin - alt_start_bin

        # Insert NaN bins in REF where variant exists in ALT
        ref_masked = self._insert_nan_bins(ref_np, ref_bin, bins_to_add)

        # Crop to maintain target size
        ref_masked = self._crop_vector(ref_masked, ref_bin, alt_start_bin)
        alt_masked = alt_np.copy()

        # Swap back for deletions
        if svtype == 'DEL':
            ref_masked, alt_masked = alt_masked, ref_masked

        self._validate_size(ref_masked, alt_masked)

        # Convert back to torch if needed
        if is_torch:
            ref_masked = torch.from_numpy(ref_masked).to(ref_pred.device).type(ref_pred.dtype)
            alt_masked = torch.from_numpy(alt_masked).to(alt_pred.device).type(alt_pred.dtype)

        return ref_masked, alt_masked

    def _insert_nan_bins(
        self,
        vector: np.ndarray,
        position: int,
        num_bins: int
    ) -> np.ndarray:
        """
        Insert NaN values at specified position in vector.

        Args:
            vector: Input prediction vector
            position: Position to insert NaN values (bin index)
            num_bins: Number of NaN values to insert

        Returns:
            Vector with NaN values inserted
        """
        result = vector.copy()
        for offset in range(num_bins):
            insert_pos = position + offset
            result = np.insert(result, insert_pos, np.nan)
        return result

    def _crop_vector(
        self,
        vector: np.ndarray,
        ref_bin: int,
        alt_bin: int
    ) -> np.ndarray:
        """
        Crop vector edges to maintain target size.

        After inserting NaN bins, the vector is longer than expected.
        This function crops from edges proportionally to center the variant.

        Args:
            vector: Vector to crop
            ref_bin: Reference bin position
            alt_bin: Alternate bin position

        Returns:
            Cropped vector of target_size length
        """
        remove_left = ref_bin - alt_bin
        remove_right = len(vector) - self.target_size - remove_left

        # Apply cropping
        start = max(0, remove_left)
        end = len(vector) - max(0, remove_right)
        return vector[start:end]

    def _align_inversion_predictions(
        self,
        ref_pred: Union[np.ndarray, 'torch.Tensor'],
        alt_pred: Union[np.ndarray, 'torch.Tensor'],
        var_pos: VariantPosition
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Align predictions for inversions.

        Strategy:
        1. Mask the inverted region in both vectors with NaN
        2. This allows comparison of only the flanking (unaffected) regions

        For strand-aware models, inversions can significantly affect predictions
        because regulatory elements now appear on the opposite strand. We mask
        the inverted region to focus comparison on unaffected flanking sequences.
        """
        is_torch = TORCH_AVAILABLE and torch.is_tensor(ref_pred)

        # Convert to numpy for manipulation
        if is_torch:
            ref_np = ref_pred.detach().cpu().numpy()
            alt_np = alt_pred.detach().cpu().numpy()
        else:
            ref_np = ref_pred.copy()
            alt_np = alt_pred.copy()

        var_start, _, var_end = var_pos.get_bin_positions(self.bin_size)

        # Mask inverted region in both REF and ALT
        ref_np[var_start:var_end + 1] = np.nan
        alt_np[var_start:var_end + 1] = np.nan

        self._validate_size(ref_np, alt_np)

        # Convert back to torch if needed
        if is_torch:
            ref_np = torch.from_numpy(ref_np).to(ref_pred.device).type(ref_pred.dtype)
            alt_np = torch.from_numpy(alt_np).to(alt_pred.device).type(alt_pred.dtype)

        return ref_np, alt_np

    def align_bnd_predictions(
        self,
        left_ref: Union[np.ndarray, 'torch.Tensor'],
        right_ref: Union[np.ndarray, 'torch.Tensor'],
        bnd_alt: Union[np.ndarray, 'torch.Tensor'],
        breakpoint_bin: int
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Align predictions for breakends (chromosomal rearrangements).

        BNDs join two distant loci, so we create a chimeric reference
        prediction from the two separate loci for comparison with the fusion ALT.

        Args:
            left_ref: Prediction from left locus
            right_ref: Prediction from right locus
            bnd_alt: Prediction from joined (alternate) sequence
            breakpoint_bin: Bin position of breakpoint

        Returns:
            Tuple of (chimeric_ref, alt) vectors
        """
        is_torch = TORCH_AVAILABLE and torch.is_tensor(left_ref)

        # Convert to numpy for manipulation
        if is_torch:
            left_np = left_ref.detach().cpu().numpy()
            right_np = right_ref.detach().cpu().numpy()
            alt_np = bnd_alt.detach().cpu().numpy()
        else:
            left_np = left_ref
            right_np = right_ref
            alt_np = bnd_alt

        # Extract the relevant portions from each reference
        left_portion = left_np[:breakpoint_bin]
        right_portion = right_np[-(self.target_size - breakpoint_bin):]

        # Assemble chimeric reference
        ref_chimeric = np.concatenate([left_portion, right_portion])

        # Insert NaN at breakpoint to mark the transition
        ref_chimeric[breakpoint_bin] = np.nan

        self._validate_size(ref_chimeric, alt_np)

        # Convert back to torch if needed
        if is_torch:
            ref_chimeric = torch.from_numpy(ref_chimeric).to(left_ref.device).type(left_ref.dtype)
            alt_np = torch.from_numpy(alt_np).to(bnd_alt.device).type(bnd_alt.dtype)

        return ref_chimeric, alt_np

    def _validate_size(self, ref_vector: np.ndarray, alt_vector: np.ndarray):
        """
        Validate that vectors are the correct size.

        Args:
            ref_vector: Reference prediction vector
            alt_vector: Alternate prediction vector

        Raises:
            ValueError: If either vector has incorrect size
        """
        if len(ref_vector) != self.target_size:
            raise ValueError(
                f"Reference vector wrong size: {len(ref_vector)} vs {self.target_size}"
            )
        if len(alt_vector) != self.target_size:
            raise ValueError(
                f"Alternate vector wrong size: {len(alt_vector)} vs {self.target_size}"
            )


class PredictionAligner2D:
    """
    Aligns reference and alternate prediction matrices for variant comparison.

    Handles alignment of 2D genomic predictions (e.g., Hi-C contact maps,
    Micro-C predictions) between reference and variant sequences that may
    differ in length due to structural variants.

    The aligner uses a masking strategy where matrix rows and columns that
    exist in one sequence but not the other are marked with NaN values.

    Args:
        target_size: Expected matrix dimension (NxN)
        bin_size: Number of base pairs per matrix bin (model-specific)
        diag_offset: Number of diagonal bins to mask (model-specific)

    Example:
        >>> aligner = PredictionAligner2D(
        ...     target_size=448,
        ...     bin_size=2048,
        ...     diag_offset=2
        ... )
        >>> ref_aligned, alt_aligned = aligner.align_predictions(
        ...     ref_matrix, alt_matrix, 'DEL', variant_position
        ... )
    """

    def __init__(self, target_size: int, bin_size: int, diag_offset: int):
        """
        Initialize the 2D prediction aligner.

        Args:
            target_size: Matrix dimension (e.g., 448 for Akita)
            bin_size: Base pairs per bin (e.g., 2048 for Akita)
            diag_offset: Diagonal masking offset (e.g., 2 for Akita)
        """
        self.target_size = target_size
        self.bin_size = bin_size
        self.diag_offset = diag_offset

    def align_predictions(
        self,
        ref_pred: Union[np.ndarray, 'torch.Tensor'],
        alt_pred: Union[np.ndarray, 'torch.Tensor'],
        svtype: str,
        var_pos: VariantPosition
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Main entry point for 2D matrix alignment.

        Args:
            ref_pred: Reference prediction matrix (NxN)
            alt_pred: Alternate prediction matrix (NxN)
            svtype: Variant type ('DEL', 'DUP', 'INS', 'INV', 'SNV')
            var_pos: Variant position information

        Returns:
            Tuple of (aligned_ref, aligned_alt) matrices with NaN masking applied

        Raises:
            ValueError: For unsupported variant types or if using BND (use align_bnd_matrices)
        """
        if svtype == 'BND' or svtype == 'SV_BND':
            raise ValueError("Use align_bnd_matrices() for breakends")

        # Normalize variant type names
        svtype_normalized = svtype.replace('SV_', '')

        if svtype_normalized in ['DEL', 'DUP', 'INS']:
            return self._align_indel_matrices(ref_pred, alt_pred, svtype_normalized, var_pos)
        elif svtype_normalized == 'INV':
            return self._align_inversion_matrices(ref_pred, alt_pred, var_pos)
        elif svtype_normalized in ['SNV', 'MNV']:
            # SNVs don't change coordinates, direct alignment
            is_torch = TORCH_AVAILABLE and torch.is_tensor(ref_pred)
            if is_torch:
                return ref_pred.clone(), alt_pred.clone()
            else:
                return ref_pred.copy(), alt_pred.copy()
        else:
            raise ValueError(f"Unknown variant type: {svtype}")

    def _align_indel_matrices(
        self,
        ref_pred: Union[np.ndarray, 'torch.Tensor'],
        alt_pred: Union[np.ndarray, 'torch.Tensor'],
        svtype: str,
        var_pos: VariantPosition
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Align matrices for insertions, deletions, and duplications.

        Strategy:
        1. For DEL: Swap REF/ALT (deletion removes from REF)
        2. Insert NaN bins (rows AND columns) in shorter matrix
        3. Crop edges to maintain target size
        4. For DEL: Swap back
        """
        is_torch = TORCH_AVAILABLE and torch.is_tensor(ref_pred)

        # Convert to numpy for manipulation
        if is_torch:
            ref_np = ref_pred.detach().cpu().numpy()
            alt_np = alt_pred.detach().cpu().numpy()
        else:
            ref_np = ref_pred
            alt_np = alt_pred

        # Swap for deletions (treat as insertion in reverse)
        if svtype == 'DEL':
            ref_np, alt_np = alt_np, ref_np
            var_pos = VariantPosition(var_pos.alt_pos, var_pos.ref_pos, var_pos.svlen, svtype)

        # Get bin positions
        ref_bin, alt_start_bin, alt_end_bin = var_pos.get_bin_positions(self.bin_size)
        bins_to_add = alt_end_bin - alt_start_bin

        # Insert NaN bins in REF where variant exists in ALT
        ref_masked = self._insert_nan_bins(ref_np, ref_bin, bins_to_add)

        # Crop to maintain target size
        ref_masked = self._crop_matrix(ref_masked, ref_bin, alt_start_bin)
        alt_masked = alt_np.copy()

        # Swap back for deletions
        if svtype == 'DEL':
            ref_masked, alt_masked = alt_masked, ref_masked

        self._validate_size(ref_masked, alt_masked)

        # Convert back to torch if needed
        if is_torch:
            ref_masked = torch.from_numpy(ref_masked).to(ref_pred.device).type(ref_pred.dtype)
            alt_masked = torch.from_numpy(alt_masked).to(alt_pred.device).type(alt_pred.dtype)

        return ref_masked, alt_masked

    def _insert_nan_bins(
        self,
        matrix: np.ndarray,
        position: int,
        num_bins: int
    ) -> np.ndarray:
        """
        Insert NaN bins (rows and columns) at specified position.

        For 2D matrices, we must insert both rows AND columns to maintain
        the square matrix structure and properly mask interactions.
        """
        result = matrix.copy()
        for offset in range(num_bins):
            insert_pos = position + offset
            result = np.insert(result, insert_pos, np.nan, axis=0)
            result = np.insert(result, insert_pos, np.nan, axis=1)
        return result

    def _crop_matrix(
        self,
        matrix: np.ndarray,
        ref_bin: int,
        alt_bin: int
    ) -> np.ndarray:
        """
        Crop matrix edges to maintain target size.

        After inserting NaN bins, the matrix is larger than expected.
        This function crops from edges proportionally to center the variant.
        """
        remove_left = ref_bin - alt_bin
        remove_right = len(matrix) - self.target_size - remove_left

        # Apply cropping
        start = max(0, remove_left)
        end = len(matrix) - max(0, remove_right)
        return matrix[start:end, start:end]

    def _align_inversion_matrices(
        self,
        ref_pred: Union[np.ndarray, 'torch.Tensor'],
        alt_pred: Union[np.ndarray, 'torch.Tensor'],
        var_pos: VariantPosition
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Align matrices for inversions.

        Strategy: Mask the inverted region in both REF and ALT matrices using
        a cross-pattern (mask entire rows AND columns at the inversion position).

        Why cross-masking?
        - Inversions reverse the sequence, creating geometric rotation in contact maps
        - Masking rows removes interactions with the inverted region (one dimension)
        - Masking columns removes interactions from the inverted region (other dimension)
        - This ensures only flanking regions (unaffected by inversion) are compared

        The same NaN pattern is mirrored to ALT so both matrices have identical
        masked regions, enabling fair comparison of the unaffected areas.
        """
        is_torch = TORCH_AVAILABLE and hasattr(ref_pred, 'device')

        # Convert to numpy for manipulation
        if is_torch:
            ref_np = ref_pred.detach().cpu().numpy()
            alt_np = alt_pred.detach().cpu().numpy()
        else:
            ref_np = ref_pred.copy()
            alt_np = alt_pred.copy()

        var_start, _, var_end = var_pos.get_bin_positions(self.bin_size)

        # Mask inverted region in REF (cross-pattern: rows + columns)
        ref_np[var_start:var_end + 1, :] = np.nan
        ref_np[:, var_start:var_end + 1] = np.nan

        # Mirror NaN pattern to ALT (correct approach)
        nan_mask = ref_np.copy()
        nan_mask[np.invert(np.isnan(nan_mask))] = 0  # Non-NaN → 0, NaN stays NaN
        alt_np = alt_np + nan_mask  # Adding NaN propagates NaN to ALT

        self._validate_size(ref_np, alt_np)

        # Convert back to torch if needed
        if is_torch:
            ref_np = torch.from_numpy(ref_np).to(ref_pred.device).type(ref_pred.dtype)
            alt_np = torch.from_numpy(alt_np).to(alt_pred.device).type(alt_pred.dtype)

        return ref_np, alt_np

    def align_bnd_matrices(
        self,
        left_ref: Union[np.ndarray, 'torch.Tensor'],
        right_ref: Union[np.ndarray, 'torch.Tensor'],
        bnd_alt: Union[np.ndarray, 'torch.Tensor'],
        breakpoint_bin: int
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
        """
        Align matrices for breakends (chromosomal rearrangements).

        BNDs join two distant loci, so we create a chimeric reference
        matrix from the two separate loci.

        Args:
            left_ref: Prediction from left locus
            right_ref: Prediction from right locus
            bnd_alt: Prediction from joined (alternate) sequence
            breakpoint_bin: Bin position of breakpoint

        Returns:
            Tuple of (chimeric_ref, alt) matrices
        """
        is_torch = TORCH_AVAILABLE and hasattr(left_ref, 'device')

        # Convert to numpy for manipulation
        if is_torch:
            left_np = left_ref.detach().cpu().numpy()
            right_np = right_ref.detach().cpu().numpy()
            alt_np = bnd_alt.detach().cpu().numpy()
        else:
            left_np = left_ref
            right_np = right_ref
            alt_np = bnd_alt

        # Assemble chimeric matrix from two loci
        ref_chimeric = self._assemble_chimeric_matrix(
            left_np,
            right_np,
            breakpoint_bin
        )

        self._validate_size(ref_chimeric, alt_np)

        # Convert back to torch if needed
        if is_torch:
            ref_chimeric = torch.from_numpy(ref_chimeric).to(left_ref.device).type(left_ref.dtype)
            alt_np = torch.from_numpy(alt_np).to(bnd_alt.device).type(bnd_alt.dtype)

        return ref_chimeric, alt_np

    def _assemble_chimeric_matrix(
        self,
        left_matrix: np.ndarray,
        right_matrix: np.ndarray,
        breakpoint: int
    ) -> np.ndarray:
        """
        Assemble chimeric matrix from two loci.

        Structure:
        - Upper left quadrant: left locus
        - Lower right quadrant: right locus
        - Upper right/lower left quadrants: NaN (no trans prediction)
        """
        matrix = np.zeros((self.target_size, self.target_size))

        # Fill upper left quadrant (left locus)
        matrix[:breakpoint, :breakpoint] = left_matrix[:breakpoint, :breakpoint]

        # Fill lower right quadrant (right locus)
        matrix[breakpoint:, breakpoint:] = right_matrix[breakpoint:, breakpoint:]

        # Fill transition quadrants with NaN
        matrix[:breakpoint, breakpoint:] = np.nan
        matrix[breakpoint:, :breakpoint] = np.nan

        # Mask diagonals as specified by model
        for offset in range(-self.diag_offset + 1, self.diag_offset):
            if offset < 0:
                np.fill_diagonal(matrix[abs(offset):, :], np.nan)
            else:
                np.fill_diagonal(matrix[:, offset:], np.nan)

        return matrix

    def _validate_size(self, ref_matrix: np.ndarray, alt_matrix: np.ndarray):
        """
        Validate that matrices are the correct size.

        Args:
            ref_matrix: Reference prediction matrix
            alt_matrix: Alternate prediction matrix

        Raises:
            ValueError: If either matrix has incorrect size
        """
        if ref_matrix.shape[0] != self.target_size:
            raise ValueError(
                f"Reference matrix wrong size: {ref_matrix.shape[0]} vs {self.target_size}"
            )
        if alt_matrix.shape[0] != self.target_size:
            raise ValueError(
                f"Alternate matrix wrong size: {alt_matrix.shape[0]} vs {self.target_size}"
            )


# =============================================================================
# Utility Functions for Contact Maps
# =============================================================================

def vector_to_contact_matrix(
    vector: Union[np.ndarray, 'torch.Tensor'],
    matrix_size: int
) -> Union[np.ndarray, 'torch.Tensor']:
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
        >>> # For a 3x3 matrix, vector contains [M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]]
        >>> vector = np.array([1, 2, 3, 4, 5, 6])
        >>> matrix = vector_to_contact_matrix(vector, 3)
        >>> # Result: [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    """
    # Handle both PyTorch tensors and NumPy arrays
    is_torch = TORCH_AVAILABLE and torch.is_tensor(vector)

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


def contact_matrix_to_vector(
    matrix: Union[np.ndarray, 'torch.Tensor']
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Convert full contact matrix to flattened upper triangular vector.

    This function extracts the upper triangular portion of a contact matrix,
    which is the standard representation for genomic contact maps.

    Args:
        matrix: Full symmetric contact matrix of shape (N, N)

    Returns:
        Flattened upper triangular vector of length N * (N + 1) / 2

    Example:
        >>> matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        >>> vector = contact_matrix_to_vector(matrix)
        >>> # Result: [1, 2, 3, 4, 5, 6]
    """
    # Handle both PyTorch tensors and NumPy arrays
    is_torch = TORCH_AVAILABLE and torch.is_tensor(matrix)

    if is_torch:
        triu_indices = torch.triu_indices(matrix.shape[0], matrix.shape[1])
        return matrix[triu_indices[0], triu_indices[1]]
    else:
        triu_indices = np.triu_indices(matrix.shape[0])
        return matrix[triu_indices]



def align_predictions_by_coordinate(
    ref_preds: Union[np.ndarray, 'torch.Tensor'],
    alt_preds: Union[np.ndarray, 'torch.Tensor'],
    metadata_row: dict,
    bin_size: int,
    prediction_type: str,
    matrix_size: Optional[int] = None,
    diag_offset: int = 0
) -> Tuple[Union[np.ndarray, 'torch.Tensor'], Union[np.ndarray, 'torch.Tensor']]:
    """
    Align reference and alt predictions using coordinate transformation and variant type awareness.

    This is the main public API for prediction alignment. It handles both 1D prediction
    vectors (e.g., chromatin accessibility, TF binding) and 2D matrices (e.g., Hi-C contact maps),
    routing to the appropriate alignment strategy based on variant type.

    IMPORTANT: Model-specific parameters (bin_size, matrix_size) must be explicitly
    provided by the user. There are no defaults because these vary across different models.

    Args:
        ref_preds: Reference predictions array (from model with edge cropping)
        alt_preds: Alt predictions array (same shape as ref_preds)
        metadata_row: Dictionary with variant information containing:
            - 'variant_type': Type of variant (SNV, INS, DEL, DUP, INV, BND)
            - 'window_start': Start position of window (0-based)
            - 'variant_pos0': Variant position (0-based, absolute genomic coordinate)
            - 'svlen': Length of structural variant (optional, for symbolic alleles)
        bin_size: Number of base pairs per prediction bin (REQUIRED, model-specific)
            Examples: 2048 for Akita
        prediction_type: Type of predictions ("1D" or "2D")
            - "1D": Vector predictions (chromatin accessibility, TF binding, etc.)
            - "2D": Matrix predictions (Hi-C contact maps, Micro-C, etc.)
        matrix_size: Size of contact matrix (REQUIRED for 2D type)
            Examples: 448 for Akita
        diag_offset: Number of diagonal bins to mask (default: 0 for no masking)
            Set to 0 if your model doesn't mask diagonals, or to model-specific value
            Examples: 2 for Akita, 0 for models without diagonal masking

    Returns:
        Tuple of (aligned_ref_preds, aligned_alt_preds) with NaN masking applied
        at positions that differ between reference and alternate sequences

    Raises:
        ValueError: If prediction_type is invalid, required parameters are missing,
            or variant type is unsupported

    Example (1D predictions):
        >>> ref_aligned, alt_aligned = align_predictions_by_coordinate(
        ...     ref_preds=ref_scores,
        ...     alt_preds=alt_scores,
        ...     metadata_row={'variant_type': 'INS', 'window_start': 0,
        ...                   'variant_pos0': 500, 'svlen': 100},
        ...     bin_size=128,
        ...     prediction_type="1D"
        ... )

    Example (2D contact maps with diagonal masking):
        >>> ref_aligned, alt_aligned = align_predictions_by_coordinate(
        ...     ref_preds=ref_contact_map,
        ...     alt_preds=alt_contact_map,
        ...     metadata_row={'variant_type': 'DEL', 'window_start': 0,
        ...                   'variant_pos0': 50000, 'svlen': -2048},
        ...     bin_size=2048,
        ...     prediction_type="2D",
        ...     matrix_size=448,
        ...     diag_offset=2  # Optional: use 0 if no diagonal masking
        ... )

    Example (2D contact maps without diagonal masking):
        >>> ref_aligned, alt_aligned = align_predictions_by_coordinate(
        ...     ref_preds=ref_contact_map,
        ...     alt_preds=alt_contact_map,
        ...     metadata_row={'variant_type': 'INS', 'window_start': 0,
        ...                   'variant_pos0': 1000, 'svlen': 500},
        ...     bin_size=1000,
        ...     prediction_type="2D",
        ...     matrix_size=512
        ...     # diag_offset defaults to 0 (no masking)
        ... )
    """
    # Validate prediction type and parameters
    if prediction_type not in ["1D", "2D"]:
        raise ValueError(f"prediction_type must be '1D' or '2D', got '{prediction_type}'")

    if prediction_type == "2D":
        if matrix_size is None:
            raise ValueError("matrix_size must be provided for 2D prediction type")

    # Extract variant information from metadata
    variant_type = metadata_row.get('variant_type', 'unknown')
    window_start = metadata_row.get('window_start', 0)
    variant_pos0 = metadata_row.get('variant_pos0')

    # For backward compatibility, check for effective_variant_start (deprecated)
    if variant_pos0 is not None:
        abs_variant_pos = variant_pos0
    else:
        # Fall back to old field name if present (for backward compatibility)
        effective_variant_start = metadata_row.get('effective_variant_start', 0)
        abs_variant_pos = window_start + effective_variant_start

    svlen = metadata_row.get('svlen', 0)

    # Create VariantPosition object
    var_pos = VariantPosition(
        ref_pos=abs_variant_pos,
        alt_pos=abs_variant_pos,
        svlen=svlen if svlen is not None else 0,
        variant_type=variant_type
    )

    # Determine target size from predictions
    if prediction_type == "1D":
        target_size = len(ref_preds)
        aligner = PredictionAligner1D(target_size=target_size, bin_size=bin_size)
        return aligner.align_predictions(ref_preds, alt_preds, variant_type, var_pos)
    else:  # 2D
        # Check if predictions are 1D (flattened upper triangular) or 2D (full matrix)
        is_torch = TORCH_AVAILABLE and torch.is_tensor(ref_preds)

        if is_torch:
            ndim = len(ref_preds.shape)
        else:
            ndim = ref_preds.ndim

        # If 1D, convert to 2D matrix
        if ndim == 1:
            ref_matrix = vector_to_contact_matrix(ref_preds, matrix_size)
            alt_matrix = vector_to_contact_matrix(alt_preds, matrix_size)

            # Align matrices
            aligner = PredictionAligner2D(
                target_size=matrix_size,
                bin_size=bin_size,
                diag_offset=diag_offset
            )
            aligned_ref_matrix, aligned_alt_matrix = aligner.align_predictions(
                ref_matrix, alt_matrix, variant_type, var_pos
            )

            # Convert back to flattened format
            aligned_ref_vector = contact_matrix_to_vector(aligned_ref_matrix)
            aligned_alt_vector = contact_matrix_to_vector(aligned_alt_matrix)

            return aligned_ref_vector, aligned_alt_vector
        else:
            # Already 2D matrices
            aligner = PredictionAligner2D(
                target_size=matrix_size,
                bin_size=bin_size,
                diag_offset=diag_offset
            )
            return aligner.align_predictions(ref_preds, alt_preds, variant_type, var_pos)
