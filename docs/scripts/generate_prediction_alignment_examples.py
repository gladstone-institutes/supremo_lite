"""
Generate visual examples of prediction alignment for all variant types.

This script creates comprehensive visualizations showing how prediction alignment
works for each variant type using synthetic predictions (based on dev_only prototypes).

The approach:
- Generate synthetic predictions with controlled patterns (REF = 0.5 * ALT)
- Apply alignment algorithms from supremo_lite.prediction_alignment
- Create clear visualizations showing before/after alignment

Output: Images saved to docs/_static/images/prediction_alignment_examples/
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from supremo_lite.prediction_alignment import (
    PredictionAligner1D,
    PredictionAligner2D,
    VariantPosition,
)

# Configuration
# Output directory relative to this script location (docs/scripts/)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "_static", "images", "prediction_alignment_examples"
)

# Model parameters - moderate size for clear visualization
TARGET_SIZE = 100  # Number of bins in predictions
BIN_SIZE = 128  # Base pairs per bin
DIAG_OFFSET = 0  # No diagonal masking for pedagogical clarity

print("=" * 80)
print("PREDICTION ALIGNMENT EXAMPLES GENERATOR")
print("=" * 80)
print(f"Target size: {TARGET_SIZE} bins")
print(f"Bin size: {BIN_SIZE} bp")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)


def generate_synthetic_predictions(size: int, num_peaks: int = 5,
                                  peak_width: int = 3, noise: float = 0.05,
                                  seed: int = 42) -> np.ndarray:
    """
    Generate synthetic 1D prediction scores mimicking chromatin accessibility.

    Creates realistic prediction patterns with:
    - Background baseline around 0.2
    - Peaks representing open chromatin regions
    - Gaussian-shaped peaks for biological realism
    - Random noise
    """
    predictions = np.ones(size) * 0.2

    np.random.seed(seed)
    peak_positions = np.random.choice(range(peak_width, size - peak_width),
                                     size=num_peaks, replace=False)

    for pos in peak_positions:
        x = np.arange(size)
        peak = 0.6 * np.exp(-((x - pos) ** 2) / (2 * peak_width ** 2))
        predictions += peak

    predictions += np.random.normal(0, noise, size)
    predictions = np.clip(predictions, 0, 1)

    return predictions


def generate_variant_predictions_1d(
    size: int,
    svtype: str,
    variant_bin: int,
    svlen_bins: int,
    num_peaks: int = 6
) -> tuple:
    """
    Generate REF and ALT prediction pairs with actual sequence differences.

    This creates variant-specific differences:
    - INS: ALT has inserted bins, REF doesn't
    - DEL: REF has extra bins, ALT doesn't
    - DUP: ALT has duplicated region
    - INV: Opposite gradients in the inverted region
    - REF = 0.5 * ALT for clear visual comparison
    """
    if svtype == 'INS':
        # For insertion: ALT has extra sequence
        alt_base = generate_synthetic_predictions(size, num_peaks=num_peaks, seed=42)

        # ALT: Insert duplicate bins at variant position
        insert_start = max(0, variant_bin - svlen_bins // 2)
        insert_region = alt_base[insert_start:insert_start + svlen_bins].copy()

        alt_pred = np.concatenate([
            alt_base[:variant_bin],
            insert_region,
            alt_base[variant_bin:]
        ])

        # Crop ALT to same size (remove from edges)
        crop_left = svlen_bins // 2
        crop_right = svlen_bins - crop_left
        alt_pred = alt_pred[crop_left:len(alt_pred) - crop_right]

        # REF is simply half of ALT (without the insertion)
        ref_pred = alt_base * 0.5

    elif svtype == 'DEL':
        # For deletion: REF has extra sequence that ALT lacks
        ref_base = generate_synthetic_predictions(size, num_peaks=num_peaks, seed=42)

        # REF: Insert duplicate bins at variant position
        insert_start = max(0, variant_bin - svlen_bins // 2)
        insert_region = ref_base[insert_start:insert_start + svlen_bins].copy()

        ref_pred = np.concatenate([
            ref_base[:variant_bin],
            insert_region,
            ref_base[variant_bin:]
        ])

        # Crop REF to same size
        crop_left = svlen_bins // 2
        crop_right = svlen_bins - crop_left
        ref_pred = ref_pred[crop_left:len(ref_pred) - crop_right]

        # ALT is simply half of REF (without the deletion)
        alt_pred = ref_base * 0.5

    elif svtype == 'DUP':
        # Duplication is like insertion - ALT has duplicated sequence
        alt_base = generate_synthetic_predictions(size, num_peaks=num_peaks, seed=42)

        # ALT: Duplicate the region at variant position
        dup_region = alt_base[variant_bin:variant_bin + svlen_bins].copy()

        alt_pred = np.concatenate([
            alt_base[:variant_bin + svlen_bins],
            dup_region,
            alt_base[variant_bin + svlen_bins:]
        ])

        # Crop ALT to same size
        crop_left = svlen_bins // 2
        crop_right = svlen_bins - crop_left
        alt_pred = alt_pred[crop_left:len(alt_pred) - crop_right]

        # REF is simply half of ALT (without the duplication)
        ref_pred = alt_base * 0.5

    elif svtype == 'INV':
        # For inversion: Create asymmetric pattern to show reversal clearly
        base = generate_synthetic_predictions(size, num_peaks=num_peaks, seed=42)

        # Create asymmetric pattern in the inverted region
        inv_start = variant_bin
        inv_end = min(size, variant_bin + svlen_bins)

        # Create a strong linear gradient in the inversion region
        inv_length = inv_end - inv_start
        gradient = np.linspace(0.3, 0.9, inv_length)

        # ALT: Apply gradient pattern
        alt_pred = base.copy()
        alt_pred[inv_start:inv_end] = gradient

        # REF: Apply reversed gradient pattern at half magnitude
        ref_pred = base.copy() * 0.5
        ref_pred[inv_start:inv_end] = gradient[::-1] * 0.5

    else:
        raise ValueError(f"Unknown variant type: {svtype}")

    return ref_pred, alt_pred


def generate_synthetic_contact_map(size: int, decay_rate: float = 0.05,
                                  noise: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic Hi-C contact map with distance-decay pattern.

    Hi-C data shows that chromatin interactions decay with genomic distance,
    creating the characteristic triangular pattern in contact maps.
    """
    np.random.seed(seed)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(i, size):
            distance = j - i
            value = np.exp(-decay_rate * distance) + np.random.normal(0, noise)
            value = max(0, value)
            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


def visualize_1d_alignment(ref_orig: np.ndarray, alt_orig: np.ndarray,
                          ref_aligned: np.ndarray, alt_aligned: np.ndarray,
                          variant_type: str, variant_bin: int,
                          output_path: str) -> None:
    """
    Visualize 1D predictions showing original and aligned versions.

    Layout: 1x2 subplot
    - Left: Original predictions overlaid
    - Right: Aligned predictions overlaid with NaN regions highlighted
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins_orig = np.arange(len(ref_orig))
    bins_aligned = np.arange(len(ref_aligned))

    # Left panel: Original predictions
    axes[0].plot(bins_orig, ref_orig, 'b-', linewidth=2, label='REF', alpha=0.8)
    axes[0].plot(bins_orig, alt_orig, 'r-', linewidth=2, label='ALT', alpha=0.8)
    axes[0].axvline(variant_bin, color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Variant position')
    axes[0].set_title('Original Predictions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Bin position', fontsize=11)
    axes[0].set_ylabel('Prediction score', fontsize=11)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Right panel: Aligned predictions
    axes[1].plot(bins_aligned, ref_aligned, 'b-', linewidth=2, label='REF (aligned)', alpha=0.8)
    axes[1].plot(bins_aligned, alt_aligned, 'r-', linewidth=2, label='ALT (aligned)', alpha=0.8)
    axes[1].axvline(variant_bin, color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Variant position')

    # Highlight NaN regions with shading
    nan_mask_ref = np.isnan(ref_aligned)
    nan_mask_alt = np.isnan(alt_aligned)

    # Check if masked regions are identical
    masks_identical = np.array_equal(nan_mask_ref, nan_mask_alt)

    if masks_identical and nan_mask_ref.any():
        # Show single masked region when REF and ALT masks overlap
        nan_regions = np.where(nan_mask_ref)[0]
        if len(nan_regions) > 0:
            axes[1].axvspan(nan_regions[0], nan_regions[-1] + 1,
                          alpha=0.2, color='gray', label='Masked region')
    else:
        # Show separate regions when they differ
        if nan_mask_ref.any():
            nan_regions_ref = np.where(nan_mask_ref)[0]
            if len(nan_regions_ref) > 0:
                axes[1].axvspan(nan_regions_ref[0], nan_regions_ref[-1] + 1,
                              alpha=0.2, color='blue', label='Masked (REF)', hatch='///')

        if nan_mask_alt.any():
            nan_regions_alt = np.where(nan_mask_alt)[0]
            if len(nan_regions_alt) > 0:
                axes[1].axvspan(nan_regions_alt[0], nan_regions_alt[-1] + 1,
                              alpha=0.2, color='red', label='Masked (ALT)', hatch='\\\\\\')

    axes[1].set_title('Aligned Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Bin position', fontsize=11)
    axes[1].set_ylabel('Prediction score', fontsize=11)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'1D Prediction Alignment: {variant_type} at bin {variant_bin}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved {os.path.basename(output_path)}")


def visualize_2d_alignment(ref_orig: np.ndarray, alt_orig: np.ndarray,
                          ref_aligned: np.ndarray, alt_aligned: np.ndarray,
                          variant_type: str, variant_bin: int,
                          output_path: str) -> None:
    """
    Visualize 2D contact maps showing original and aligned versions.

    Layout: 2x2 grid
    - Top: Original REF and ALT matrices
    - Bottom: Aligned REF and ALT matrices
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # Determine color scale across all matrices
    all_data = [ref_orig, alt_orig, ref_aligned, alt_aligned]
    vmin = min(np.nanmin(m) for m in all_data)
    vmax = max(np.nanmax(m) for m in all_data)

    # Top left: Original REF
    im0 = axes[0, 0].imshow(ref_orig, cmap='Reds', vmin=vmin, vmax=vmax,
                            origin='upper', aspect='auto')
    axes[0, 0].set_title('Original REF', fontsize=11, fontweight='bold')
    axes[0, 0].axvline(variant_bin, color='blue', linestyle='--', linewidth=1, alpha=0.6)
    axes[0, 0].axhline(variant_bin, color='blue', linestyle='--', linewidth=1, alpha=0.6)
    axes[0, 0].set_xlabel('Bin position')
    axes[0, 0].set_ylabel('Bin position')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Top right: Original ALT
    im1 = axes[0, 1].imshow(alt_orig, cmap='Reds', vmin=vmin, vmax=vmax,
                            origin='upper', aspect='auto')
    axes[0, 1].set_title('Original ALT', fontsize=11, fontweight='bold')
    axes[0, 1].axvline(variant_bin, color='blue', linestyle='--', linewidth=1, alpha=0.6)
    axes[0, 1].axhline(variant_bin, color='blue', linestyle='--', linewidth=1, alpha=0.6)
    axes[0, 1].set_xlabel('Bin position')
    axes[0, 1].set_ylabel('Bin position')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Bottom left: Aligned REF
    im2 = axes[1, 0].imshow(ref_aligned, cmap='Reds', vmin=vmin, vmax=vmax,
                            origin='upper', aspect='auto')
    axes[1, 0].set_title('Aligned REF', fontsize=11, fontweight='bold', color='darkblue')

    axes[1, 0].axvline(variant_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.8,
                      label='Variant position')
    axes[1, 0].axhline(variant_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xlabel('Bin position')
    axes[1, 0].set_ylabel('Bin position')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Bottom right: Aligned ALT
    im3 = axes[1, 1].imshow(alt_aligned, cmap='Reds', vmin=vmin, vmax=vmax,
                            origin='upper', aspect='auto')
    axes[1, 1].set_title('Aligned ALT', fontsize=11, fontweight='bold', color='darkred')

    axes[1, 1].axvline(variant_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.8,
                      label='Variant position')
    axes[1, 1].axhline(variant_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_xlabel('Bin position')
    axes[1, 1].set_ylabel('Bin position')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(f'2D Prediction Alignment: {variant_type} at bin {variant_bin}',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved {os.path.basename(output_path)}")


def visualize_bnd_1d(left_ref: np.ndarray, right_ref: np.ndarray, bnd_alt: np.ndarray,
                    chimeric_ref: np.ndarray, breakpoint_bin: int, output_path: str) -> None:
    """
    Visualize BND 1D alignment with dual reference loci.

    Layout: 3x2 grid
    - Top row: Left and Right locus references (original)
    - Middle row: Chimeric REF and BND ALT (assembled sequences)
    - Bottom row: Aligned comparison (overlaid)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    bins = np.arange(len(left_ref))

    # Top left: Left REF
    axes[0, 0].plot(bins, left_ref, 'b-', linewidth=2, label='Left REF')
    axes[0, 0].axvline(breakpoint_bin, color='green', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='Breakpoint')
    axes[0, 0].axvspan(0, breakpoint_bin, alpha=0.1, color='green', label='Used region')
    axes[0, 0].set_title('Left Locus REF (Original)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Bin position')
    axes[0, 0].set_ylabel('Prediction score')
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Right REF
    axes[0, 1].plot(bins, right_ref, 'b-', linewidth=2, label='Right REF')
    axes[0, 1].axvline(breakpoint_bin, color='green', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='Breakpoint')
    axes[0, 1].axvspan(breakpoint_bin, len(bins), alpha=0.1, color='green', label='Used region')
    axes[0, 1].set_title('Right Locus REF (Original)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Bin position')
    axes[0, 1].set_ylabel('Prediction score')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # Middle left: Chimeric REF (use prototype strategy - plot directly, matplotlib handles NaN)
    axes[1, 0].plot(bins, chimeric_ref, 'b-', linewidth=2, label='Chimeric REF', alpha=0.8)
    axes[1, 0].axvline(breakpoint_bin, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label='Fusion point')
    axes[1, 0].text(breakpoint_bin/2, 0.9, 'L', ha='center', fontsize=14,
                   fontweight='bold', color='green')
    axes[1, 0].text((len(bins)+breakpoint_bin)/2, 0.9, 'R', ha='center', fontsize=14,
                   fontweight='bold', color='green')
    axes[1, 0].set_title('Chimeric REF (Assembled)', fontsize=12, fontweight='bold', color='darkblue')
    axes[1, 0].set_xlabel('Bin position')
    axes[1, 0].set_ylabel('Prediction score')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # Middle right: BND ALT
    axes[1, 1].plot(bins, bnd_alt, 'r-', linewidth=2, label='BND ALT', alpha=0.8)
    axes[1, 1].axvline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='Breakpoint')
    axes[1, 1].set_title('BND ALT (Fusion)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Bin position')
    axes[1, 1].set_ylabel('Prediction score')
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    # Bottom row spans both columns: Aligned comparison
    axes[2, 0].remove()
    axes[2, 1].remove()
    ax_aligned = fig.add_subplot(3, 1, 3)

    # Plot using prototype strategy - plot directly and let matplotlib handle NaN
    ax_aligned.plot(bins, chimeric_ref, 'b-', linewidth=2, label='Chimeric REF (aligned)', alpha=0.8)
    ax_aligned.plot(bins, bnd_alt, 'r-', linewidth=2, label='BND ALT (original)', alpha=0.8)

    ax_aligned.axvline(breakpoint_bin, color='purple', linestyle='--', linewidth=1.5,
                      alpha=0.5, label='Breakpoint')
    ax_aligned.set_title('Aligned Predictions Comparison', fontsize=12, fontweight='bold')
    ax_aligned.set_xlabel('Bin position')
    ax_aligned.set_ylabel('Prediction score')
    ax_aligned.set_ylim(-0.05, 1.05)
    ax_aligned.legend(loc='upper right')
    ax_aligned.grid(True, alpha=0.3)

    fig.suptitle(f'1D Prediction Alignment: BND at bin {breakpoint_bin}\n'
                 'Top: Source loci | Middle: Assembled | Bottom: Aligned comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved {os.path.basename(output_path)}")


def visualize_bnd_2d(left_ref: np.ndarray, right_ref: np.ndarray, bnd_alt: np.ndarray,
                    chimeric_ref: np.ndarray, breakpoint_bin: int, output_path: str) -> None:
    """
    Visualize BND 2D alignment with quadrant assembly.

    Layout: 2x3 grid showing the assembly process
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    vmin = min(np.nanmin(left_ref), np.nanmin(right_ref),
              np.nanmin(chimeric_ref), np.nanmin(bnd_alt))
    vmax = max(np.nanmax(left_ref), np.nanmax(right_ref),
              np.nanmax(chimeric_ref), np.nanmax(bnd_alt))

    # Top row: Original matrices
    ax1 = fig.add_subplot(gs[0, 0])
    # Flip matrix left-right to match flipped x-axis
    left_ref_flipped = np.fliplr(left_ref)
    im1 = ax1.imshow(left_ref_flipped, cmap='Reds', vmin=vmin, vmax=vmax, origin='upper')
    ax1.set_title('Left Locus REF', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Bin position')
    ax1.set_ylabel('Bin position')
    ax1.invert_xaxis()  # Flip x-axis for standard Hi-C orientation
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    # Flip matrix left-right to match flipped x-axis
    right_ref_flipped = np.fliplr(right_ref)
    im2 = ax2.imshow(right_ref_flipped, cmap='Reds', vmin=vmin, vmax=vmax, origin='upper')
    ax2.set_title('Right Locus REF', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Bin position')
    ax2.set_ylabel('Bin position')
    ax2.invert_xaxis()  # Flip x-axis for standard Hi-C orientation
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    # Flip matrix left-right to match flipped x-axis
    bnd_alt_flipped = np.fliplr(bnd_alt)
    im3 = ax3.imshow(bnd_alt_flipped, cmap='Reds', vmin=vmin, vmax=vmax, origin='upper')
    ax3.set_title('BND ALT (Fusion)', fontsize=12, fontweight='bold')
    ax3.axvline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Bin position')
    ax3.set_ylabel('Bin position')
    ax3.invert_xaxis()  # Flip x-axis for standard Hi-C orientation
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Bottom row: Assembly process
    # LEFTMOST: Left Used Quadrant
    ax4 = fig.add_subplot(gs[1, 0])
    # Flip matrix left-right to match flipped x-axis (reuse from above)
    im4 = ax4.imshow(left_ref_flipped, cmap='Reds', vmin=vmin, vmax=vmax, origin='upper')
    ax4.set_title('Left: Used Quadrant', fontsize=12, fontweight='bold', color='green')
    ax4.axvline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    # Adjust rectangle position for flipped matrix: upper-left quadrant is now at upper-right
    matrix_size = left_ref.shape[0]
    rect_x = matrix_size - breakpoint_bin
    rect = patches.Rectangle((rect_x, 0), breakpoint_bin, breakpoint_bin,
                            linewidth=3, edgecolor='green', facecolor='none', alpha=0.5)
    ax4.add_patch(rect)
    # Adjust text position for flipped matrix
    text_x = matrix_size - breakpoint_bin/2
    ax4.text(text_x, breakpoint_bin/2, 'Used',
            ha='center', va='center', color='green', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Bin position')
    ax4.set_ylabel('Bin position')
    ax4.invert_xaxis()  # Flip x-axis for standard Hi-C orientation
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # MIDDLE: Right Used Quadrant
    ax6 = fig.add_subplot(gs[1, 1])
    # Flip matrix left-right to match flipped x-axis (reuse from above)
    im6 = ax6.imshow(right_ref_flipped, cmap='Reds', vmin=vmin, vmax=vmax, origin='upper')
    ax6.set_title('Right: Used Quadrant', fontsize=12, fontweight='bold', color='green')
    ax6.axvline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.axhline(breakpoint_bin, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    matrix_size = right_ref.shape[0]
    # Adjust rectangle position for flipped matrix: lower-right quadrant is now at lower-left
    rect2_x = 0
    rect2 = patches.Rectangle((rect2_x, breakpoint_bin),
                             matrix_size-breakpoint_bin, matrix_size-breakpoint_bin,
                             linewidth=3, edgecolor='green', facecolor='none', alpha=0.5)
    ax6.add_patch(rect2)
    # Adjust text position for flipped matrix
    text_x = (matrix_size - breakpoint_bin)/2
    ax6.text(text_x, (matrix_size+breakpoint_bin)/2, 'Used',
            ha='center', va='center', color='green', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Bin position')
    ax6.set_ylabel('Bin position')
    ax6.invert_xaxis()  # Flip x-axis for standard Hi-C orientation
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # RIGHTMOST: Chimeric REF (Aligned)
    ax5 = fig.add_subplot(gs[1, 2])
    # Flip matrix left-right to match flipped x-axis
    chimeric_ref_flipped = np.fliplr(chimeric_ref)
    im5 = ax5.imshow(chimeric_ref_flipped, cmap='Reds', vmin=vmin, vmax=vmax, origin='upper')
    ax5.set_title('Chimeric REF (Aligned)', fontsize=12, fontweight='bold', color='darkred')
    ax5.axvline(breakpoint_bin, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax5.axhline(breakpoint_bin, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    matrix_size = chimeric_ref.shape[0]
    # Adjust text positions for flipped matrix
    # Left quadrant is now on the right side of the display
    text_L_x = matrix_size - breakpoint_bin/2
    ax5.text(text_L_x, breakpoint_bin/2, 'L',
            ha='center', va='center', color='green', fontsize=16, fontweight='bold')
    # Right quadrant is now on the left side of the display
    text_R_x = (matrix_size - breakpoint_bin)/2
    ax5.text(text_R_x, (matrix_size+breakpoint_bin)/2, 'R',
            ha='center', va='center', color='green', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Bin position')
    ax5.set_ylabel('Bin position')
    ax5.invert_xaxis()  # Flip x-axis for standard Hi-C orientation
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    fig.suptitle(f'2D Prediction Alignment: BND at bin {breakpoint_bin}\n'
                 'Top: Original predictions | Bottom: Assembly process',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved {os.path.basename(output_path)}")


def main():
    """Generate all prediction alignment examples using synthetic data."""

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize aligners
    aligner_1d = PredictionAligner1D(target_size=TARGET_SIZE, bin_size=BIN_SIZE)
    aligner_2d = PredictionAligner2D(target_size=TARGET_SIZE, bin_size=BIN_SIZE,
                                     diag_offset=DIAG_OFFSET)

    # Test parameters
    # Use an odd svlen_bins so centering is visually obvious
    # For a 100-bin prediction with variant at bin 50 (center):
    # - With 5 bins of masking, centered masking would be bins 48-52 (centered on 50)
    # - With right-aligned masking (the bug), it would be bins 50-54 (starting at 50)
    variant_bin = 50  # Middle of the sequence (bin 50 out of 100 bins)
    svlen_bins = 5  # Use odd number to make centering obvious
    svlen = svlen_bins * BIN_SIZE  # 5 bins * 128bp = 640bp

    # ========================================================================
    # 1. INSERTION
    # ========================================================================
    print("\n" + "="*80)
    print("Processing INS")
    print("="*80)

    # Generate variant-specific predictions
    ref_pred_ins, alt_pred_ins = generate_variant_predictions_1d(
        TARGET_SIZE, 'INS', variant_bin, svlen_bins
    )

    var_pos = VariantPosition(ref_pos=variant_bin * BIN_SIZE,
                              alt_pos=variant_bin * BIN_SIZE,
                              svlen=svlen,
                              variant_type='INS')

    # For synthetic data, window starts at 0
    window_start = 0

    ref_aligned_ins, alt_aligned_ins = aligner_1d.align_predictions(
        ref_pred_ins, alt_pred_ins, 'INS', var_pos, window_start
    )

    # 1D visualization
    visualize_1d_alignment(ref_pred_ins, alt_pred_ins, ref_aligned_ins, alt_aligned_ins,
                          'INS', variant_bin,
                          os.path.join(OUTPUT_DIR, 'ins_1d_alignment.png'))

    # 2D predictions
    ref_matrix_ins = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=42) * 0.5
    alt_matrix_ins = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=42)

    ref_aligned_ins_2d, alt_aligned_ins_2d = aligner_2d.align_predictions(
        ref_matrix_ins, alt_matrix_ins, 'INS', var_pos, window_start
    )

    visualize_2d_alignment(ref_matrix_ins, alt_matrix_ins,
                          ref_aligned_ins_2d, alt_aligned_ins_2d,
                          'INS', variant_bin,
                          os.path.join(OUTPUT_DIR, 'ins_2d_alignment.png'))

    print("   ✓ Completed INS")

    # ========================================================================
    # 2. DELETION
    # ========================================================================
    print("\n" + "="*80)
    print("Processing DEL")
    print("="*80)

    ref_pred_del, alt_pred_del = generate_variant_predictions_1d(
        TARGET_SIZE, 'DEL', variant_bin, svlen_bins
    )

    var_pos = VariantPosition(ref_pos=variant_bin * BIN_SIZE,
                              alt_pos=variant_bin * BIN_SIZE,
                              svlen=svlen,
                              variant_type='DEL')

    ref_aligned_del, alt_aligned_del = aligner_1d.align_predictions(
        ref_pred_del, alt_pred_del, 'DEL', var_pos, window_start
    )

    visualize_1d_alignment(ref_pred_del, alt_pred_del, ref_aligned_del, alt_aligned_del,
                          'DEL', variant_bin,
                          os.path.join(OUTPUT_DIR, 'del_1d_alignment.png'))

    # 2D
    ref_matrix_del = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=43) * 0.5
    alt_matrix_del = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=43)

    ref_aligned_del_2d, alt_aligned_del_2d = aligner_2d.align_predictions(
        ref_matrix_del, alt_matrix_del, 'DEL', var_pos, window_start
    )

    visualize_2d_alignment(ref_matrix_del, alt_matrix_del,
                          ref_aligned_del_2d, alt_aligned_del_2d,
                          'DEL', variant_bin,
                          os.path.join(OUTPUT_DIR, 'del_2d_alignment.png'))

    print("   ✓ Completed DEL")

    # ========================================================================
    # 3. DUPLICATION
    # ========================================================================
    print("\n" + "="*80)
    print("Processing DUP")
    print("="*80)

    ref_pred_dup, alt_pred_dup = generate_variant_predictions_1d(
        TARGET_SIZE, 'DUP', variant_bin, svlen_bins
    )

    var_pos = VariantPosition(ref_pos=variant_bin * BIN_SIZE,
                              alt_pos=variant_bin * BIN_SIZE,
                              svlen=svlen,
                              variant_type='DUP')

    ref_aligned_dup, alt_aligned_dup = aligner_1d.align_predictions(
        ref_pred_dup, alt_pred_dup, 'DUP', var_pos, window_start
    )

    visualize_1d_alignment(ref_pred_dup, alt_pred_dup, ref_aligned_dup, alt_aligned_dup,
                          'DUP', variant_bin,
                          os.path.join(OUTPUT_DIR, 'dup_1d_alignment.png'))

    # 2D
    ref_matrix_dup = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=44) * 0.5
    alt_matrix_dup = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=44)

    ref_aligned_dup_2d, alt_aligned_dup_2d = aligner_2d.align_predictions(
        ref_matrix_dup, alt_matrix_dup, 'DUP', var_pos, window_start
    )

    visualize_2d_alignment(ref_matrix_dup, alt_matrix_dup,
                          ref_aligned_dup_2d, alt_aligned_dup_2d,
                          'DUP', variant_bin,
                          os.path.join(OUTPUT_DIR, 'dup_2d_alignment.png'))

    print("   ✓ Completed DUP")

    # ========================================================================
    # 4. INVERSION
    # ========================================================================
    print("\n" + "="*80)
    print("Processing INV")
    print("="*80)

    ref_pred_inv, alt_pred_inv = generate_variant_predictions_1d(
        TARGET_SIZE, 'INV', variant_bin, svlen_bins
    )

    var_pos = VariantPosition(ref_pos=variant_bin * BIN_SIZE,
                              alt_pos=variant_bin * BIN_SIZE,
                              svlen=svlen,
                              variant_type='INV')

    ref_aligned_inv, alt_aligned_inv = aligner_1d.align_predictions(
        ref_pred_inv, alt_pred_inv, 'INV', var_pos, window_start
    )

    visualize_1d_alignment(ref_pred_inv, alt_pred_inv, ref_aligned_inv, alt_aligned_inv,
                          'INV', variant_bin,
                          os.path.join(OUTPUT_DIR, 'inv_1d_alignment.png'))

    # 2D
    ref_matrix_inv = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=45) * 0.5
    alt_matrix_inv = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=45)

    ref_aligned_inv_2d, alt_aligned_inv_2d = aligner_2d.align_predictions(
        ref_matrix_inv, alt_matrix_inv, 'INV', var_pos, window_start
    )

    visualize_2d_alignment(ref_matrix_inv, alt_matrix_inv,
                          ref_aligned_inv_2d, alt_aligned_inv_2d,
                          'INV', variant_bin,
                          os.path.join(OUTPUT_DIR, 'inv_2d_alignment.png'))

    print("   ✓ Completed INV")

    # ========================================================================
    # 5. BREAKEND
    # ========================================================================
    print("\n" + "="*80)
    print("Processing BND")
    print("="*80)

    # Generate predictions from two separate loci (full amplitude for ALT)
    left_full = generate_synthetic_predictions(TARGET_SIZE, num_peaks=5, seed=50)
    right_full = generate_synthetic_predictions(TARGET_SIZE, num_peaks=5, seed=51)

    # BND ALT should show fusion: combine left and right patterns at breakpoint
    # Use same pattern as other variants: REF = 0.5 * ALT for clear visual difference
    breakpoint_bin = 50
    bnd_alt = np.zeros(TARGET_SIZE)
    bnd_alt[:breakpoint_bin] = left_full[:breakpoint_bin]  # Left locus pattern (full amplitude)
    bnd_alt[breakpoint_bin:] = right_full[breakpoint_bin:]  # Right locus pattern (full amplitude)

    # Create scaled-down REF versions to emphasize the change
    left_ref = left_full * 0.5
    right_ref = right_full * 0.5

    # 1D: Create chimeric reference
    chimeric_ref_1d, bnd_alt_aligned_1d = aligner_1d.align_bnd_predictions(
        left_ref, right_ref, bnd_alt, breakpoint_bin
    )

    visualize_bnd_1d(left_ref, right_ref, bnd_alt, chimeric_ref_1d, breakpoint_bin,
                    os.path.join(OUTPUT_DIR, 'bnd_1d_alignment.png'))

    # 2D: Create chimeric matrix (full amplitude for ALT)
    left_full_2d = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=50)
    right_full_2d = generate_synthetic_contact_map(TARGET_SIZE, decay_rate=0.05, seed=51)

    # BND ALT should show a fusion: blend left and right patterns at breakpoint
    # Create by taking upper-left from left and lower-right from right (full amplitude)
    bnd_alt_2d = np.zeros((TARGET_SIZE, TARGET_SIZE))
    # Upper-left quadrant from left locus (full amplitude)
    bnd_alt_2d[:breakpoint_bin, :breakpoint_bin] = left_full_2d[:breakpoint_bin, :breakpoint_bin]
    # Lower-right quadrant from right locus (full amplitude)
    bnd_alt_2d[breakpoint_bin:, breakpoint_bin:] = right_full_2d[breakpoint_bin:, breakpoint_bin:]
    # Upper-right and lower-left quadrants: blend both loci to show trans-interactions (full amplitude)
    bnd_alt_2d[:breakpoint_bin, breakpoint_bin:] = (left_full_2d[:breakpoint_bin, breakpoint_bin:] +
                                                     right_full_2d[:breakpoint_bin, breakpoint_bin:]) / 2
    bnd_alt_2d[breakpoint_bin:, :breakpoint_bin] = (left_full_2d[breakpoint_bin:, :breakpoint_bin] +
                                                     right_full_2d[breakpoint_bin:, :breakpoint_bin]) / 2

    # Create scaled-down REF matrices to emphasize the change
    left_ref_2d = left_full_2d * 0.5
    right_ref_2d = right_full_2d * 0.5

    chimeric_ref_2d, bnd_alt_aligned_2d = aligner_2d.align_bnd_matrices(
        left_ref_2d, right_ref_2d, bnd_alt_2d, breakpoint_bin
    )

    visualize_bnd_2d(left_ref_2d, right_ref_2d, bnd_alt_2d, chimeric_ref_2d, breakpoint_bin,
                    os.path.join(OUTPUT_DIR, 'bnd_2d_alignment.png'))

    print("   ✓ Completed BND")

    # ========================================================================
    # COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"All images saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for variant in ['ins', 'del', 'dup', 'inv', 'bnd']:
        print(f"  - {variant}_1d_alignment.png")
        print(f"  - {variant}_2d_alignment.png")


if __name__ == "__main__":
    main()
