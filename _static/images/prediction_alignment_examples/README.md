# Prediction Alignment Examples

This directory contains visual examples demonstrating how prediction alignment works for different variant types in supremo_lite.

## Contents

### Generated Images (10 total)

**1D Predictions (Line Plots):**
- `ins_1d_alignment.png` - Insertion: Masked bins centered on variant position
- `del_1d_alignment.png` - Deletion: Masked bins centered on variant position
- `dup_1d_alignment.png` - Duplication: Masked bins centered on variant position
- `inv_1d_alignment.png` - Inversion: Bilateral masking in inverted region
- `bnd_1d_alignment.png` - Breakend: Chimeric sequence assembly from two loci

**2D Predictions (Contact Maps):**
- `ins_2d_alignment.png` - Insertion: Cross-pattern NaN masking centered on variant
- `del_2d_alignment.png` - Deletion: Cross-pattern NaN masking centered on variant
- `dup_2d_alignment.png` - Duplication: Cross-pattern NaN masking centered on variant
- `inv_2d_alignment.png` - Inversion: Extensive cross-pattern masking
- `bnd_2d_alignment.png` - Breakend: Chimeric matrix assembly from two loci

## How Images Were Generated

All images are generated from **synthetic predictions** using:

```bash
# From the repo root
poetry run python docs/scripts/generate_prediction_alignment_examples.py
```

### Data Generation
- **Synthetic predictions**: Generated with controlled patterns (REF = 0.5 × ALT for clarity)
- **Noise**: Random normal noise added for realistic appearance
- **Peaks**: Multiple peaks distributed across prediction vectors for visual interest

### Parameters
- **Target size**: 100 bins
- **Bin size**: 128 bp
- **Variant position**: Bin 50 (centered)
- **SV length**: 5 bins (640 bp) - odd number makes centering visually obvious
- **Diagonal offset**: 0 (no diagonal masking for pedagogical clarity)

## Key Features Demonstrated

### Centered Masking
All visualizations clearly show that masked bins are **centered on the variant position**:
- For a variant at bin 50 with 5 bins of masking, bins 48-52 are masked
- The blue dashed lines mark the variant position
- The masked region (gray shaded area in 1D, white cross in 2D) is symmetric around this position

### Legend Clarity
- **1D plots**: When REF and ALT masks overlap (most cases), shows single "Masked region" label
- **2D plots**: Masked regions visible as white cross-pattern where NaN values appear

## Usage in Documentation

These images can be displayed in documentation to illustrate prediction alignment behavior.

## Regenerating Images

To regenerate all images after code changes:

```bash
# From the repo root
poetry run python docs/scripts/generate_prediction_alignment_examples.py

# Images will be saved to:
# docs/_static/images/prediction_alignment_examples/
```

## Image Format

- **Format**: PNG
- **DPI**: 150
- **Size**: Automatically determined based on subplot layout
  - 1D: 14×5 inches (1×2 subplots)
  - 2D: 12×11 inches (2×2 subplots)
  - BND 1D: 14×14 inches (3×1 subplots with merged bottom panel)
  - BND 2D: 15×10 inches (2 rows, custom grid layout)
