# Prediction Alignment Examples

This directory contains visual examples demonstrating how prediction alignment works for different variant types in supremo_lite.

## Contents

### Generated Images (10 total)

**1D Predictions (Line Plots):**
- `snv_1d_alignment.png` - SNV: No masking needed
- `ins_1d_alignment.png` - Insertion: NaN in REF where ALT has extra sequence
- `del_1d_alignment.png` - Deletion: NaN in ALT where REF has deleted sequence
- `dup_1d_alignment.png` - Duplication: Same as insertion
- `inv_1d_alignment.png` - Inversion: Bilateral masking in inverted region

**2D Predictions (Contact Maps):**
- `snv_2d_alignment.png` - SNV: No masking needed
- `ins_2d_alignment.png` - Insertion: Cross-pattern NaN masking
- `del_2d_alignment.png` - Deletion: Cross-pattern NaN masking
- `dup_2d_alignment.png` - Duplication: Cross-pattern NaN masking
- `inv_2d_alignment.png` - Inversion: Extensive cross-pattern masking

## How Images Were Generated

All images are generated from **real test data** using:

```bash
poetry run python create_prediction_alignment_examples.py
```

### Data Sources
- **Test VCFs**: `tests/data/{snp,ins,del,dup,inv}/`
- **Reference genome**: `tests/data/test_genome.fa`
- **Models**: `TestModel` and `TestModel2D` from `supremo_lite.mock_models`

### Parameters
- Sequence length: 512 bp
- Bin size: 32 bp
- Crop length: 64 bp
- Target size: 12 bins
- Diagonal offset: 0

## Usage in Documentation

These images are displayed in the documentation page:
- **Source**: `docs/user_guide/prediction_alignment_examples.md`
- **Built HTML**: `docs/_build/html/user_guide/prediction_alignment_examples.html`

## Note on BND Variants

BND (breakend) variants are not included in the automated examples because they require special handling with predictions from two separate loci. For BND examples, see:
- Test suite: `tests/test_sv_prediction_alignment.py`
- Prototype: `dev_only/1D_prediction_alignment_prototype.py`

## Regenerating Images

To regenerate all images:

```bash
# From the repo root
poetry run python create_prediction_alignment_examples.py

# Images will be saved to this directory
# docs/_static/images/prediction_alignment_examples/
```

## Image Format

- **Format**: PNG
- **DPI**: 150
- **Size**: Automatically determined based on subplot layout
  - 1D: 14×5 inches (1x2 subplots)
  - 2D: 12×11 inches (2x2 subplots)
