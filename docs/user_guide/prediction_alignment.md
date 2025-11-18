# Prediction Alignment

Align model predictions to account for coordinate changes from variants.

## Overview

Indels and structural variants change sequence length, making direct prediction comparison invalid. `align_predictions_by_coordinate()` inserts NaN bins to maintain genomic coordinate correspondence.

## Function Signature

```python
align_predictions_by_coordinate(
    ref_pred,               # Reference predictions (array/tensor)
    alt_pred,               # Alternate predictions (array/tensor)
    metadata,               # Variant metadata dict
    prediction_type,        # "1D" or "2D"
    bin_size,               # int: Model bin size
    crop_length,            # int: Model crop length (REQUIRED)
    diag_offset=0,          # int: Diagonal offset for 2D (default 0)
    matrix_size=None        # int: Required for 2D predictions
) -> tuple[array/tensor, array/tensor]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_pred` | array/tensor | required | Reference predictions |
| `alt_pred` | array/tensor | required | Alternate predictions |
| `metadata` | dict | required | Variant metadata (from `get_alt_ref_sequences()`) |
| `prediction_type` | str | required | `"1D"` or `"2D"` |
| `bin_size` | int | required | Model binning resolution (bp per prediction) |
| `crop_length` | int | **required** | Bases cropped from each edge by model |
| `diag_offset` | int | 0 | Diagonal bins masked (2D only) |
| `matrix_size` | int | None | Square matrix size (required for 2D) |

## Alignment Strategies by Variant Type

| Variant | Strategy |
|---------|----------|
| SNV | No alignment needed (same length) |
| INS | NaN bins in reference where insertion occurs |
| DEL | NaN bins in alternate where deletion occurs |
| DUP | Same as INS (duplication adds sequence) |
| INV | 1D: mask inverted bins; 2D: cross-pattern masking (rows + columns) |
| BND | Chimeric reference comparison |

:::{tip}
For details on how variants are classified into these types, see the [Variant Classification Flow Chart](../_static/images/variant_classification.png).
:::

## Model Architecture

**Binning**: Models predict at lower resolution. Example: `bin_size=8` means 1 prediction per 8 bp.

**Edge Cropping**: Models may crop edges. Example: `crop_length=10` removes 10 bp from each end.

**Diagonal Masking** (2D only): Contact maps often mask diagonal. Example: `diag_offset=2` masks 2 bins from diagonal.

## Examples

### Basic 1D Alignment

```python
import supremo_lite as sl

# Generate sequences
results = list(sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200
))
alt_seqs, ref_seqs, metadata = results[0]

# Run 1D model
from supremo_lite.mock_models import TestModel
model = TestModel(n_targets=2, bin_size=8, crop_length=10)
ref_preds = model(ref_seqs)
alt_preds = model(alt_seqs)

# Align predictions for first variant
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds[0],      # Shape: (2, 22) [2 targets, 22 bins]
    alt_pred=alt_preds[0],
    metadata=metadata[0],
    prediction_type="1D",
    bin_size=8,
    crop_length=10
)
```

### 2D Contact Map Alignment

```python
from supremo_lite.mock_models import TestModel2D

model_2d = TestModel2D(n_targets=1, bin_size=8, crop_length=10, diag_offset=2)
ref_preds_2d = model_2d(ref_seqs)
alt_preds_2d = model_2d(alt_seqs)

matrix_size = (200 - 2*10) // 8  # (seq_len - 2*crop_length) // bin_size

ref_aligned_2d, alt_aligned_2d = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds_2d[0, 0],
    alt_pred=alt_preds_2d[0, 0],
    metadata=metadata[0],
    prediction_type="2D",
    bin_size=8,
    crop_length=10,
    diag_offset=2,
    matrix_size=matrix_size
)
```

## Interpreting Results

**NaN values** indicate regions affected by indels, diagonal bins (2D), or cross-pattern masking (inversions).

**Use nan-aware functions:**

```python
import numpy as np
diff = alt_aligned - ref_aligned
mean_effect = np.nanmean(np.abs(diff))
```

## Troubleshooting

**Shape mismatch**: Check `bin_size`, `crop_length`, `matrix_size` (2D), and that metadata matches predictions.

**All NaN output**: Incorrect `crop_length` or `bin_size`, or variant outside window.

**Unexpected masking**: Inversions show cross-pattern (2D), indels show NaN stripes.

## See Also

- [Prediction Alignment Examples](prediction_alignment_examples.md) - Visual examples for all variant types (INS, DEL, DUP, INV, BND)
- [Notebook: Prediction Alignment](../notebooks/03_prediction_alignment.ipynb)
- [`get_alt_ref_sequences()`](variant_centered_sequences.md)
