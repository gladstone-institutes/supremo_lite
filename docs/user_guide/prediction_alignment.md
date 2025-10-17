# Prediction Alignment

This guide covers aligning model predictions across reference and alternate sequences to account for coordinate changes caused by variants.

## Overview

When variants (especially indels and structural variants) change sequence length or coordinates, direct prediction comparison becomes invalid. The `align_predictions_by_coordinate()` function solves this by:
- Inserting NaN bins where sequences differ in length
- Applying variant-specific masking strategies
- Maintaining genomic coordinate correspondence

## Why Alignment is Necessary

### The Problem

```python
# Reference sequence: 100 bp
# Alternate sequence: 110 bp (10 bp insertion)

ref_predictions: [p1, p2, p3, ..., p10]  # 10 bins
alt_predictions: [p1, p2, p3, ..., p11]  # 11 bins

# Which bins correspond to the same genomic positions?
# → Need alignment!
```

### The Solution

```python
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_predictions,
    alt_pred=alt_predictions,
    metadata=variant_metadata,
    ...
)
# Both now have same shape with NaN masking
# → Can compare directly!
```

## Function Signature

```python
align_predictions_by_coordinate(
    ref_pred,               # Reference predictions (array/tensor)
    alt_pred,               # Alternate predictions (array/tensor)
    metadata,               # Variant metadata dict
    prediction_type,        # "1D" or "2D"
    bin_size,               # int: Model bin size
    crop_length=0,          # int: Model crop length (default 0)
    diag_offset=0,          # int: Diagonal offset for 2D (default 0)
    matrix_size=None        # int: Required for 2D predictions
) -> tuple[array/tensor, array/tensor]
```

## Parameters

### Required Parameters

**ref_pred, alt_pred** (array or tensor):
- 1D predictions: Shape `(n_targets, n_bins)` or `(n_bins,)`
- 2D predictions: Shape `(n_flattened_bins,)` or `(n_bins, n_bins)`

**metadata** (dict):
- Must contain: `variant_pos0`, `window_start`, `window_end`, `ref`, `alt`, `variant_type`
- From `get_alt_ref_sequences()` or `get_personal_sequences()`

**prediction_type** (str):
- `"1D"`: Genomic signal predictions (ChIP-seq, ATAC-seq, etc.)
- `"2D"`: Contact map predictions (Hi-C, Micro-C, etc.)

**bin_size** (int):
- Model's binning resolution (predictions per base)
- Example: `bin_size=8` → 1 prediction per 8 bp

### Optional Parameters

**crop_length** (int, default=0):
- Number of bases cropped from each sequence edge
- Used by models to avoid edge effects
- Example: `crop_length=10` → 10 bp removed from each end

**diag_offset** (int, default=0, 2D only):
- Number of diagonal bins masked in 2D models
- Removes self-interactions and short-range contacts
- Example: `diag_offset=2` → mask 2 bins from diagonal

**matrix_size** (int, required for 2D):
- Size of the square contact matrix
- Calculate as: `(seq_len - 2*crop_length) // bin_size`

## Variant-Specific Alignment Strategies

### SNV (Single Nucleotide Variant)

**Strategy**: No alignment needed
- Reference and alternate have same length
- Predictions already correspond

```python
# SNV: Direct comparison works
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_pred,
    alt_pred=alt_pred,
    metadata=metadata,  # variant_type = 'SNV'
    prediction_type="1D",
    bin_size=8
)
# ref_aligned == ref_pred
# alt_aligned == alt_pred
```

### INS (Insertion)

**Strategy**: Mask inserted bins in alternate
- Reference is shorter → reference bins padded with NaN
- Maintains coordinate correspondence

```python
# 10 bp insertion
# ref bins: valid everywhere
# alt bins: NaN where insertion occurred
```

### DEL (Deletion)

**Strategy**: Mask deleted bins in reference
- Alternate is shorter → alternate bins padded with NaN
- Shows where reference positions no longer exist in alternate

```python
# 10 bp deletion
# ref bins: NaN where deletion occurred
# alt bins: valid everywhere
```

### DUP (Duplication)

**Strategy**: Same as INS (duplication adds sequence)
- Reference lacks duplicated region → NaN padding
- Alternate contains tandem repeat

### INV (Inversion)

**Strategy**:
- **1D**: Mask inverted region bins in both ref and alt
- **2D**: **Cross-pattern masking** - mask entire rows AND columns at inversion

```python
# Inversion: chr1:100-200 (100 bp)
# 1D: Bins covering positions 100-200 masked in both

# 2D: Cross-pattern (critical!)
# - Rows at inverted positions: masked
# - Columns at inverted positions: masked
# → Creates characteristic cross pattern in contact map
```

### BND (Breakend/Translocation)

**Strategy**: Chimeric reference comparison
- Reference assembled from two loci
- Shows interaction at fusion junction

## Model Architecture Considerations

### Binning

Models often predict at lower resolution than input:

```python
# Example model:
# - Input: 1000 bp sequence
# - Bin size: 8 bp
# - Output: 125 bins (1000 / 8)

bin_size = 8
n_bins = seq_len // bin_size
```

### Edge Cropping

Models may crop edges to avoid boundary effects:

```python
# Example model:
# - Input: 1000 bp
# - Crop: 10 bp from each edge
# - Effective length: 980 bp (1000 - 20)
# - With bin_size=8: 122 bins (980 / 8)

crop_length = 10
effective_length = seq_len - 2 * crop_length
n_bins = effective_length // bin_size
```

### Diagonal Masking (2D only)

Contact map models often mask diagonal bins:

```python
# Example 2D model:
# - diag_offset=2 → mask 2 bins from diagonal
# - Removes self-interactions
# - Reduces prediction space

# Upper triangle bins with offset:
n_bins = 100
n_ut_bins = (n_bins * (n_bins - 1)) // 2  # All upper triangle
n_masked = (n_bins - diag_offset) * diag_offset  # Diagonal bins
n_pred_bins = n_ut_bins - n_masked  # Final prediction count
```

## Examples

### Basic 1D Alignment

```python
import supremo_lite as sl

# Generate sequences
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200
)

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

print(f"Aligned shape: {ref_aligned.shape}")  # Both same shape
```

### 2D Contact Map Alignment

```python
from supremo_lite.mock_models import TestModel2D

# 2D model
model_2d = TestModel2D(
    n_targets=1,
    bin_size=8,
    crop_length=10,
    diag_offset=2
)

# Predictions
ref_preds_2d = model_2d(ref_seqs)
alt_preds_2d = model_2d(alt_seqs)

# Calculate matrix size
seq_len = 200
effective_len = seq_len - 2 * 10  # crop_length
n_bins = effective_len // 8        # bin_size
matrix_size = n_bins              # 22

# Align
ref_aligned_2d, alt_aligned_2d = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds_2d[0, 0],   # First variant, first target
    alt_pred=alt_preds_2d[0, 0],
    metadata=metadata[0],
    prediction_type="2D",
    bin_size=8,
    crop_length=10,
    diag_offset=2,
    matrix_size=matrix_size
)

print(f"2D aligned shape: {ref_aligned_2d.shape}")  # (22, 22)
```

### Handling Inversions

```python
# Load inversion variant
inv_variants = sl.read_vcf('inversions.vcf')
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=inv_variants,
    seq_len=200
)

# 2D predictions
ref_preds = model_2d(ref_seqs)
alt_preds = model_2d(alt_seqs)

# Align with cross-pattern masking
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds[0, 0],
    alt_pred=alt_preds[0, 0],
    metadata=metadata[0],  # variant_type = 'SV_INV'
    prediction_type="2D",
    bin_size=8,
    crop_length=10,
    diag_offset=2,
    matrix_size=22
)

# Visualize cross-pattern
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(ref_aligned, cmap='Reds')
plt.title('Reference')
plt.subplot(1, 2, 2)
plt.imshow(alt_aligned, cmap='Blues')
plt.title('Alternate (cross-pattern masked)')
plt.show()
```

### Processing Multiple Variants

```python
# Align all variants
aligned_pairs = []

for i in range(len(metadata)):
    ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
        ref_pred=ref_preds[i],
        alt_pred=alt_preds[i],
        metadata=metadata[i],
        prediction_type="1D",
        bin_size=model.bin_size,
        crop_length=model.crop_length
    )
    aligned_pairs.append((ref_aligned, alt_aligned))

# Calculate effects
import numpy as np
effects = []
for ref, alt in aligned_pairs:
    # Use nanmean to ignore NaN values
    diff = alt - ref
    effect = np.nanmean(np.abs(diff))
    effects.append(effect)

print(f"Mean absolute effect: {np.mean(effects):.4f}")
```

## Interpreting Aligned Predictions

### Understanding NaN Values

```python
# NaN values indicate:
# 1. Regions affected by indels
# 2. Diagonal bins (2D models with diag_offset)
# 3. Cross-pattern masking (inversions in 2D)

# Use numpy masked arrays or nanfunctions
import numpy as np

# Calculate difference (ignoring NaN)
diff = alt_aligned - ref_aligned
valid_diff = diff[~np.isnan(diff)]
mean_effect = np.nanmean(np.abs(diff))
```

### Visualizing Differences

```python
import matplotlib.pyplot as plt
import numpy as np

# 1D visualization
plt.figure(figsize=(14, 5))
positions = np.arange(len(ref_aligned[0]))

plt.plot(positions, ref_aligned[0], 'o-', label='Reference')
plt.plot(positions, alt_aligned[0], 's-', label='Alternate')
plt.axvline(x=variant_bin, color='red', linestyle='--', label='Variant')
plt.xlabel('Bin position')
plt.ylabel('Prediction')
plt.legend()
plt.show()

# 2D visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(ref_aligned, cmap='Reds')
axes[0].set_title('Reference')
axes[1].imshow(alt_aligned, cmap='Blues')
axes[1].set_title('Alternate')

diff = alt_aligned - ref_aligned
axes[2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
axes[2].set_title('Difference')
plt.show()
```

## Best Practices

### 1. Always Provide Correct Model Parameters

```python
# Match your model's architecture
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ...,
    bin_size=model.bin_size,        # Model's binning
    crop_length=model.crop_length,  # Model's cropping
    diag_offset=model.diag_offset   # Model's diagonal masking (2D)
)
```

### 2. Check Variant Types

```python
# Different variants need different handling
if metadata['variant_type'] == 'SV_INV':
    # Expect cross-pattern in 2D
    prediction_type = "2D"
elif metadata['variant_type'] in ['INS', 'DEL']:
    # Expect NaN regions
    pass
```

### 3. Handle NaN Values Properly

```python
# Use nan-aware functions
effect = np.nanmean(alt_aligned - ref_aligned)
max_effect = np.nanmax(np.abs(alt_aligned - ref_aligned))

# Or use masked arrays
import numpy.ma as ma
diff = alt_aligned - ref_aligned
masked_diff = ma.masked_invalid(diff)
mean_effect = masked_diff.mean()
```

### 4. Validate Alignment Visually

```python
# Always visualize for structural variants
import matplotlib.pyplot as plt

if metadata['variant_type'].startswith('SV_'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ref_aligned)
    plt.title('Reference')
    plt.subplot(1, 2, 2)
    plt.imshow(alt_aligned)
    plt.title('Alternate')
    plt.show()
```

## Troubleshooting

### Issue: Shape Mismatch

```python
# Error: ref_aligned and alt_aligned have different shapes

# Check:
# 1. Provided correct bin_size
# 2. Provided correct crop_length
# 3. For 2D: provided correct matrix_size
# 4. Metadata is from same variant as predictions
```

### Issue: All NaN Output

```python
# Possible causes:
# 1. Incorrect crop_length (too large)
# 2. Incorrect bin_size
# 3. Metadata variant_pos0 outside window

# Debug:
print(f"Window: {metadata['window_start']}-{metadata['window_end']}")
print(f"Variant: {metadata['variant_pos0']}")
print(f"Effective start: {metadata['window_start'] + crop_length}")
```

### Issue: Unexpected Masking Pattern

```python
# For inversions, expect cross-pattern in 2D
# For indels, expect NaN stripes

# Visualize to debug:
import matplotlib.pyplot as plt
plt.imshow(np.isnan(alt_aligned).astype(float))
plt.title('NaN mask (1=NaN, 0=valid)')
plt.colorbar()
plt.show()
```

## Related Functions

- [`get_alt_ref_sequences()`](sequences.md) - Generate sequence pairs
- [Mock models](mock_models.md) - TestModel and TestModel2D
- [`get_personal_genome()`](personalization.md) - Full genome personalization

## See Also

- **[Notebook: Prediction Alignment](../notebooks/03_prediction_alignment.ipynb)** - Complete workflow with visualizations
- **[Notebook: Structural Variants](../notebooks/04_structural_variants.ipynb)** - Advanced SV handling
- **[API Reference](../autoapi/index.rst)** - Detailed API documentation
