# Mock Models

This guide covers the mock genomic models included with supremo_lite for testing, demonstrations, and learning.

## Overview

supremo_lite includes production-ready mock models that simulate realistic genomic prediction architectures:
- **TestModel**: 1D genomic signal predictions (ChIP-seq, ATAC-seq, etc.)
- **TestModel2D**: 2D contact map predictions (Hi-C, Micro-C, etc.)

These models are useful for:
- 🧪 **Testing** prediction alignment workflows
- 📚 **Learning** how to use supremo_lite
- 🔍 **Prototyping** analysis pipelines
- 📊 **Demonstrating** variant effect prediction

## Models

### TestModel (1D Predictions)

Simulates models that predict genomic signals at genomic positions.

```python
from supremo_lite.mock_models import TestModel

model = TestModel(
    n_targets=2,      # Number of signals (e.g., H3K27ac, H3K4me3)
    bin_size=8,       # Prediction resolution (1 per 8 bp)
    crop_length=10    # Edge bases removed
)
```

**Architecture Features:**
- **Binning**: Predicts at lower resolution than input
- **Edge cropping**: Removes bases from edges to avoid boundary effects
- **Multiple targets**: Simultaneous prediction of multiple signals

**Input/Output:**
- Input: `(batch, seq_len, 4)` - One-hot encoded sequences
- Output: `(batch, n_targets, n_bins)` - Predictions per target per bin

### TestModel2D (Contact Map Predictions)

Simulates models that predict chromatin contacts between genomic positions.

```python
from supremo_lite.mock_models import TestModel2D

model = TestModel2D(
    n_targets=1,       # Number of contact types
    bin_size=8,        # Bin resolution
    crop_length=10,    # Edge cropping
    diag_offset=2      # Diagonal bins masked
)
```

**Architecture Features:**
- **2D output**: Pairwise contact predictions
- **Diagonal masking**: Removes self-interactions and short-range contacts
- **Flattened format**: Upper triangle flattened for efficiency
- **Binning and cropping**: Same as 1D model

**Input/Output:**
- Input: `(batch, seq_len, 4)` - One-hot encoded sequences
- Output: `(batch, n_targets, n_flattened_ut_bins)` - Flattened upper triangle

## Importing Models

```python
from supremo_lite.mock_models import TestModel, TestModel2D, TORCH_AVAILABLE

# Check if PyTorch is available
if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for mock models")

# Create models
model_1d = TestModel(n_targets=2, bin_size=8, crop_length=10)
model_2d = TestModel2D(n_targets=1, bin_size=8, crop_length=10, diag_offset=2)
```

## TestModel Parameters

### n_targets (int)
Number of different signals to predict simultaneously.

```python
# Single signal (e.g., ATAC-seq)
model = TestModel(n_targets=1, bin_size=8, crop_length=10)

# Multiple signals (e.g., different histone marks)
model = TestModel(n_targets=4, bin_size=8, crop_length=10)
# Could represent: H3K27ac, H3K4me3, H3K27me3, H3K9me3
```

### bin_size (int)
Prediction resolution in base pairs.

```python
# Per-base predictions (computationally expensive)
model = TestModel(n_targets=1, bin_size=1, crop_length=0)

# Standard binning (1 prediction per 8 bp)
model = TestModel(n_targets=1, bin_size=8, crop_length=10)

# Coarse binning (1 prediction per 32 bp)
model = TestModel(n_targets=1, bin_size=32, crop_length=10)
```

**Number of bins calculation:**
```python
n_bins = (seq_len - 2 * crop_length) // bin_size
```

### crop_length (int)
Number of bases removed from each edge before prediction.

```python
# No cropping
model = TestModel(n_targets=1, bin_size=8, crop_length=0)

# Standard cropping (removes edge effects)
model = TestModel(n_targets=1, bin_size=8, crop_length=10)

# Heavy cropping
model = TestModel(n_targets=1, bin_size=8, crop_length=50)
```

**Effective sequence length:**
```python
effective_len = seq_len - 2 * crop_length
```

## TestModel2D Additional Parameters

### diag_offset (int, default=0)
Number of diagonal bins to mask (remove self-interactions).

```python
# No diagonal masking
model = TestModel2D(n_targets=1, bin_size=8, crop_length=10, diag_offset=0)

# Mask 2 bins from diagonal (typical)
model = TestModel2D(n_targets=1, bin_size=8, crop_length=10, diag_offset=2)

# Mask 5 bins (removes more short-range contacts)
model = TestModel2D(n_targets=1, bin_size=8, crop_length=10, diag_offset=5)
```

**Diagonal masking visualization:**
```
diag_offset=0:  All bins predicted
diag_offset=1:  Diagonal masked
diag_offset=2:  Diagonal + 1 off-diagonal masked
```

## Output Formats

### 1D Output Shape

```python
model = TestModel(n_targets=2, bin_size=8, crop_length=10)
seq_len = 200

predictions = model(sequences)  # Input: (batch, 200, 4)

print(predictions.shape)
# Output: (batch, 2, 22)
# - batch: Number of sequences
# - 2: n_targets
# - 22: n_bins = (200 - 20) // 8 = 22.5 → 22
```

### 2D Output Shape (Flattened)

```python
model = TestModel2D(n_targets=1, bin_size=8, crop_length=10, diag_offset=2)
seq_len = 200

predictions = model(sequences)  # Input: (batch, 200, 4)

# Calculate expected output size
n_bins = (200 - 20) // 8  # 22
n_ut_bins = (n_bins * (n_bins - 1)) // 2  # 231 (upper triangle)
n_masked = (n_bins - 2) * 2  # 40 (diagonal offset bins)
n_pred = n_ut_bins - n_masked  # 191

print(predictions.shape)
# Output: (batch, 1, 191)
```

## Examples

### Basic 1D Prediction

```python
import supremo_lite as sl
from supremo_lite.mock_models import TestModel

# Generate sequences
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200,
    encode=True
)

# Create model
model = TestModel(n_targets=2, bin_size=8, crop_length=10)

# Run predictions
ref_preds = model(ref_seqs)
alt_preds = model(alt_seqs)

print(f"Reference predictions: {ref_preds.shape}")
print(f"Alternate predictions: {alt_preds.shape}")
```

### Basic 2D Prediction

```python
from supremo_lite.mock_models import TestModel2D

# Create 2D model
model_2d = TestModel2D(
    n_targets=1,
    bin_size=8,
    crop_length=10,
    diag_offset=2
)

# Run predictions
ref_preds_2d = model_2d(ref_seqs)
alt_preds_2d = model_2d(alt_seqs)

print(f"Contact map predictions: {ref_preds_2d.shape}")
# (batch, 1, n_flattened_bins)
```

### Complete Workflow with Alignment

```python
from supremo_lite.mock_models import TestModel

# 1. Generate sequences
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200,
    encode=True
)

# 2. Create model
model = TestModel(n_targets=1, bin_size=8, crop_length=10)

# 3. Run predictions
ref_preds = model(ref_seqs)
alt_preds = model(alt_seqs)

# 4. Align predictions for first variant
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds[0],
    alt_pred=alt_preds[0],
    metadata=metadata[0],
    prediction_type="1D",
    bin_size=model.bin_size,
    crop_length=model.crop_length
)

# 5. Calculate effect
import numpy as np
effect = np.nanmean(alt_aligned - ref_aligned)
print(f"Mean prediction change: {effect:.4f}")
```

### Visualizing 1D Predictions

```python
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy
ref_np = ref_aligned[0].cpu().numpy() if hasattr(ref_aligned, 'cpu') else ref_aligned[0]
alt_np = alt_aligned[0].cpu().numpy() if hasattr(alt_aligned, 'cpu') else alt_aligned[0]

# Plot
plt.figure(figsize=(14, 5))
positions = np.arange(len(ref_np))

plt.plot(positions, ref_np, 'o-', label='Reference', linewidth=2)
plt.plot(positions, alt_np, 's-', label='Alternate', linewidth=2)

# Mark variant position
variant_bin = (metadata[0]['variant_pos0'] - metadata[0]['window_start'] - model.crop_length) // model.bin_size
plt.axvline(x=variant_bin, color='red', linestyle='--', label='Variant')

plt.xlabel('Bin position')
plt.ylabel('Prediction value')
plt.legend()
plt.title('1D Prediction Comparison')
plt.show()
```

### Visualizing 2D Contact Maps

```python
# 2D predictions (already aligned)
ref_2d = ref_aligned_2d.cpu().numpy() if hasattr(ref_aligned_2d, 'cpu') else ref_aligned_2d
alt_2d = alt_aligned_2d.cpu().numpy() if hasattr(alt_aligned_2d, 'cpu') else alt_aligned_2d

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Reference
axes[0].imshow(ref_2d, cmap='Reds', vmin=0, vmax=1)
axes[0].set_title('Reference Contact Map')

# Alternate
axes[1].imshow(alt_2d, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Alternate Contact Map')

# Difference
diff = alt_2d - ref_2d
axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[2].set_title('Difference (Alt - Ref)')

plt.tight_layout()
plt.show()
```

### Understanding Model Outputs

```python
# Demonstrate binning and cropping
model = TestModel(n_targets=1, bin_size=8, crop_length=10)
seq_len = 200

print(f"Model configuration:")
print(f"  Input sequence length: {seq_len}")
print(f"  Crop length: {model.crop_length}")
print(f"  Effective length: {seq_len - 2*model.crop_length}")
print(f"  Bin size: {model.bin_size}")
print(f"  Number of bins: {(seq_len - 2*model.crop_length) // model.bin_size}")

# Create dummy input
import torch
dummy_input = torch.randn(1, seq_len, 4)
output = model(dummy_input)

print(f"\nActual output shape: {output.shape}")
print(f"  Batch: {output.shape[0]}")
print(f"  Targets: {output.shape[1]}")
print(f"  Bins: {output.shape[2]}")
```

## Model Behavior

### Prediction Values

Both models generate **random predictions** (not based on sequence content):
- Values between 0 and 1
- Different each time the model is created
- Consistent for the same model instance

**Note**: These are mock models for demonstration, not trained models!

```python
# Predictions are random
model = TestModel(n_targets=1, bin_size=8, crop_length=0)
pred1 = model(sequences)
pred2 = model(sequences)

# Same model, same predictions
assert torch.allclose(pred1, pred2)

# Different model, different predictions
model2 = TestModel(n_targets=1, bin_size=8, crop_length=0)
pred3 = model2(sequences)
assert not torch.allclose(pred1, pred3)
```

### PyTorch Compatibility

Models require PyTorch:

```python
from supremo_lite.mock_models import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    print("PyTorch not available. Install with: pip install torch")
else:
    from supremo_lite.mock_models import TestModel, TestModel2D
    # Use models...
```

## Design Principles

### Why These Models Exist

1. **Realistic Architecture**: Models reflect real genomic prediction models
   - Binning (computational efficiency)
   - Cropping (edge effects)
   - Diagonal masking (uninformative interactions)

2. **Testing Infrastructure**: Validate prediction alignment works with realistic outputs
   - Different output shapes
   - Flattened 2D format
   - Various bin sizes and crop lengths

3. **User Education**: Demonstrate complete workflows without requiring:
   - Large genomic datasets
   - Trained models
   - GPU resources

### Code Adapts to Models (Not Vice Versa)

The models define the interface; `align_predictions_by_coordinate()` adapts:
- Handles any bin_size
- Handles any crop_length
- Handles flattened or full 2D matrices
- Handles any diag_offset

**This ensures supremo_lite works with real-world model architectures!**

## When to Use Mock Models

### ✅ Use Mock Models For:
- Learning how to use supremo_lite
- Testing analysis pipelines
- Demonstrating workflows
- Prototyping without trained models
- Teaching genomic analysis concepts

### ❌ Don't Use Mock Models For:
- Real variant effect prediction (use trained models)
- Publication-quality analyses
- Clinical applications
- Biological discovery

## Replacing with Real Models

To use your own trained model:

```python
# Replace mock model with your trained model
from your_package import YourModel

# Must have similar interface:
# - Input: (batch, seq_len, 4)
# - Output: (batch, n_targets, n_bins) for 1D
#           (batch, n_targets, n_flattened) for 2D

model = YourModel(...)
ref_preds = model(ref_seqs)
alt_preds = model(alt_seqs)

# Rest of workflow unchanged!
ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds[0],
    alt_pred=alt_preds[0],
    metadata=metadata[0],
    prediction_type="1D",  # or "2D"
    bin_size=model.bin_size,
    crop_length=model.crop_length,
    diag_offset=model.diag_offset  # if 2D
)
```

## See Also

- **[Notebook: Prediction Alignment](../notebooks/03_prediction_alignment.ipynb)** - Complete workflow with mock models
- **[Notebook: Structural Variants](../notebooks/04_structural_variants.ipynb)** - Advanced examples
- **[Prediction Alignment Guide](prediction_alignment.md)** - Using models with alignment
- **[API Reference](../autoapi/index.rst)** - Detailed documentation
