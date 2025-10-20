# Saturation Mutagenesis

Systematically mutate every position in a genomic region.

## Overview

Generate all possible single-nucleotide mutations for identifying functional elements, regulatory motifs, and predicting variant effects.

## Functions

### get_sm_sequences() - Regional Mutagenesis

Mutate every position in a genomic region.

```python
ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
    chrom='chr1',
    start=1000,
    end=1100,  # 100 bp region
    reference_fasta=reference
)
```

Generates `(end - start) × 3` mutations. Example: 100 bp → 300 mutations.

**Returns:** `(ref_seq, alt_seqs, metadata)` where metadata contains columns `['chrom', 'start', 'end', 'offset', 'ref', 'alt']`

### get_sm_subsequences() - Targeted Mutagenesis

Mutate only around a specific anchor point.

```python
ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    anchor=1050,        # Center position
    anchor_radius=10,   # ±10 bp
    seq_len=200,        # Total sequence length
    reference_fasta=reference
)
```

Generates `(2 × radius + 1) × 3` mutations. Example: radius=10 → 63 mutations.

Mutates positions from `anchor-radius` to `anchor+radius` within a `seq_len` context window.

**Returns:** `(ref_seq, alt_seqs, metadata)`

## Metadata

`offset`: Position within region (0-based from `start` or within `seq_len` window)

## Examples

### Basic Regional Mutagenesis

```python
import supremo_lite as sl
from pyfaidx import Fasta

reference = Fasta('reference.fa')

# Mutate 50 bp region
ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
    chrom='chr1',
    start=100,
    end=150,
    reference_fasta=reference
)

print(f"Reference shape: {ref_seq.shape}")    # (50, 4)
print(f"Alternatives shape: {alt_seqs.shape}") # (150, 50, 4)
print(f"Mutations: {len(metadata)}")          # 150 (50 × 3)
```

### Model Predictions and Effect Calculation

```python
from supremo_lite.mock_models import TestModel

model = TestModel(n_targets=1, bin_size=1, crop_length=0)
ref_pred = model(ref_seq.unsqueeze(0))
alt_preds = model(alt_seqs)

# Calculate effects
effects = []
for i, row in metadata.iterrows():
    effect = alt_preds[i, 0, row['offset']].item() - ref_pred[0, 0, row['offset']].item()
    effects.append({'position': row['start'] + row['offset'], 'effect': effect})
```

### Targeted Mutagenesis

```python
# Mutate around specific site
ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    anchor=1050,        # Center position
    anchor_radius=5,    # ±5 bp
    seq_len=100,        # Context window
    reference_fasta=reference
)
```

## Which Function?

**Use `get_sm_sequences()`** for comprehensive regional screening (entire region, <200 bp)

**Use `get_sm_subsequences()`** for focused analysis around known sites (lower computational cost, with context for models)

## See Also

- [Notebook: Saturation Mutagenesis](../notebooks/05_saturation_mutagenesis.ipynb)
- [`get_alt_ref_sequences()`](variant_centered_sequences.md)
- [`align_predictions_by_coordinate()`](prediction_alignment.md)
