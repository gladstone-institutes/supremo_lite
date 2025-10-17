# Saturation Mutagenesis

This guide covers in-silico saturation mutagenesis - systematically mutating every position in a genomic region to identify functional elements and predict variant effects.

## Overview

Saturation mutagenesis generates all possible single-nucleotide mutations at every position in a region. This is powerful for:
- Identifying functionally important positions
- Discovering regulatory motifs
- Predicting effects of any possible SNV
- Quantifying position-specific mutation sensitivity
- Finding CRISPR target sites

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

**Generates**: `(end - start) × 3` mutations
- 3 alternative nucleotides per position
- Example: 100 bp → 300 mutations

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

**Generates**: `(2 × radius + 1) × 3` mutations
- Mutate positions from `anchor-radius` to `anchor+radius`
- Longer sequence provides context
- Example: radius=10 → 21 positions × 3 = 63 mutations

## Function Signatures

### get_sm_sequences()

```python
get_sm_sequences(
    chrom,              # str: Chromosome name
    start,              # int: Region start (0-based)
    end,                # int: Region end (0-based exclusive)
    reference_fasta,    # Fasta object or path
    encoder=None        # Optional custom encoder
) -> tuple[tensor/array, tensor/array, DataFrame]
```

**Returns**:
- `ref_seq`: Reference sequence (encoded)
- `alt_seqs`: All mutated sequences (encoded)
- `metadata`: DataFrame with columns: `['chrom', 'start', 'end', 'offset', 'ref', 'alt']`

### get_sm_subsequences()

```python
get_sm_subsequences(
    chrom,              # str: Chromosome name
    anchor,             # int: Center position (0-based)
    anchor_radius,      # int: Mutation radius
    seq_len,            # int: Total sequence length
    reference_fasta,    # Fasta object or path
    bed_regions=None,   # Optional BED regions to restrict mutations
    encoder=None        # Optional custom encoder
) -> tuple[tensor/array, tensor/array, DataFrame]
```

**Returns**:
- `ref_seq`: Reference sequence (encoded), length = `seq_len`
- `alt_seqs`: Mutated sequences (encoded)
- `metadata`: DataFrame with mutation information

## Output Structure

### Metadata DataFrame

```python
# From get_sm_sequences()
   chrom  start   end  offset ref alt
0   chr1   1000  1100       0   A   C
1   chr1   1000  1100       0   A   G
2   chr1   1000  1100       0   A   T
3   chr1   1000  1100       1   T   A
4   chr1   1000  1100       1   T   C
...

# From get_sm_subsequences()
   chrom  anchor  start   end  offset ref alt
0   chr1    1050    950  1150      90   G   A
1   chr1    1050    950  1150      90   G   C
2   chr1    1050    950  1150      90   G   T
...
```

**offset**: Position within the region (0-based)
- For `get_sm_sequences()`: offset from `start`
- For `get_sm_subsequences()`: offset within `seq_len` window

## Examples

### Basic Regional Mutagenesis

```python
import supremo_lite as sl
from pyfaidx import Fasta

# Load reference
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

### Running Predictions

```python
from supremo_lite.mock_models import TestModel

# Initialize model
model = TestModel(n_targets=1, bin_size=1, crop_length=0)

# Add batch dimension for reference
ref_batch = ref_seq.unsqueeze(0)

# Predict
ref_pred = model(ref_batch)      # (1, 1, 50)
alt_preds = model(alt_seqs)      # (150, 1, 50)

print(f"Reference prediction: {ref_pred.shape}")
print(f"Alternate predictions: {alt_preds.shape}")
```

### Calculating Mutation Effects

```python
import numpy as np

# Calculate effect at each mutated position
effects = []
for i, row in metadata.iterrows():
    offset = row['offset']

    # Get predictions at mutated position
    ref_val = ref_pred[0, 0, offset].item()
    alt_val = alt_preds[i, 0, offset].item()

    effect = alt_val - ref_val
    effects.append({
        'position': row['start'] + offset,
        'ref': row['ref'],
        'alt': row['alt'],
        'ref_pred': ref_val,
        'alt_pred': alt_val,
        'effect': effect
    })

import pandas as pd
effects_df = pd.DataFrame(effects)
print(effects_df.head())
```

### Visualizing Effect Landscape

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create position × nucleotide matrix
positions = sorted(effects_df['position'].unique())
nucleotides = ['A', 'C', 'G', 'T']

# Get reference sequence
ref_sequence = sl.decode_seq(ref_seq)

# Build effect matrix
effect_matrix = np.zeros((len(nucleotides), len(positions)))
for i, pos in enumerate(positions):
    offset = pos - metadata['start'].iloc[0]
    ref_nt = ref_sequence[offset]

    for j, nt in enumerate(nucleotides):
        if nt == ref_nt:
            effect_matrix[j, i] = 0  # Reference
        else:
            mask = (effects_df['position'] == pos) & (effects_df['alt'] == nt)
            if mask.any():
                effect_matrix[j, i] = effects_df.loc[mask, 'effect'].values[0]

# Heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(effect_matrix,
            xticklabels=positions,
            yticklabels=nucleotides,
            cmap='RdBu_r',
            center=0)
plt.xlabel('Position')
plt.ylabel('Alternative Nucleotide')
plt.title('Saturation Mutagenesis Effect Landscape')
plt.show()
```

### Identifying High-Impact Positions

```python
# Max absolute effect per position
position_importance = effects_df.groupby('position')['effect'].apply(
    lambda x: x.abs().max()
).reset_index()
position_importance.columns = ['position', 'max_abs_effect']
position_importance = position_importance.sort_values('max_abs_effect', ascending=False)

# Top positions
print("Top 10 most important positions:")
print(position_importance.head(10))

# Critical positions (>75th percentile)
threshold = position_importance['max_abs_effect'].quantile(0.75)
critical = position_importance[position_importance['max_abs_effect'] > threshold]

print(f"\nCritical positions (effect > {threshold:.3f}):")
for _, row in critical.iterrows():
    print(f"  Position {int(row['position'])}: effect = {row['max_abs_effect']:.3f}")
```

### Targeted Mutagenesis Around Known Site

```python
# Focus on known TF binding site
tf_site_position = 1050

ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    anchor=tf_site_position,
    anchor_radius=5,    # Mutate ±5 bp
    seq_len=100,        # 100 bp context
    reference_fasta=reference
)

print(f"Mutations around position {tf_site_position}:")
print(f"  Window: {tf_site_position - 50} to {tf_site_position + 50}")
print(f"  Mutated region: {tf_site_position - 5} to {tf_site_position + 5}")
print(f"  Number of mutations: {len(metadata)}")  # (2*5+1)*3 = 33
```

### Finding Regulatory Motifs

```python
def find_motif_regions(position_importance, window_size=3, threshold_percentile=75):
    """Find consecutive high-impact positions (potential motifs)."""
    threshold = position_importance['max_abs_effect'].quantile(threshold_percentile / 100)
    sorted_pos = position_importance.sort_values('position')

    motifs = []
    current_motif = []

    for _, row in sorted_pos.iterrows():
        if row['max_abs_effect'] > threshold:
            if not current_motif or row['position'] == current_motif[-1]['position'] + 1:
                current_motif.append({
                    'position': int(row['position']),
                    'effect': row['max_abs_effect']
                })
            else:
                if len(current_motif) >= window_size:
                    motifs.append(current_motif)
                current_motif = [{
                    'position': int(row['position']),
                    'effect': row['max_abs_effect']
                }]
        else:
            if len(current_motif) >= window_size:
                motifs.append(current_motif)
            current_motif = []

    if len(current_motif) >= window_size:
        motifs.append(current_motif)

    return motifs

# Find motifs
motifs = find_motif_regions(position_importance)

for i, motif in enumerate(motifs, 1):
    positions = [m['position'] for m in motif]
    motif_start = positions[0]
    motif_end = positions[-1] + 1

    # Get sequence
    motif_seq = reference['chr1'][motif_start:motif_end].seq

    print(f"\nMotif {i}:")
    print(f"  Position: chr1:{motif_start}-{motif_end}")
    print(f"  Sequence: {motif_seq}")
    print(f"  Length: {len(positions)} bp")
```

### Restricting Mutations with BED Regions

```python
# Only mutate positions within specific regions
bed_regions = pd.DataFrame([
    {'chrom': 'chr1', 'start': 1040, 'end': 1060},  # TF binding site
    {'chrom': 'chr1', 'start': 1080, 'end': 1090},  # Another site
])

ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    anchor=1050,
    anchor_radius=50,
    seq_len=200,
    reference_fasta=reference,
    bed_regions=bed_regions  # Only mutate within these regions
)

print(f"Mutations restricted to BED regions: {len(metadata)}")
```

## Choosing the Right Approach

### Use get_sm_sequences() when:
- Exploring an entire region
- Don't know which positions are important
- Want comprehensive coverage
- Region is small (<200 bp)

```python
# Screen entire promoter
ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
    chrom='chr1',
    start=promoter_start,
    end=promoter_end,
    reference_fasta=reference
)
```

### Use get_sm_subsequences() when:
- Focusing on a known site (e.g., TF binding site)
- Need surrounding context for model
- Want to reduce computational cost
- Validating a specific hypothesis

```python
# Validate TF site importance
ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    anchor=tf_site,
    anchor_radius=10,
    seq_len=500,  # Context for model
    reference_fasta=reference
)
```

## Best Practices

### 1. Consider Computational Cost

```python
# Number of mutations = region_length × 3
# 100 bp → 300 mutations
# 1000 bp → 3000 mutations!

# For large regions, use batching
region_len = 1000
batch_size = 100

for batch_start in range(0, region_len, batch_size):
    batch_end = min(batch_start + batch_size, region_len)

    ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
        chrom='chr1',
        start=region_start + batch_start,
        end=region_start + batch_end,
        reference_fasta=reference
    )

    # Process batch...
```

### 2. Use Multiple Effect Metrics

```python
# Don't just look at magnitude
effects_df['abs_effect'] = effects_df['effect'].abs()
effects_df['relative_effect'] = effects_df['effect'] / effects_df['ref_pred']
effects_df['direction'] = np.sign(effects_df['effect'])

# Analyze by direction
print("\nMutations that increase prediction:")
print(effects_df[effects_df['direction'] > 0].nlargest(10, 'effect'))

print("\nMutations that decrease prediction:")
print(effects_df[effects_df['direction'] < 0].nsmallest(10, 'effect'))
```

### 3. Validate Findings

```python
# Cross-validate with different models
from supremo_lite.mock_models import TestModel

model1 = TestModel(n_targets=1, bin_size=1, crop_length=0)
model2 = TestModel(n_targets=1, bin_size=4, crop_length=5)

# Run both models
preds1 = model1(alt_seqs)
preds2 = model2(alt_seqs)

# Compare rankings
# Positions important in both models are more confident
```

### 4. Consider Biological Context

```python
# Filter by biological plausibility
# Example: Only consider transitions (purine↔purine, pyrimidine↔pyrimidine)
transitions = {
    'A': ['G'], 'G': ['A'],  # Purines
    'C': ['T'], 'T': ['C']   # Pyrimidines
}

transition_effects = effects_df[
    effects_df.apply(
        lambda row: row['alt'] in transitions.get(row['ref'], []),
        axis=1
    )
]

print("Transition mutation effects:")
print(transition_effects.nlargest(10, 'abs_effect'))
```

## Use Cases

### 1. Variant Effect Prediction
Predict the effect of any possible SNV in a region.

### 2. Regulatory Element Discovery
Find positions where mutations have large effects → likely functional.

### 3. CRISPR Guide Design
Identify positions to avoid when designing guides.

### 4. Model Interpretation
Understand which sequence features the model uses.

### 5. Therapeutic Target Identification
Find critical positions for drug targeting.

## Related Functions

- [`get_alt_ref_sequences()`](sequences.md) - Variant-centered windows
- [`align_predictions_by_coordinate()`](prediction_alignment.md) - Align predictions
- [Mock models](mock_models.md) - TestModel for predictions

## See Also

- **[Notebook: Saturation Mutagenesis](../notebooks/05_saturation_mutagenesis.ipynb)** - Complete workflow with visualizations
- **[API Reference](../autoapi/index.rst)** - Detailed API documentation
