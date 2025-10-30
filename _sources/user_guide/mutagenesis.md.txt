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

**Returns:** `(ref_seq, alt_seqs, metadata)` where metadata contains columns `['chrom', 'window_start', 'window_end', 'variant_pos0', 'ref', 'alt']`

### get_sm_subsequences() - Targeted Mutagenesis

Mutate specific regions using either anchor-based or BED-based approach.

**Approach 1: Anchor-based** (requires both `anchor` and `anchor_radius`):

```python
ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    seq_len=200,
    reference_fasta=reference,
    anchor=1050,        # Center position
    anchor_radius=10    # ±10 bp
)
```

Generates `(2 × radius) × 3` mutations. Example: radius=10 → 60 mutations.

**Approach 2: BED-based** (mutually exclusive with anchor/anchor_radius):

```python
ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    seq_len=200,
    reference_fasta=reference,
    bed_regions='regulatory_regions.bed'
)
```

**Returns:** `(ref_seq, alt_seqs, metadata)` where metadata contains columns `['chrom', 'window_start', 'window_end', 'variant_pos0', 'ref', 'alt']`

## Metadata Columns

- `chrom`: Chromosome name
- `window_start`: Start position of the sequence window (0-based)
- `window_end`: End position of the sequence window (0-based, exclusive)
- `variant_pos0`: Position of the mutation within the sequence window (0-based)
- `ref`: Reference nucleotide
- `alt`: Alternate nucleotide


