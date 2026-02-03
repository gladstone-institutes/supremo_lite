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

**Returns:** `(ref_seq, alt_seqs, metadata)` where metadata contains columns `['chrom', 'window_start', 'window_end', 'variant_offset0', 'ref', 'alt']`

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

**Returns:** `(ref_seq, alt_seqs, metadata)` where metadata contains columns `['chrom', 'window_start', 'window_end', 'variant_offset0', 'ref', 'alt']`

### get_scrambled_subsequences() - Control Sequence Generation

Generate negative control sequences by scrambling BED-defined regions using k-mer shuffling.

```python
ref_seqs, scrambled_seqs, metadata = sl.get_scrambled_subsequences(
    chrom='chr1',
    seq_len=200,
    reference_fasta=reference,
    bed_regions='regulatory_regions.bed',
    n_scrambles=5,          # 5 scrambled versions per region
    kmer_size=1,            # Size of k-mers to shuffle (see below)
    random_state=42         # For reproducibility
)
```

**K-mer Size Options:**

The `kmer_size` parameter controls what level of sequence composition is preserved:

| kmer_size | Shuffle Unit | Preserves |
|-----------|--------------|-----------|
| 1 (default) | Individual nucleotides | Nucleotide composition |
| 2 | Dinucleotides (2-mers) | Dinucleotide frequencies |
| 3 | Trinucleotides (3-mers) | Trinucleotide frequencies |

**Example: Different k-mer sizes**

```python
# Nucleotide shuffle (default) - preserves GC content
ref, scrambled, meta = sl.get_scrambled_subsequences(
    'chr1', 200, reference, bed_regions=bed_df, kmer_size=1
)

# Dinucleotide shuffle - preserves dinucleotide frequencies
ref, scrambled, meta = sl.get_scrambled_subsequences(
    'chr1', 200, reference, bed_regions=bed_df, kmer_size=2
)

# Trinucleotide shuffle - preserves trinucleotide frequencies
ref, scrambled, meta = sl.get_scrambled_subsequences(
    'chr1', 200, reference, bed_regions=bed_df, kmer_size=3
)
```

**Empty Chromosome Handling:**

If no BED regions match the specified chromosome, the function returns the original unshuffled sequence (centered on the chromosome) with a warning. The metadata will have `scramble_start=0` and `scramble_end=0` to indicate no scrambling occurred.

**Returns:** `(ref_seqs, scrambled_seqs, metadata)` where:
- `ref_seqs`: shape (N, 4, seq_len) - one reference per BED region
- `scrambled_seqs`: shape (N × n_scrambles, 4, seq_len)
- `metadata` columns: `['chrom', 'window_start', 'window_end', 'scramble_start', 'scramble_end', 'scramble_idx', 'ref', 'alt']`

## Metadata Columns

### Mutagenesis Metadata

- `chrom`: Chromosome name
- `window_start`: Start position of the sequence window (0-based)
- `window_end`: End position of the sequence window (0-based, exclusive)
- `variant_offset0`: Offset of the mutation within the sequence window (0-based, relative to window_start)
- `ref`: Reference nucleotide
- `alt`: Alternate nucleotide

### Scrambled Sequences Metadata

- `chrom`: Chromosome name
- `window_start`, `window_end`: Sequence window boundaries (0-based)
- `scramble_start`, `scramble_end`: Region within window that was scrambled (0-based). If both are 0, no scrambling occurred (e.g., no BED regions matched the chromosome)
- `scramble_idx`: Index of this scramble (0 to n_scrambles-1)
- `ref`: Original/reference sequence in scrambled region
- `alt`: Scrambled/alternate sequence (same nucleotide composition, or identical to ref if no scrambling)
