# Personalized Genome Generation

Apply variants from VCF files to reference genomes.

## Overview

`get_personal_genome()` applies variants to a reference genome. Handles SNV, MNV, INS, DEL, and structural variants (INV, DUP, BND) with automatic chromosome name matching and memory-efficient chunked processing.

## Basic Usage

```python
import supremo_lite as sl
from pyfaidx import Fasta

# Load data
reference = Fasta('reference.fa')
variants = sl.read_vcf('variants.vcf')

# Generate personalized genome
personal_genome = sl.get_personal_genome(
    reference_fn=reference,
    variants_fn=variants,
    encode=True,  # Returns encoded sequences
    verbose=True  # Show progress
)
```

## Function Signature

```python
get_personal_genome(
    reference_fn,           # str, Fasta object, or dict
    variants_fn,            # str (path) or DataFrame
    encode=True,            # bool: return encoded (True) or raw strings (False)
    chunk_size=1,           # int: variants per chunk (1 = no chunking)
    verbose=False,          # bool: show progress and skip information
    encoder=None            # optional custom encoding function
) -> dict
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_fn` | str, Fasta, or dict | required | Reference genome (file path, Fasta object, or dict) |
| `variants_fn` | str or DataFrame | required | Variants (VCF file path or DataFrame from `read_vcf()`) |
| `encode` | bool | True | True: one-hot encoded arrays; False: raw strings |
| `chunk_size` | int | 1 | Variants per chunk (use 10k-50k for large VCFs) |
| `verbose` | bool | False | Show progress and skip information |
| `encoder` | function | None | Optional custom encoding function |

## Output

Returns dict mapping chromosome names to sequences (order matches reference):

```python
{'chr1': <sequence>, 'chr2': <sequence>, ...}
```

## Variant Handling

### Supported Variant Types

| Type | Description | Sequence Effect |
|------|-------------|-----------------|
| SNV | Single nucleotide variant | Single base substitution |
| MNV | Multiple nucleotide variant | Multiple base substitution |
| INS | Insertion | Sequence lengthens |
| DEL | Deletion | Sequence shortens |
| INV | Inversion | Reverse complement |
| DUP | Duplication | Tandem repeat (length increases) |
| BND | Breakend/Translocation | Fusion sequence created |

### Overlapping Variants

First variant in VCF is applied; overlapping variants are skipped (reported in verbose mode).

### Skipped Variants

Variants skipped if overlapping, unsupported type, validation errors, or missing INFO fields. Use `verbose=True` to see details.

## Examples

### Basic Personalization

```python
personal_genome = sl.get_personal_genome(
    reference_fn='ref.fa',
    variants_fn='vars.vcf'
)
```

### Large-Scale Processing

```python
# For millions of variants
personal_genome = sl.get_personal_genome(
    reference_fn='/data/hg38.fa',
    variants_fn='/data/variants.vcf.gz',
    encode=False,      # Lower memory
    chunk_size=50000,  # 50k per chunk
    verbose=True
)
```

## Chromosome Name Matching

Automatically handles naming differences: chr1 ↔ 1, chrX ↔ X, chrM ↔ MT.

## See Also

- [Notebook: Personalized Genomes](../notebooks/02_personalized_genomes.ipynb)
- [`read_vcf()`](variant_centered_sequences.md#vcf-reading-functions)
- [`get_alt_ref_sequences()`](variant_centered_sequences.md)
