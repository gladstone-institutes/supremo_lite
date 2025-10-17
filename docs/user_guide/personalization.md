# Personalized Genome Generation

This guide covers creating personalized genomes by applying variants from VCF files to reference genomes.

## Overview

The `get_personal_genome()` function applies variants to a reference genome to create a personalized genome sequence. It handles:
- All variant types: SNV, MNV, INS, DEL, and structural variants (INV, DUP, BND)
- Automatic chromosome name matching
- Memory-efficient chunked processing
- Detailed progress reporting

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

### Input Formats

**reference_fn** accepts:
- File path (str): `'hg38.fa'`
- pyfaidx Fasta object: `Fasta('hg38.fa')`
- Dictionary: `{'chr1': 'ATCG...', 'chr2': 'GCTA...'}`

**variants_fn** accepts:
- VCF file path (str): `'variants.vcf'`
- pandas DataFrame: `variants_df` (from `read_vcf()`)

### Output Control

**encode** (bool):
- `True`: Returns one-hot encoded numpy arrays or PyTorch tensors
  - Shape: `(seq_length, 4)` per chromosome
  - Memory-intensive for large genomes
- `False`: Returns raw sequence strings
  - More memory-efficient
  - Easier for inspection

**chunk_size** (int):
- `1` (default): Process all variants at once
- `> 1`: Process variants in chunks
  - Reduces memory usage for large VCF files
  - Recommended: 10,000-50,000 for files with >100k variants

**verbose** (bool):
- `True`: Show processing progress, variant statistics, and skip reporting
- `False`: Silent operation

## Output Format

Returns a dictionary:
```python
{
    'chr1': <sequence>,  # Encoded array or string
    'chr2': <sequence>,
    ...
}
```

**Chromosome order**: Matches reference genome order (not variant application order)

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

When variants overlap:
1. First variant in VCF is applied
2. Subsequent overlapping variants are skipped
3. Skip reason reported in verbose mode

### Skipped Variants

Variants may be skipped due to:
- Overlap with previously applied variant
- Unsupported variant type
- Validation errors (e.g., reference mismatch)
- Missing required INFO fields

**Verbose mode** shows:
- Total variants processed
- Number skipped (with reasons)
- VCF line numbers of skipped variants
- Chromosome:position information

## Memory Optimization

### For Large Genomes

```python
# Use string output (less memory)
personal_genome = sl.get_personal_genome(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    encode=False  # Strings instead of arrays
)
```

### For Large VCF Files

```python
# Use chunked processing
personal_genome = sl.get_personal_genome(
    reference_fn='reference.fa',
    variants_fn='large_variants.vcf',
    chunk_size=25000,  # Process 25k variants at a time
    verbose=True       # Monitor progress
)
```

## Chromosome Name Matching

supremo_lite automatically handles chromosome naming differences:

| VCF | Reference | Match |
|-----|-----------|-------|
| chr1 | 1 | ✅ |
| chr2 | 2 | ✅ |
| chrX | X | ✅ |
| chrM | MT | ✅ |
| chr1 | chr1 | ✅ |

Unmatched chromosomes trigger warnings in verbose mode.

## Examples

### Basic Personalization
```python
personal_genome = sl.get_personal_genome(
    reference_fn='ref.fa',
    variants_fn='vars.vcf'
)
```

### With Progress Monitoring
```python
personal_genome = sl.get_personal_genome(
    reference_fn='ref.fa',
    variants_fn='vars.vcf',
    verbose=True
)
# Output:
# 🔄 Processing chromosome chr1: 150 variants (100 SNV, 30 INS, 20 DEL)
#   ⚠️  Skipped 5 variant(s):
#      • overlap with previous variant: VCF line(s) 145, 146 at chr1:12345
#   ✅ Applied 145/150 variants (5 skipped)
```

### Memory-Efficient Large-Scale Processing
```python
# For whole genome with millions of variants
personal_genome = sl.get_personal_genome(
    reference_fn='/data/hg38.fa',
    variants_fn='/data/population_variants.vcf.gz',
    encode=False,      # Raw strings
    chunk_size=50000,  # 50k variants per chunk
    verbose=True       # Monitor progress
)
```

## Related Functions

- [`read_vcf()`](sequences.md#reading-vcf-files) - Read VCF files
- [`get_alt_ref_sequences()`](sequences.md) - Generate variant-centered windows
- [Chromosome utilities](../autoapi/supremo_lite/chromosome_utils/index.rst) - Name matching functions

## See Also

- **[Notebook: Personalized Genomes](../notebooks/02_personalized_genomes.ipynb)** - Hands-on examples
- **[Notebook: Getting Started](../notebooks/01_getting_started.ipynb)** - Basic introduction
- **[API Reference](../autoapi/index.rst)** - Complete API documentation
