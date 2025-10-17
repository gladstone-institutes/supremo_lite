# Sequence Generation

This guide covers generating variant-centered sequence windows for model predictions and analysis.

## Overview

supremo_lite provides functions to extract sequence windows around variants, which is essential for:
- Running genomic model predictions
- Comparing reference vs. alternate alleles
- Analyzing local sequence context
- PAM site disruption analysis

## Main Functions

### get_alt_ref_sequences()

Generate reference and alternate sequence pairs centered on each variant.

```python
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,  # 1000 bp windows
    encode=True     # One-hot encoded
)
```

**Returns:**
- `ref_seqs`: Reference sequences (encoded or raw)
- `alt_seqs`: Alternate sequences with variant applied
- `metadata`: List of dicts with variant information

### get_pam_disrupting_personal_sequences()

Find variants that disrupt PAM sites (e.g., for CRISPR analysis).

```python
results = sl.get_pam_disrupting_personal_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    max_pam_distance=50,  # Search within 50 bp of variant
    pam_sequence="NGG"     # SpCas9 PAM
)
```

**Returns dict:**
```python
{
    'variants': DataFrame,          # PAM-disrupting variants
    'pam_intact': sequences,        # Reference sequences (PAM intact)
    'pam_disrupted': sequences      # Alternate sequences (PAM disrupted)
}
```

## Function Signatures

### get_alt_ref_sequences()

```python
get_alt_ref_sequences(
    reference_fn,           # Reference genome (path, Fasta, or dict)
    variants_fn,            # Variants (path or DataFrame)
    seq_len,                # int: Total sequence length
    encode=True,            # bool: Return encoded sequences
    chunk_size=1,           # int: Variants per chunk
    encoder=None            # Optional custom encoder
) -> tuple[array/list, array/list, list]
```

### get_pam_disrupting_personal_sequences()

```python
get_pam_disrupting_personal_sequences(
    reference_fn,           # Reference genome
    variants_fn,            # Variants
    seq_len,                # int: Sequence window length
    max_pam_distance,       # int: Max distance from variant to PAM
    pam_sequence="NGG",     # str: PAM sequence (supports IUPAC codes)
    encode=True,            # bool: Return encoded sequences
    chunk_size=1,           # int: Variants per chunk
    encoder=None            # Optional custom encoder
) -> dict
```

## Parameters

### Sequence Length (`seq_len`)

Choose based on your model's receptive field:

```python
# For short-range models (e.g., promoter analysis)
seq_len = 200

# For medium-range models (e.g., enhancer-gene links)
seq_len = 1000

# For long-range models (e.g., TAD analysis)
seq_len = 10000
```

**Variant position**: Always centered in the window (position = `seq_len // 2`)

### PAM Sequence

Supports IUPAC nucleotide codes:

```python
# Common PAM sequences
pam_sequence = "NGG"      # SpCas9
pam_sequence = "NGGNG"    # SpCas9 with extended PAM
pam_sequence = "TTTN"     # Cpf1/Cas12a
pam_sequence = "NNGRRT"   # SaCas9

# IUPAC codes supported:
# N = any base
# R = A or G (purine)
# Y = C or T (pyrimidine)
# W = A or T
# etc.
```

## Metadata Structure

Each variant returns metadata with essential information:

```python
{
    'chrom': str,              # Chromosome name
    'window_start': int,       # Window start (0-based)
    'window_end': int,         # Window end (0-based exclusive)
    'variant_pos0': int,       # Variant position (0-based)
    'variant_pos1': int,       # Variant position (1-based VCF)
    'ref': str,                # Reference allele
    'alt': str,                # Alternate allele
    'variant_type': str,       # SNV, INS, DEL, SV_INV, etc.

    # Optional fields for structural variants:
    'sym_variant_end': int,    # END position for <INV>, <DUP>

    # BND-specific fields:
    'mate_chrom': str,         # Mate chromosome (BND only)
    'mate_pos': int,           # Mate position (BND only)
    'orientation_1': str,      # Orientation (BND only)
    'orientation_2': str,      # Orientation (BND only)
    'fusion_name': str         # Fusion name (BND only)
}
```

## Output Formats

### Encoded Output (`encode=True`)

Returns numpy arrays or PyTorch tensors:

```python
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=ref,
    variants_fn=vars,
    seq_len=100,
    encode=True
)

print(ref_seqs.shape)  # (n_variants, 100, 4)
print(alt_seqs.shape)  # (n_variants, 100, 4)
```

Shape: `(n_variants, seq_len, 4)`
- Dimension 0: Variant index
- Dimension 1: Position in sequence
- Dimension 2: Nucleotide channel (A, C, G, T)

### Raw Output (`encode=False`)

Returns lists of strings:

```python
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=ref,
    variants_fn=vars,
    seq_len=100,
    encode=False
)

print(type(ref_seqs))  # list
print(len(ref_seqs))   # n_variants
print(ref_seqs[0])     # 'ATCGATCG...'
```

## Examples

### Basic Sequence Generation

```python
import supremo_lite as sl
from pyfaidx import Fasta

# Load data
reference = Fasta('reference.fa')
variants = sl.read_vcf('variants.vcf')

# Generate 500bp windows
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=500,
    encode=True
)

print(f"Generated {len(metadata)} sequence pairs")
print(f"Reference shape: {ref_seqs.shape}")
print(f"Alternate shape: {alt_seqs.shape}")
```

### Inspecting Sequences

```python
# Get raw sequences for inspection
ref_raw, alt_raw, meta = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants.iloc[:5],  # First 5 variants
    seq_len=80,
    encode=False
)

for i, m in enumerate(meta):
    print(f"\nVariant {i+1}: {m['variant_type']}")
    print(f"  {m['chrom']}:{m['variant_pos1']} {m['ref']} → {m['alt']}")
    print(f"  Ref: {ref_raw[i]}")
    print(f"  Alt: {alt_raw[i]}")
```

### PAM Disruption Analysis

```python
# Find variants that disrupt SpCas9 PAM sites
pam_results = sl.get_pam_disrupting_personal_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    max_pam_distance=50,
    pam_sequence="NGG"
)

print(f"Found {len(pam_results['variants'])} PAM-disrupting variants")
print(pam_results['variants'][['chrom', 'pos', 'ref', 'alt']])

# Access sequences
pam_intact = pam_results['pam_intact']      # Reference (PAM works)
pam_disrupted = pam_results['pam_disrupted'] # Alternate (PAM broken)
```

### Memory-Efficient Processing

```python
# For thousands of variants
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn='large_variants.vcf',
    seq_len=1000,
    chunk_size=1000,  # Process 1000 variants at a time
    encode=True
)
```

### Using Custom Encoders

```python
def custom_encoder(seq):
    # Custom encoding logic
    # Must return array of shape (len(seq), 4)
    ...
    return encoded_array

ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200,
    encoder=custom_encoder
)
```

## Reading VCF Files

### read_vcf()

Read entire VCF file into DataFrame:

```python
variants = sl.read_vcf('variants.vcf')
print(variants.columns)
# ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'vcf_line']
```

### read_vcf_chunked()

Read large VCF files in chunks (generator):

```python
for chunk in sl.read_vcf_chunked('large.vcf', chunk_size=10000):
    # Process each chunk of 10,000 variants
    print(f"Processing {len(chunk)} variants")
    ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
        reference_fn=reference,
        variants_fn=chunk,
        seq_len=1000
    )
    # ... process sequences
```

## Best Practices

### 1. Choose Appropriate Sequence Length

```python
# Consider your model's receptive field
model_receptive_field = 1000
seq_len = model_receptive_field  # Match model expectations

# For very long-range models, consider memory:
if model_receptive_field > 5000:
    # Use smaller batches or chunked processing
    chunk_size = 100
```

### 2. Validate Metadata

```python
# Always check variant types for structural variants
for meta in metadata:
    if meta['variant_type'].startswith('SV_'):
        print(f"Structural variant: {meta['variant_type']}")
        if 'sym_variant_end' in meta:
            print(f"  Spans: {meta['variant_pos1']}-{meta['sym_variant_end']}")
```

### 3. Handle Edge Cases

```python
# Check for chromosome boundary issues
for i, meta in enumerate(metadata):
    if meta['window_start'] == 0:
        print(f"Variant {i} at chromosome start")

    chrom_len = len(reference[meta['chrom']])
    if meta['window_end'] == chrom_len:
        print(f"Variant {i} at chromosome end")
```

### 4. Combine with Prediction Alignment

```python
# Generate sequences
ref_seqs, alt_seqs, metadata = sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    encode=True
)

# Run model predictions
ref_preds = model(ref_seqs)
alt_preds = model(alt_seqs)

# Align for comparison (see prediction_alignment.md)
for i in range(len(metadata)):
    ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
        ref_pred=ref_preds[i],
        alt_pred=alt_preds[i],
        metadata=metadata[i],
        ...
    )
```

## Related Functions

- [`get_personal_genome()`](personalization.md) - Full genome personalization
- [`align_predictions_by_coordinate()`](prediction_alignment.md) - Align predictions
- [Saturation mutagenesis](mutagenesis.md) - Systematic mutations

## See Also

- **[Notebook: Getting Started](../notebooks/01_getting_started.ipynb)** - Basic sequence generation
- **[Notebook: Prediction Alignment](../notebooks/03_prediction_alignment.ipynb)** - Complete workflow
- **[API Reference](../autoapi/index.rst)** - Detailed API documentation
