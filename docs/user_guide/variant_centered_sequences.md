# Variant-Centered Sequence Windows

Generate sequence windows around variants for model predictions.

## Overview

Extract sequence windows centered on variants for genomic model predictions, variant effect analysis, and PAM site disruption analysis.

## Main Functions

### get_alt_ref_sequences()

Generate reference and alternate sequence pairs centered on each variant.

```python
# Note: get_alt_ref_sequences is a generator that yields chunks
results = list(sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,  # 1000 bp windows
    encode=True     # One-hot encoded
))
# Unpack from the first chunk
alt_seqs, ref_seqs, metadata = results[0]
```

**Parameters:**
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

**Returns:** `(ref_seqs, alt_seqs, metadata)`
- Variants centered at position `seq_len // 2`
- `encode=True`: shape `(n_variants, seq_len, 4)`
- `encode=False`: list of strings

### get_pam_disrupting_alt_sequences()

Identify variants that disrupt PAM sites (e.g., for CRISPR analysis).

**IMPORTANT**: This function correctly handles INDELs that might CREATE new PAM sites - these are NOT scored as PAM-disrupting since the PAM remains functional.

```python
results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    max_pam_distance=50,  # Search within 50 bp of variant
    pam_sequence="NGG"     # SpCas9 PAM
)
```

**Parameters:**
```python
get_pam_disrupting_alt_sequences(
    reference_fn,           # Reference genome
    variants_fn,            # Variants
    seq_len,                # int: Sequence window length
    max_pam_distance,       # int: Max distance from variant to PAM
    pam_sequence="NGG",     # str: PAM sequence (supports IUPAC codes)
    encode=True,            # bool: Return encoded sequences
    n_chunks=1,             # int: Number of chunks for processing
    encoder=None            # Optional custom encoder
) -> dict
```

**Returns dict:** `{'variants': DataFrame, 'pam_intact': sequences, 'pam_disrupted': sequences}`

**Supported PAM sequences:** `"NGG"` (SpCas9), `"NGGNG"` (SpCas9 extended), `"TTTN"` (Cpf1/Cas12a), `"NNGRRT"` (SaCas9)

**Key Feature**: Detects when INDELs create new PAMs and correctly excludes them from the disruption list.

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
    'mate_chrom': str,         # Mate chromosome (BND only)
    'mate_pos': int,           # Mate position (BND only)
    'orientation_1': str,      # Orientation (BND only)
    'orientation_2': str,      # Orientation (BND only)
    'fusion_name': str         # Fusion name (BND only)
}
```

:::{tip}
The `variant_type` field is automatically determined using the classification logic shown in the [Variant Classification Flow Chart](../_static/images/variant_classification.png).
:::

## Output Formats

**Encoded** (`encode=True`): numpy arrays or PyTorch tensors, shape `(n_variants, seq_len, 4)`

**Raw** (`encode=False`): list of strings

## Examples

### Basic Sequence Generation

```python
import supremo_lite as sl
from pyfaidx import Fasta

# File locations
reference = 'reference.fa'
variants = 'variants.vcf'

# Generate 500bp windows
results = list(sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=500,
    encode=True
))
alt_seqs, ref_seqs, metadata = results[0]

print(f"Generated {len(metadata)} sequence pairs")
print(f"Reference shape: {ref_seqs.shape}")
print(f"Alternate shape: {alt_seqs.shape}")
```

### PAM Disruption Analysis

```python
pam_results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    max_pam_distance=50,
    pam_sequence="NGG"
)
```


## See Also

- [Notebook: Getting Started](../notebooks/01_getting_started.ipynb)
- [`get_personal_genome()`](personalization.md)
- [`align_predictions_by_coordinate()`](prediction_alignment.md)
