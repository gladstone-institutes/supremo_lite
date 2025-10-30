# PAM Disruption Analysis

Detect variants that disrupt Protospacer Adjacent Motif (PAM) sites for CRISPR genome editing analysis.

## Overview

PAM (Protospacer Adjacent Motif) sites are short DNA sequences required by CRISPR-Cas systems for target recognition. Different Cas proteins recognize different PAM sequences:

- **SpCas9**: NGG 
- **SaCas9**: NNGRRT

Variants that disrupt PAM sites can make genomic loci resistant to repeated CRISPR editing at the same site.

## Function

### get_pam_disrupting_alt_sequences()

Identifies variants that disrupt PAM sites and returns reference/alternate sequences with metadata.

## Supported PAM Sequences

The function supports [IUPAC ambiguity codes](https://en.wikipedia.org/wiki/Nucleic_acid_notation) in PAM patterns:

| PAM Pattern | CRISPR System | Description |
|-------------|---------------|-------------|
| `NGG` | SpCas9 | Most common; N = any nucleotide |
| `NNGRRT` | SaCas9 | Complex pattern with R = purine |


```python
import supremo_lite as sl

# Generator yields (alt_seqs, ref_seqs, metadata) tuples
results = list(sl.get_pam_disrupting_alt_sequences(
    reference_fn='genome.fa',
    variants_fn='variants.vcf',
    seq_len=1000,              # Sequence window size
    max_pam_distance=50,       # Max distance from variant to PAM
    pam_sequence="NGG",        # PAM pattern to search for
    encode=True                # Return one-hot encoded sequences
))

# Unpack results (single chunk in this case)
alt_seqs, ref_seqs, metadata = results[0]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_fn` | str/Fasta/dict | Required | Reference genome |
| `variants_fn` | str/DataFrame | Required | Variants to analyze |
| `seq_len` | int | Required | Length of sequence windows to extract |
| `max_pam_distance` | int | Required | Maximum distance (bp) from variant to PAM site |
| `pam_sequence` | str | `"NGG"` | PAM pattern (supports IUPAC ambiguity codes) |
| `encode` | bool | `True` | Return one-hot encoded sequences |
| `n_chunks` | int | `1` | Number of chunks for memory-efficient processing |
| `encoder` | callable | `None` | Custom encoder function |

### Return Value

Returns a generator yielding `(alt_seqs, ref_seqs, metadata)` tuples:

- **`alt_seqs`**: Encoded alternate sequences with variants applied
- **`ref_seqs`**: Encoded reference sequences without variants
- **`metadata`**: DataFrame with variant info plus PAM-specific columns:
  - `pam_site_pos`: Position of disrupted PAM site
  - `pam_ref_sequence`: PAM sequence in reference
  - `pam_alt_sequence`: PAM sequence in alternate (disrupted)
  - `pam_distance`: Distance from variant to PAM site

## Usage Example


```python
import supremo_lite as sl
from pyfaidx import Fasta

# Load data
reference = Fasta('hg38.fa')
variants = 'patient_variants.vcf'

# Analyze SpCas9 PAM disruption
results = list(sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200,
    max_pam_distance=50,
    pam_sequence="NGG",
    encode=False  # Get raw sequences for inspection
))

alt_seqs, ref_seqs, metadata = results[0]
print(f"Found {len(metadata)} PAM-disrupting variants")

# Inspect first variant
if len(metadata) > 0:
    var = metadata.iloc[0]
    print(f"\nVariant: {var['chrom']}:{var['pos1']} {var['ref']}→{var['alt']}")
    print(f"PAM disrupted at position: {var['pam_site_pos']}")
    print(f"Reference PAM: {var['pam_ref_sequence']} → Alternate: {var['pam_alt_sequence']}")
```






## Performance Considerations

### Memory Efficiency

For large variant sets, use chunking:

```python
gen = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn='large_variants.vcf',
    seq_len=1000,
    max_pam_distance=50,
    pam_sequence="NGG",
    n_chunks=10  # Process in 10 chunks
)

# Process each chunk
for alt_seqs, ref_seqs, metadata in gen:
    # Process chunk
    print(f"Processing {len(metadata)} variants")
```


## Troubleshooting

### Unexpected results for INDELs

**Remember**: INDELs that create or shift PAMs are NOT scored as disrupting.

```python
# Manually inspect INDEL effects
var = variants.iloc[0]  # An INDEL
print(f"Variant: {var['ref']} → {var['alt']}")

results = list(sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=pd.DataFrame([var]),
    seq_len=100,
    max_pam_distance=50,
    encode=False
))

alt_seqs, ref_seqs, metadata = results[0]
if len(metadata) > 0:
    print("Disrupts PAM")
else:
    print("Does NOT disrupt PAM (may create new PAM)")
```

## See Also

- [Variant-Centered Sequences](variant_centered_sequences.md) - Generate sequence windows around variants
- [Prediction Alignment](prediction_alignment.md) - Align model predictions across reference and alternate sequences
- [Getting Started Notebook](../notebooks/01_getting_started.ipynb) - Basic usage examples including PAM disruption
