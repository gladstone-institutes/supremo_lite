# PAM Disruption Analysis

Detect variants that disrupt Protospacer Adjacent Motif (PAM) sites for CRISPR genome editing analysis.

## Overview

PAM (Protospacer Adjacent Motif) sites are short DNA sequences required by CRISPR-Cas systems for target recognition. Different Cas proteins recognize different PAM sequences:

- **SpCas9**: NGG (most common)
- **SpCas9 (extended)**: NGGNG
- **Cas12a/Cpf1**: TTTN
- **SaCas9**: NNGRRT

Variants that disrupt PAM sites can make genomic loci resistant to CRISPR editing, which is important for:
- Designing CRISPR-resistant therapeutic variants
- Evaluating off-target editing risks
- Understanding natural CRISPR resistance mechanisms

## Function

### get_pam_disrupting_alt_sequences()

Identifies variants that truly disrupt PAM sites by comparing PAM presence in reference vs. alternate sequences.

```python
import supremo_lite as sl

results = sl.get_pam_disrupting_alt_sequences(
    reference_fn='genome.fa',
    variants_fn='variants.vcf',
    seq_len=1000,              # Sequence window size
    max_pam_distance=50,       # Max distance from variant to PAM
    pam_sequence="NGG",        # PAM pattern to search for
    encode=True,               # Return one-hot encoded sequences
    n_chunks=1                 # Chunking for large variant sets
)
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

Returns a dictionary with three keys:

```python
{
    'variants': List[pd.Series],      # Variants that disrupt PAMs
    'pam_intact': List[tuple],        # (chrom, start, end, sequence) with variant but PAM intact
    'pam_disrupted': List[tuple]      # (chrom, start, end, sequence) with PAM disrupted (NNN)
}
```

- **`variants`**: List of variant records (as pandas Series) that disrupt at least one PAM site
- **`pam_intact`**: Sequences with the variant applied but PAM site still present (for comparison)
- **`pam_disrupted`**: Sequences with both variant and PAM site disrupted (replaced with NNN for masking)

Each sequence tuple contains:
- `chrom`: Chromosome name
- `start`: Window start position (0-based)
- `end`: Window end position (0-based, exclusive)
- `sequence`: Encoded sequence (if `encode=True`) or raw string (if `encode=False`)

## Key Feature: INDEL Detection

🎯 **Critical Enhancement**: The function correctly identifies when INDELs create NEW PAM sites rather than disrupting existing ones.

### Why This Matters

Consider these scenarios:

#### Scenario 1: Deletion Creates New PAM ✅ NOT Disrupting
```
Reference:  ATCG-AT-NGG-CATCG  (no PAM near variant)
Variant:    Delete "AT"
Alternate:  ATCG-NGG-CATCG     (deletion brings N and GG together → new PAM!)
```

**Result**: NOT scored as PAM-disrupting because a functional PAM still exists.

#### Scenario 2: Substitution Disrupts PAM ❌ IS Disrupting
```
Reference:  ATCG-NGG-CATCG     (PAM present)
Variant:    G→A at position 5
Alternate:  ATCG-NAG-CATCG     (PAM destroyed)
```

**Result**: Scored as PAM-disrupting because the PAM is truly lost.

#### Scenario 3: Deletion Shifts PAM ✅ NOT Disrupting
```
Reference:  ATCG-AT-NGG-CATCG  (PAM at position 7-9)
Variant:    Delete "AT" before PAM
Alternate:  ATCG-NGG-CATCG     (PAM shifted to position 5-7 but still present)
```

**Result**: NOT scored as PAM-disrupting because the PAM is maintained (albeit at a shifted position).

### Implementation Details

The function uses a sophisticated two-step process:

1. **Identify Reference PAMs**: Find all PAM sites in the reference sequence within `max_pam_distance` of the variant
2. **Check Alternate Sequence**: Apply the variant and search for PAM sites in the alternate sequence
3. **Compare Positions**: Determine if each reference PAM is truly disrupted or maintained (possibly at a shifted position)
4. **Filter Results**: Only return variants where at least one PAM is genuinely disrupted

This ensures accurate detection even with complex INDELs.

## Usage Examples

### Example 1: Basic PAM Disruption Analysis

```python
import supremo_lite as sl
from pyfaidx import Fasta

# Load data
reference = Fasta('hg38.fa')
variants = 'patient_variants.vcf'

# Analyze SpCas9 PAM disruption
results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=200,
    max_pam_distance=50,
    pam_sequence="NGG",
    encode=False  # Get raw sequences for inspection
)

print(f"Found {len(results['variants'])} PAM-disrupting variants")

# Inspect first variant
if results['variants']:
    var = results['variants'][0]
    print(f"\nVariant: {var['chrom']}:{var['pos1']} {var['ref']}→{var['alt']}")

    # Get corresponding sequences
    intact_seq = results['pam_intact'][0][3]  # (chrom, start, end, seq)
    disrupted_seq = results['pam_disrupted'][0][3]

    print(f"PAM-intact sequence:    {intact_seq}")
    print(f"PAM-disrupted sequence: {disrupted_seq}")
```

### Example 2: Multi-PAM Analysis

Screen variants against multiple CRISPR systems:

```python
pam_systems = {
    'SpCas9': 'NGG',
    'Cas12a': 'TTTN',
    'SaCas9': 'NNGRRT'
}

for system_name, pam_seq in pam_systems.items():
    results = sl.get_pam_disrupting_alt_sequences(
        reference_fn=reference,
        variants_fn=variants,
        seq_len=200,
        max_pam_distance=50,
        pam_sequence=pam_seq
    )

    print(f"{system_name}: {len(results['variants'])} PAM-disrupting variants")
```

### Example 3: Model Prediction with PAM Disruption

Combine with model predictions to assess editing outcomes:

```python
# Get PAM-disrupting variants
pam_results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    max_pam_distance=100,
    pam_sequence="NGG",
    encode=True
)

# Run model predictions on PAM-intact vs PAM-disrupted sequences
import torch

model = load_pretrained_model()

for i, (intact_seq, disrupted_seq) in enumerate(zip(
    pam_results['pam_intact'],
    pam_results['pam_disrupted']
)):
    # Extract encoded sequences
    intact_encoded = intact_seq[3]  # Already one-hot encoded
    disrupted_encoded = disrupted_seq[3]

    # Predict
    with torch.no_grad():
        pred_intact = model(intact_encoded.unsqueeze(0))
        pred_disrupted = model(disrupted_encoded.unsqueeze(0))

    # Compare predictions
    diff = (pred_intact - pred_disrupted).abs().mean()
    print(f"Variant {i}: Prediction difference = {diff:.4f}")
```

### Example 4: INDEL-Specific Analysis

Focus on INDELs to find variants that might create new PAMs:

```python
import pandas as pd

# Load variants
all_variants = sl.read_vcf('variants.vcf')

# Filter for INDELs
indels = all_variants[all_variants['variant_type'].isin(['INS', 'DEL'])]

print(f"Total INDELs: {len(indels)}")

# Check how many disrupt PAMs
pam_results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=indels,
    seq_len=200,
    max_pam_distance=50,
    pam_sequence="NGG",
    encode=False
)

disrupting_count = len(pam_results['variants'])
non_disrupting_count = len(indels) - disrupting_count

print(f"\nPAM-disrupting INDELs: {disrupting_count}")
print(f"Non-disrupting INDELs: {non_disrupting_count}")
print(f"  (These may create new PAMs or shift existing ones)")
```

## Supported PAM Sequences

The function supports IUPAC ambiguity codes in PAM patterns:

| PAM Pattern | CRISPR System | Description |
|-------------|---------------|-------------|
| `NGG` | SpCas9 | Most common; N = any nucleotide |
| `NGGNG` | SpCas9 (extended) | Extended PAM pattern |
| `TTTN` | Cas12a/Cpf1 | T-rich PAM |
| `NNGRRT` | SaCas9 | Complex pattern with R = purine |

### IUPAC Codes Supported

- `N`: Any nucleotide (A, C, G, or T)
- `R`: Purine (A or G)
- `Y`: Pyrimidine (C or T)
- `W`: Weak (A or T)
- `S`: Strong (C or G)
- `M`: Amino (A or C)
- `K`: Keto (G or T)

## Performance Considerations

### Memory Efficiency

For large variant sets, use chunking:

```python
results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn='large_variants.vcf',
    seq_len=1000,
    max_pam_distance=50,
    pam_sequence="NGG",
    n_chunks=10  # Process in 10 chunks
)
```

### Sequence Length

- **Small windows** (100-500 bp): Faster, but may miss distant PAM sites
- **Large windows** (1000-2000 bp): More comprehensive, but slower
- **Recommended**: 500-1000 bp for most analyses

### PAM Distance

- **Close range** (≤25 bp): Strict filtering, fewer candidates
- **Medium range** (25-100 bp): Balanced approach
- **Long range** (>100 bp): May include spurious associations

## Best Practices

### 1. Validate PAM Patterns

Ensure your PAM sequence matches your CRISPR system:

```python
# Verify PAM exists in reference
ref_seq = reference['chr1'][1000:1100].seq
if 'NGG' in ref_seq:
    print("PAM found in reference region")
```

### 2. Filter by Variant Type

Different variant types have different effects on PAMs:

```python
# Analyze only SNVs (most predictable)
snvs = variants[variants['variant_type'] == 'SNV']

# Analyze INDELs separately (may create new PAMs)
indels = variants[variants['variant_type'].isin(['INS', 'DEL'])]
```

### 3. Use Raw Sequences for Inspection

When debugging or validating:

```python
results = sl.get_pam_disrupting_alt_sequences(
    ...,
    encode=False  # Get raw strings for manual inspection
)

# Manually verify PAM disruption
intact_seq = results['pam_intact'][0][3]
print(f"PAM pattern in sequence: {intact_seq.count('NGG')}")
```

### 4. Combine with Prediction Alignment

For comprehensive analysis:

```python
# Get PAM-disrupting sequences
pam_results = sl.get_pam_disrupting_alt_sequences(...)

# Align predictions accounting for coordinate changes
from supremo_lite import align_predictions_by_coordinate

aligned = align_predictions_by_coordinate(
    ref_pred=ref_predictions,
    alt_pred=alt_predictions,
    metadata=variant_metadata
)
```

## Common Pitfalls

### ❌ Incorrect: Assuming all nearby PAMs are disrupted

```python
# DON'T assume all variants near PAMs disrupt them
variants_near_pam = filter_variants_by_distance(variants, pam_sites, distance=50)
```

### ✅ Correct: Use the function to determine true disruption

```python
# DO use the function to accurately identify disruption
results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=variants,
    max_pam_distance=50,
    ...
)
```

### ❌ Incorrect: Ignoring INDEL effects

```python
# DON'T treat INDELs the same as SNVs
# (INDELs can shift or create PAMs)
```

### ✅ Correct: Trust the function's INDEL detection

```python
# DO rely on the function's sophisticated INDEL handling
# It accounts for new PAM formation and positional shifts
results = sl.get_pam_disrupting_alt_sequences(...)
```

## Troubleshooting

### No variants returned

**Possible causes**:
1. No PAM sites within `max_pam_distance` of any variant
2. All variants create new PAMs (not disrupting)
3. PAM sequence pattern doesn't match reference

**Solutions**:
```python
# Increase search distance
results = sl.get_pam_disrupting_alt_sequences(..., max_pam_distance=100)

# Try different PAM patterns
for pam in ['NGG', 'NAG', 'NGA']:
    results = sl.get_pam_disrupting_alt_sequences(..., pam_sequence=pam)
    print(f"{pam}: {len(results['variants'])} variants")

# Check reference for PAM presence
ref_seq = reference['chr1'][:1000].seq
print(f"NGG count in reference: {ref_seq.count('NGG')}")
```

### Unexpected results for INDELs

**Remember**: INDELs that create or shift PAMs are NOT scored as disrupting.

```python
# Manually inspect INDEL effects
var = variants.iloc[0]  # An INDEL
print(f"Variant: {var['ref']} → {var['alt']}")

results = sl.get_pam_disrupting_alt_sequences(
    reference_fn=reference,
    variants_fn=pd.DataFrame([var]),
    seq_len=100,
    max_pam_distance=50,
    encode=False
)

if results['variants']:
    print("Disrupts PAM")
else:
    print("Does NOT disrupt PAM (may create new PAM)")
```

## See Also

- [Variant-Centered Sequences](variant_centered_sequences.md) - Generate sequence windows around variants
- [Prediction Alignment](prediction_alignment.md) - Align model predictions across reference and alternate sequences
- [Getting Started Notebook](../notebooks/01_getting_started.ipynb) - Basic usage examples including PAM disruption
