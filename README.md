# supremo_lite

A lightweight memory first, model agnostic version of [SuPreMo](https://github.com/ketringjoni/SuPreMo).

## Installation

```bash
$ git clone https://github.com/gladstone-institutes/supremo_lite.git
$ cd supremo_lite
$ pip install .
```

## Usage

`supremo_lite` provides functionality for generating personalized genome sequences from reference genomes and variant files, as well as performing in-silico mutagenesis. The package supports both PyTorch tensors and NumPy arrays for sequence encoding.

### Basic Imports

```python
import supremo_lite as sl
from pyfaidx import Fasta
import pandas as pd
```

### 1. Personalized Genome Generation

Create a complete personalized genome by applying variants to a reference genome:

```python
# Load reference genome and variants
reference = Fasta('hg38.fa')
variants_df = sl.read_vcf('variants.vcf')

# Generate personalized genome (returns encoded sequences by default)
personal_genome = sl.get_personal_genome(
    reference_fn=reference,
    variants_fn=variants_df,
    encode=True  # Returns one-hot encoded arrays/tensors
)

# Access chromosome sequences
chr1_encoded = personal_genome['chr1']  # Shape: (seq_length, 4)

# Or get raw sequence strings
personal_genome_raw = sl.get_personal_genome(
    reference_fn=reference,
    variants_fn=variants_df,
    encode=False  # Returns sequence strings
)
chr1_sequence = personal_genome_raw['chr1']  # String of nucleotides
```

### 2. Variant-Centered Sequence Windows

Generate sequence windows centered on each variant position:

```python
# Create 1000bp windows around each variant
sequences = sl.get_personal_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    seq_len=1000,
    encode=True
)

# Returns tensor/array of shape (n_variants, seq_len, 4)
print(f"Generated {sequences.shape[0]} sequences of length {sequences.shape[1]}")

# Or get raw sequences with metadata
sequences_raw = sl.get_personal_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    seq_len=1000,
    encode=False
)

# Returns list of tuples: (chrom, start, end, sequence_string)
for chrom, start, end, sequence in sequences_raw:
    print(f"{chrom}:{start}-{end}: {sequence[:50]}...")
```

### 3. PAM-Disrupting Variants

Identify variants that disrupt PAM sites and generate corresponding sequences:

```python
# Find variants that disrupt SpCas9 PAM sites (NGG)
pam_results = sl.get_pam_disrupting_personal_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    seq_len=1000,
    max_pam_distance=50,  # Maximum distance from variant to PAM
    pam_sequence="NGG",   # SpCas9 PAM sequence
    encode=True
)

# Access results
pam_disrupting_variants = pam_results['variants']
pam_intact_sequences = pam_results['pam_intact']
pam_disrupted_sequences = pam_results['pam_disrupted']

print(f"Found {len(pam_disrupting_variants)} PAM-disrupting variants")
```

### 4. In-Silico Mutagenesis

Generate sequences with systematic mutations at every position:

```python
# Saturation mutagenesis for a genomic region
ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
    chrom='chr1',
    start=1000000,
    end=1001000,
    reference_fasta=reference
)

print(f"Reference sequence shape: {ref_seq.shape}")
print(f"Alternative sequences shape: {alt_seqs.shape}")
print(f"Generated {len(metadata)} mutated sequences")

# Metadata contains mutation information
print(metadata.head())
#    chrom    start      end  offset ref alt
# 0   chr1  1000000  1001000       0   A   C
# 1   chr1  1000000  1001000       0   A   G
# 2   chr1  1000000  1001000       0   A   T
# 3   chr1  1000000  1001000       1   T   A
# 4   chr1  1000000  1001000       1   T   C
```

### 5. Targeted Mutagenesis Around Anchor Points

Generate mutations around specific genomic positions:

```python
# Mutagenesis around a specific anchor point
ref_seq, alt_seqs, metadata = sl.get_sm_subsequences(
    chrom='chr1',
    anchor=1000500,      # Center position
    anchor_radius=25,    # Mutate ±25 bp around anchor
    seq_len=1000,        # Total sequence length
    reference_fasta=reference
)

print(f"Generated mutations around position 1000500")
print(f"Mutated {alt_seqs.shape[0]} positions")
```

### 6. Sequence Utilities

Work with encoded sequences using utility functions:

```python
# Encode a sequence string
sequence = "ATCGATCGATCG"
encoded = sl.encode_seq(sequence)
print(f"Encoded shape: {encoded.shape}")  # (12, 4)

# Decode back to string
decoded = sl.decode_seq(encoded)
print(f"Decoded: {decoded}")  # "ATCGATCGATCG"

# Reverse complement
rc_encoded = sl.rc(encoded)
rc_string = sl.rc_str(sequence)
print(f"Reverse complement: {rc_string}")  # "CGATCGATCGAT"

# Encode multiple sequences
sequences = ["ATCG", "GCTA", "TTAA"]
batch_encoded = sl.encode_seq(sequences)
print(f"Batch encoded shape: {batch_encoded.shape}")  # (3, 4, 4)
```

### 7. Reading VCF Files

Read and process VCF files:

```python
# Read VCF file into DataFrame
variants_df = sl.read_vcf('variants.vcf')
print(variants_df.head())

# VCF DataFrame contains columns: chrom, pos, id, ref, alt
filtered_variants = variants_df[variants_df['chrom'] == 'chr1']
```

### 8. PyTorch Integration

When PyTorch is available, sequences are returned as PyTorch tensors:

```python
import torch

# Check if PyTorch is available
print(f"PyTorch available: {sl.TORCH_AVAILABLE}")

# Sequences will be PyTorch tensors if available
sequences = sl.get_personal_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    seq_len=1000
)

if sl.TORCH_AVAILABLE:
    print(f"Tensor type: {type(sequences)}")  # <class 'torch.Tensor'>
    print(f"Device: {sequences.device}")      # cpu
    print(f"Dtype: {sequences.dtype}")        # torch.float32
    
    # Move to GPU if available
    if torch.cuda.is_available():
        sequences = sequences.cuda()
```

### 9. Working with Different Reference Formats

The package supports multiple reference genome formats:

```python
# From FASTA file path
personal_genome = sl.get_personal_genome('hg38.fa', variants_df)

# From pyfaidx Fasta object
reference = Fasta('hg38.fa')
personal_genome = sl.get_personal_genome(reference, variants_df)

# From dictionary (for small sequences)
reference_dict = {
    'chr1': 'ATCGATCGATCG...',
    'chr2': 'GCTAGCTAGCTA...'
}
personal_genome = sl.get_personal_genome(reference_dict, variants_df)
```

### 10. Error Handling and Warnings

The package provides informative warnings for common issues:

```python
import warnings

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # This may generate warnings for overlapping variants
    personal_genome = sl.get_personal_genome(
        reference_fn='hg38.fa',
        variants_fn='variants.vcf'
    )
    
    for warning in w:
        print(f"Warning: {warning.message}")
```

## API Reference

### Core Functions

- `get_personal_genome(reference_fn, variants_fn, encode=True)` - Generate personalized genome
- `get_personal_sequences(reference_fn, variants_fn, seq_len, encode=True)` - Generate variant-centered windows
- `get_pam_disrupting_personal_sequences(...)` - Find PAM-disrupting variants
- `get_sm_sequences(chrom, start, end, reference_fasta)` - Saturation mutagenesis
- `get_sm_subsequences(chrom, anchor, anchor_radius, seq_len, reference_fasta)` - Targeted mutagenesis

### Utility Functions

- `encode_seq(seq)` - Convert nucleotide string to one-hot encoding
- `decode_seq(seq_1h)` - Convert one-hot encoding to nucleotide string
- `rc(seq_1h)` - Reverse complement encoded sequence
- `rc_str(seq)` - Reverse complement string
- `read_vcf(path)` - Read VCF file into DataFrame

### Constants

- `TORCH_AVAILABLE` - Boolean indicating if PyTorch is available
- `nt_to_1h` - Nucleotide to one-hot encoding mapping
- `nts` - Nucleotide array `['A', 'C', 'G', 'T']`

## Issues and Support

We welcome feedback, bug reports, and feature requests! If you encounter any issues or have suggestions for improvements, please:

1. **Check existing issues** first to see if your problem has already been reported
2. **File a new issue** on our [GitHub Issues page](https://github.com/gladstone-institutes/supremo_lite/issues)
3. **Provide detailed information** including:
   - Python version and operating system
   - Package version (`supremo_lite.__version__`)
   - Complete error messages and stack traces
   - Minimal reproducible example
   - Expected vs. actual behavior

### Common Issues to Report

- **Performance problems** with large genomes or variant files
- **Compatibility issues** with different file formats
- **Unexpected behavior** with edge cases
- **Documentation gaps** or unclear examples
- **Feature requests** for new functionality


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`supremo_lite` was created by Natalie Gill and Sean Whalen, based on Sequence Mutator for Predictive Models ([SuPreMo](https://github.com/ketringjoni/SuPreMo)) by Katie Gjoni. It is licensed under the terms of the MIT license.

## Credits

`supremo_lite` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).