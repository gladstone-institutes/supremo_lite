# supremo_lite

A lightweight memory first, model agnostic version of [SuPreMo](https://github.com/ketringjoni/SuPreMo).

## Key Features

- 🧬 **Personalized Genome Generation**: Apply variants from VCF files to reference genomes
- 🎯 **Variant-Centered Sequences**: Generate sequence windows around variants
- ✂️ **PAM Site Analysis**: Identify variants that disrupt CRISPR PAM sites
- 🧪 **Saturation Mutagenesis**: Systematic single-nucleotide mutations at every position for predictive modeling
- 🔧 **Memory Efficient**: Chunked processing for large VCF files
- 🗺️ **Smart Chromosome Matching**: Automatic handling of chromosome naming differences (chr1 ↔ 1, chrM ↔ MT)
- ⚡ **PyTorch Integration**: Automatic tensor support when PyTorch is available
- 📊 **Format Flexibility**: Works with file paths, objects, or DataFrames

## Installation

### Install from GitHub (Recommended)

For the latest features and bug fixes:

```bash
# Install directly from GitHub (latest release)
pip install git+https://github.com/gladstone-institutes/supremo_lite.git

# Or install a specific version/tag
pip install git+https://github.com/gladstone-institutes/supremo_lite.git@v0.5.0

# Or install from a specific branch
pip install git+https://github.com/gladstone-institutes/supremo_lite.git@main
```

### Install from Local Clone

For development or customization:

```bash
git clone https://github.com/gladstone-institutes/supremo_lite.git
cd supremo_lite
pip install .

# For development with editable install
pip install -e .
```

### Dependencies

Required dependencies will be installed automatically:
- `pandas` - For VCF data handling
- `numpy` - For numerical operations  
- `pyfaidx` - For FASTA file reading

Optional dependencies:
- `torch` - For PyTorch tensor support (automatically detected)

## Usage

`supremo_lite` provides functionality for generating personalized genome sequences from reference genomes and variant files, as well as performing saturation mutagenesis. The package supports both PyTorch tensors and NumPy arrays for sequence encoding.

### Basic Imports

```python
import supremo_lite as sl
from pyfaidx import Fasta
import pandas as pd
```

### Default Sequence Encoding

`supremo_lite` uses one-hot encoding by default to convert DNA sequences into numerical arrays. The default encoding scheme is:

```python
# Default one-hot encoding (4-dimensional vectors)
'A' = [1, 0, 0, 0]  # Position 0
'C' = [0, 1, 0, 0]  # Position 1  
'G' = [0, 0, 1, 0]  # Position 2
'T' = [0, 0, 0, 1]  # Position 3

# Ambiguous or unknown nucleotides (N, Y, R, etc.)
'N' = [0, 0, 0, 0]  # All zeros for ambiguous bases
```

**Key Points:**
- Each nucleotide is represented as a 4-dimensional vector
- Standard bases (A, C, G, T) use one-hot encoding with exactly one position set to 1
- Ambiguous bases and unrecognized characters default to `[0, 0, 0, 0]`
- The encoding is case-insensitive (`'A'` and `'a'` both map to `[1, 0, 0, 0]`)
- When PyTorch is available, sequences are returned as float32 tensors; otherwise as NumPy arrays

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
chr1_sequence = personal_genome_raw['chr1']
```

#### Memory-Efficient Processing for Large VCF Files

For large variant files, use chunked processing to manage memory usage:

```python
# Process large VCF files in chunks (reduces memory usage)
personal_genome = sl.get_personal_genome(
    reference_fn='hg38.fa',
    variants_fn='large_variants.vcf',
    encode=False,
    chunk_size=10000  # Process 10,000 variants at a time
)

# chunk_size=1 (default) loads entire VCF
# chunk_size>1 enables chunked processing for memory efficiency
```

### 2. Variant-Centered Sequence Windows

Generate sequence windows centered on each variant position:

```python
# Create 1000bp windows around each variant
sequences = sl.get_alt_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    seq_len=1000,
    encode=True,
    chunk_size=1  # Process variants individually (default)
)

# Returns tensor/array of shape (n_variants, seq_len, 4)
print(f"Generated {sequences.shape[0]} sequences of length {sequences.shape[1]}")

# Or get raw sequences with metadata
sequences_raw = sl.get_alt_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf',
    seq_len=1000,
    encode=False
)

# Returns list of tuples: (chrom, start, end, sequence_string)
for chrom, start, end, sequence in sequences_raw:
    print(f"{chrom}:{start}-{end}: {sequence[:50]}...")
```

#### Automatic Chromosome Name Matching

The package automatically handles chromosome naming mismatches between VCF and FASTA files:

```python
# Works automatically even with mismatched naming
# VCF has: 'chr1', 'chr2', 'chrX', 'chrM' 
# FASTA has: '1', '2', 'X', 'MT'

personal_genome = sl.get_personal_genome(
    reference_fn='reference.fa',  # Uses '1', '2', 'X', 'MT'
    variants_fn='variants.vcf'    # Uses 'chr1', 'chr2', 'chrX', 'chrM'
)
# Chromosomes are automatically matched and mapped

# You can also use the chromosome utilities directly
ref_chroms = {'1', '2', 'X', 'MT'}
vcf_chroms = {'chr1', 'chr2', 'chrX', 'chrM'}

mapping, unmatched = sl.create_chromosome_mapping(ref_chroms, vcf_chroms)
print(f"Chromosome mapping: {mapping}")
# Output: {'chr1': '1', 'chr2': '2', 'chrX': 'X', 'chrM': 'MT'}
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
    encode=True,
    chunk_size=1         # Default chunk size
)

# Access results
pam_disrupting_variants = pam_results['variants']
pam_intact_sequences = pam_results['pam_intact']
pam_disrupted_sequences = pam_results['pam_disrupted']

print(f"Found {len(pam_disrupting_variants)} PAM-disrupting variants")
```

### 4. In-Silico Saturation Mutagenesis

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
# Encode a sequence string (uses default one-hot encoding)
sequence = "ATCGATCGATCG"
encoded = sl.encode_seq(sequence)
print(f"Encoded shape: {encoded.shape}")  # (12, 4)

# Or with a custom encoder
# encoded = sl.encode_seq(sequence, encoder=my_custom_encoder)

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

### 7. Custom Encoding Functions

You can provide your own encoding function for specialized use cases:

```python
import numpy as np

def custom_encoder(seq):
    """
    Custom encoder example: reverse the default A,C,G,T order to T,G,C,A
    and use uniform probability for ambiguous bases.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        numpy array with shape (L, 4) where L is sequence length
    """
    # Custom encoding map (reversed order from supremo_lite default)
    encoding_map = {
        'A': [0, 0, 0, 1],  # A -> position 3 (default: position 0)
        'C': [0, 0, 1, 0],  # C -> position 2 (default: position 1)
        'G': [0, 1, 0, 0],  # G -> position 1 (default: position 2)
        'T': [1, 0, 0, 0],  # T -> position 0 (default: position 3)
        'N': [0.25, 0.25, 0.25, 0.25]   # Uniform probability (default: [0,0,0,0])
    }
    
    result = np.array([encoding_map.get(nt.upper(), [0.25, 0.25, 0.25, 0.25) for nt in seq])
    return result.astype(np.float32)

# Use custom encoder with any function
sequence = "ATCG"
custom_encoded = sl.encode_seq(sequence, encoder=custom_encoder)

# Works with all personalization and mutagenesis functions
personal_genome = sl.get_personal_genome(
    'reference.fa', 
    'variants.vcf', 
    encoder=custom_encoder
)

ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
    'chr1', 1000, 1100, 
    reference_fasta, 
    encoder=custom_encoder
)
```

**Custom Encoder Requirements:**
- Must accept a single sequence string as input
- Must return a numpy array with shape `(L, 4)` where L is sequence length
- Should handle uppercase nucleotides (A, C, G, T, N)
- Return type should be compatible with numpy/torch operations

### 8. Reading VCF Files

Read and process VCF files:

```python
# Read VCF file into DataFrame
variants_df = sl.read_vcf('variants.vcf')
print(variants_df.head())

# VCF DataFrame contains columns: chrom, pos, id, ref, alt
filtered_variants = variants_df[variants_df['chrom'] == 'chr1']

# For very large VCF files, use chunked reading
for chunk in sl.read_vcf_chunked('large_variants.vcf', chunk_size=50000):
    # Process each chunk of 50,000 variants
    print(f"Processing chunk with {len(chunk)} variants")
    # Your processing code here...
```

### 8. PyTorch Integration

When PyTorch is available, sequences are returned as PyTorch tensors:

```python
import torch

# Check if PyTorch is available
print(f"PyTorch available: {sl.TORCH_AVAILABLE}")

# Sequences will be PyTorch tensors if available
sequences = sl.get_alt_sequences(
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

### 11. Performance Tips

For optimal performance with large datasets:

```python
# Use chunked processing for large VCF files
personal_genome = sl.get_personal_genome(
    reference_fn='hg38.fa',
    variants_fn='large_variants.vcf',
    encode=False,        # Use strings to save memory
    chunk_size=50000     # Process in 50k variant chunks
)

# For sequence generation, balance chunk size with memory
sequences = list(sl.get_alt_sequences(
    reference_fn='hg38.fa', 
    variants_fn='variants.vcf',
    seq_len=1000,
    chunk_size=100      # Generate 100 sequences at a time
))

# Disable encoding for large datasets to save memory
sequences_raw = sl.get_alt_sequences(
    reference_fn='hg38.fa',
    variants_fn='variants.vcf', 
    seq_len=1000,
    encode=False        # Returns strings instead of tensors
)
```

## API Reference

### Core Functions

- `get_personal_genome(reference_fn, variants_fn, encode=True, n_chunks=1, verbose=False, encoder=None)` - Generate personalized genome with chunked processing
- `get_alt_sequences(reference_fn, variants_fn, seq_len, encode=True, n_chunks=1, encoder=None)` - Generate variant-centered windows  
- `get_pam_disrupting_personal_sequences(reference_fn, variants_fn, seq_len, max_pam_distance, pam_sequence="NGG", encode=True, n_chunks=1, encoder=None)` - Find PAM-disrupting variants
- `get_sm_sequences(chrom, start, end, reference_fasta, encoder=None)` - Saturation mutagenesis
- `get_sm_subsequences(chrom, anchor, anchor_radius, seq_len, reference_fasta, bed_regions=None, encoder=None)` - Targeted mutagenesis

### VCF Processing Functions

- `read_vcf(path)` - Read VCF file into DataFrame
- `read_vcf_chunked(path, chunk_size)` - Read large VCF files in chunks (generator)

### Chromosome Utilities

- `normalize_chromosome_name(chrom_name)` - Normalize chromosome naming
- `create_chromosome_mapping(ref_chroms, vcf_chroms)` - Create chromosome name mapping
- `match_chromosomes_with_report(ref_chroms, vcf_chroms, verbose=True)` - Match with detailed reporting

### Sequence Utilities

- `encode_seq(seq, encoder=None)` - Convert nucleotide string to one-hot encoding (A→[1,0,0,0], C→[0,1,0,0], G→[0,0,1,0], T→[0,0,0,1], ambiguous→[0,0,0,0])
- `decode_seq(seq_1h)` - Convert one-hot encoding to nucleotide string
- `rc(seq_1h)` - Reverse complement encoded sequence
- `rc_str(seq)` - Reverse complement string

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
- **Unexpected behavior** with edge cases
- **Documentation gaps** or unclear examples
- **Feature requests** for new functionality


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`supremo_lite` was created by Natalie Gill and Sean Whalen, based on Sequence Mutator for Predictive Models ([SuPreMo](https://github.com/ketringjoni/SuPreMo)) by Katie Gjoni. It is licensed under the terms of the MIT license.

## Credits

`supremo_lite` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).