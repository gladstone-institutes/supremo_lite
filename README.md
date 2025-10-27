# supremo_lite

A lightweight memory-first, model-agnostic version of [SuPreMo](https://github.com/ketringjoni/SuPreMo).

## Key Features

- 🧬 **Personalized Genome Generation**: Apply variants from VCF files to reference genomes
- 🎯 **Variant-Centered Sequences**: Generate sequence windows around variants
- ✂️ **PAM Site Analysis**: Identify variants that disrupt CRISPR PAM sites
- 🧪 **Saturation Mutagenesis**: Systematic single-nucleotide mutations at every position for predictive modeling
- 🔧 **Memory Efficient**: Chunked processing for large VCF files
- 🗺️ **Chromosome Matching**: Optional handling of chromosome naming differences (chr1 ↔ 1, chrM ↔ MT) via `auto_map_chromosomes=True`
- ⚡ **PyTorch Integration**: Automatic tensor support when PyTorch is available

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
- [https://github.com/gladstone-institutes/brisket](brisket) - Cython powered faster 1 hot encoding for DNA sequences (automatically detected)

## Quick Start

```python
import supremo_lite as sl
from pyfaidx import Fasta

# Load reference genome and variants
reference = Fasta('hg38.fa')
variants = sl.read_vcf('variants.vcf')
```

### DNA Sequence Encoding

supremo_lite uses **one-hot encoding** by default:
- `A` = `[1,0,0,0]`, `C` = `[0,1,0,0]`, `G` = `[0,0,1,0]`, `T` = `[0,0,0,1]`
- Ambiguous bases = `[0,0,0,0]`
- Returns PyTorch tensors when available, otherwise NumPy arrays

### Personalized Genome Generation

```python
# Apply variants to create personalized genome
personal_genome = sl.get_personal_genome(
    reference_fn=reference,
    variants_fn=variants,
    encode=True,      # One-hot encoded (or False for strings)
    chunk_size=10000, # Process 10k variants at a time
    verbose=True      # Show progress
)

# If your VCF uses 'chr1' and reference uses '1', enable chromosome mapping
personal_genome = sl.get_personal_genome(
    reference_fn=reference,
    variants_fn=variants,
    auto_map_chromosomes=True  # Handle chromosome name differences
)
```

**📖 [Full Guide: Personalized Genomes](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/personalization.md) | [Tutorial Notebook](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/notebooks/02_personalized_genomes.ipynb)**

### Variant-Centered Sequences

```python
# Generate reference and alternate sequences around variants
# Note: get_alt_ref_sequences is a generator that yields chunks
results = list(sl.get_alt_ref_sequences(
    reference_fn=reference,
    variants_fn=variants,
    seq_len=1000,
    encode=True
))
# Unpack from the first chunk
alt_seqs, ref_seqs, metadata = results[0]
# Returns: (n_variants, seq_len, 4) shaped arrays
```

**📖 [Full Guide: Variant-Centered Sequences](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/variant_centered_sequences.md) | [Getting Started Notebook](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/notebooks/01_getting_started.ipynb)**

### Prediction Alignment

```python
# Align model predictions accounting for variant coordinate changes
from supremo_lite.mock_models import TestModel

model = TestModel(n_targets=2, bin_size=8, crop_length=10)
ref_preds = model(ref_seqs)
alt_preds = model(alt_seqs)

ref_aligned, alt_aligned = sl.align_predictions_by_coordinate(
    ref_pred=ref_preds[0],
    alt_pred=alt_preds[0],
    metadata=metadata[0],
    prediction_type="1D",
    bin_size=8,
    crop_length=10
)
```

**📖 [Full Guide: Prediction Alignment](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/prediction_alignment.md) | [Tutorial with Visualizations](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/notebooks/03_prediction_alignment.ipynb)**

### Saturation Mutagenesis

```python
# Mutate every position in a region
ref_seq, alt_seqs, metadata = sl.get_sm_sequences(
    chrom='chr1',
    start=1000,
    end=1100,  # 100 bp → 300 mutations (3 per position)
    reference_fasta=reference
)
```

**📖 [Full Guide: Mutagenesis](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/mutagenesis.md)**

## Documentation

### 📚 User Guides
Detailed documentation for each major feature:
- **[Personalized Genomes](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/personalization.md)** - Apply variants to genomes
- **[Variant-Centered Sequences](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/variant_centered_sequences.md)** - Extract sequence windows around variants
- **[Prediction Alignment](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/prediction_alignment.md)** - Align model predictions for variant effect analysis
- **[Saturation Mutagenesis](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/user_guide/mutagenesis.md)** - In-silico mutagenesis workflows
- **[Variant Classification](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/_static/images/variant_classification.png)** - Flow chart showing automatic variant classification logic

### 📓 Interactive Tutorials
Hands-on Jupyter notebooks with visualizations:
- **[Getting Started](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/notebooks/01_getting_started.ipynb)** - Installation and basic concepts
- **[Personalized Genomes](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/notebooks/02_personalized_genomes.ipynb)** - Genome personalization workflows
- **[Prediction Alignment](https://github.com/gladstone-institutes/supremo_lite/blob/main/docs/notebooks/03_prediction_alignment.ipynb)** - Complete prediction workflow with visualizations ⭐

### 🔍 API Reference
**Core Functions:**
- `get_personal_genome()` - Generate personalized genomes
- `get_alt_ref_sequences()` - Generate variant-centered sequences
- `align_predictions_by_coordinate()` - Align model predictions
- `get_sm_sequences()` - Saturation mutagenesis
- `read_vcf()` - Read VCF files

For complete API documentation with all parameters, see the [docs/](https://github.com/gladstone-institutes/supremo_lite/tree/main/docs) directory.

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