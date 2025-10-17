```{include} ../README.md
```

## Documentation Structure

This documentation is organized into the following sections:

- **[User Guide](user_guide/index)**: Detailed guides for each major feature
- **[Notebooks](notebooks/index)**: Interactive tutorials with hands-on examples
- **[API Reference](autoapi/index)**: Complete API documentation

## Quick Links

### Getting Started
- [Installation & Basic Concepts](notebooks/01_getting_started.ipynb) - Start here!
- [Personalized Genomes](user_guide/personalization.md) - Apply variants to genomes
- [Sequence Generation](user_guide/sequences.md) - Create variant-centered windows

### Core Workflows
- [Prediction Alignment Tutorial](notebooks/03_prediction_alignment.ipynb) - Complete workflow with visualizations
- [Prediction Alignment Guide](user_guide/prediction_alignment.md) - Detailed alignment documentation
- [Mock Models](user_guide/mock_models.md) - TestModel and TestModel2D

### Advanced Topics
- [Structural Variants](notebooks/04_structural_variants.ipynb) - INV, DUP, BND handling
- [Saturation Mutagenesis](notebooks/05_saturation_mutagenesis.ipynb) - In-silico mutagenesis
- [Mutagenesis Guide](user_guide/mutagenesis.md) - Systematic mutation workflows

```{toctree}
:maxdepth: 2
:caption: User Guide
:hidden:

user_guide/personalization
user_guide/sequences
user_guide/prediction_alignment
user_guide/mutagenesis
user_guide/mock_models
```

```{toctree}
:maxdepth: 2
:caption: Notebooks
:hidden:

notebooks/01_getting_started
notebooks/02_personalized_genomes
notebooks/03_prediction_alignment
notebooks/04_structural_variants
notebooks/05_saturation_mutagenesis
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

autoapi/index
changelog.md
contributing.md
conduct.md
```