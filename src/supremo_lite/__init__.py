"""
supremo_lite: A module for generating personalized genome sequences from a reference
fasta and a variants file, or sequences for in-silico mutagenesis.

This package provides functionality for:
- Sequence encoding and transformation
- Variant reading and application
- In-silico mutagenesis
"""

# Import core components
from .core import TORCH_AVAILABLE, nt_to_1h, nts

# Import sequence transformation utilities
from .sequence_utils import encode_seq, decode_seq, rc, rc_str

# Import variant reading utilities
from .variant_utils import read_vcf, read_vcf_chunked

# Import chromosome matching utilities
from .chromosome_utils import (
    normalize_chromosome_name,
    create_chromosome_mapping,
    match_chromosomes_with_report,
)

# Import personalize functions
from .personalize import (
    get_personal_genome,
    get_personal_sequences,
    get_pam_disrupting_personal_sequences,
)

# Import mutagenesis functions
from .mutagenesis import get_sm_sequences, get_sm_subsequences

# Version
__version__ = "0.4.0"
# Package metadata
__description__ = (
    "A module for generating personalized genome sequences and in-silico mutagenesis"
)
