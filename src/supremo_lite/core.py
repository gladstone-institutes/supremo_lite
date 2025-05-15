"""
Core utilities, constants and common functions for supremo_lite.

This module provides the basic constants and utility functions used throughout
the package.
"""

import numpy as np
from collections import defaultdict
import warnings

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Will return numpy arrays instead of tensors.")

# Nucleotide to one-hot encoding mapping
# Using a defaultdict to handle ambiguous bases with uniform probabilities
nt_to_1h = defaultdict(lambda: np.array([0.25, 0.25, 0.25, 0.25]))
nt_to_1h['A'] = np.array([1, 0, 0, 0])
nt_to_1h['a'] = np.array([1, 0, 0, 0])
nt_to_1h['C'] = np.array([0, 1, 0, 0])
nt_to_1h['c'] = np.array([0, 1, 0, 0])
nt_to_1h['G'] = np.array([0, 0, 1, 0])
nt_to_1h['g'] = np.array([0, 0, 1, 0])
nt_to_1h['T'] = np.array([0, 0, 0, 1])
nt_to_1h['t'] = np.array([0, 0, 0, 1])
nts = np.array(list('ACGT'))