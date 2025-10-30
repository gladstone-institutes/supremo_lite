"""
Mock models for testing and demonstration purposes.

This module provides simple PyTorch models that mimic realistic genomic deep learning
architectures without requiring actual training. These models are intended for:

1. **Testing**: Verifying that prediction alignment functions work correctly with
   realistic model outputs (binned predictions, edge cropping, diagonal masking)

2. **Documentation**: Providing immediately runnable examples for users who want to
   understand the package workflow without training their own models

**Important**: These models return constant values and should NOT be used for actual
genomic predictions or biological interpretation.

Available Models
----------------
TestModel : nn.Module
    Mock 1D genomic prediction model
    - Output shape: (batch_size, n_targets, n_final_bins)
    - Features: binning, edge cropping

TestModel2D : nn.Module
    Mock 2D contact map prediction model
    - Output shape: (batch_size, n_targets, n_flattened_ut_bins)
    - Features: binning, edge cropping, diagonal masking, flattened output

Examples
--------
Using TestModel for 1D predictions:

>>> from supremo_lite.mock_models import TestModel, TORCH_AVAILABLE
>>> if TORCH_AVAILABLE:
...     import torch
...     model = TestModel(seq_length=1024, bin_length=32, crop_length=128)
...     x = torch.randn(4, 4, 1024)
...     predictions = model(x)
...     print(predictions.shape)
torch.Size([4, 1, 24])

Using TestModel2D for contact maps:

>>> from supremo_lite.mock_models import TestModel2D
>>> if TORCH_AVAILABLE:
...     import torch
...     model = TestModel2D(seq_length=2048, bin_length=64, crop_length=256)
...     x = torch.randn(4, 4, 2048)
...     predictions = model(x)
...     print(predictions.shape)
torch.Size([4, 1, 276])

Checking PyTorch Availability
------------------------------
>>> from supremo_lite.mock_models import TORCH_AVAILABLE
>>> if not TORCH_AVAILABLE:
...     print("Please install PyTorch to use mock models")

Notes
-----
- Requires PyTorch to be installed
- If PyTorch is not available, attempting to instantiate models will raise ImportError
- Check TORCH_AVAILABLE before using models
- See individual model documentation for architecture details
"""

try:
    from .testmodel_1d import TestModel, TORCH_AVAILABLE as TORCH_AVAILABLE_1D
    from .testmodel_2d import TestModel2D, TORCH_AVAILABLE as TORCH_AVAILABLE_2D

    # Both should have the same value, but check for consistency
    TORCH_AVAILABLE = TORCH_AVAILABLE_1D and TORCH_AVAILABLE_2D

except ImportError as e:
    # This should rarely happen since the modules handle their own imports
    # But we provide a graceful fallback
    import warnings

    warnings.warn(
        f"Could not import mock models: {e}\n"
        "Mock models require PyTorch. Install with: pip install torch",
        ImportWarning,
    )

    # Create placeholder classes
    class TestModel:
        """TestModel requires PyTorch. Please install with: pip install torch"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TestModel requires PyTorch. Install with: pip install torch\n"
                "See https://pytorch.org/get-started/locally/"
            )

    class TestModel2D:
        """TestModel2D requires PyTorch. Please install with: pip install torch"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TestModel2D requires PyTorch. Install with: pip install torch\n"
                "See https://pytorch.org/get-started/locally/"
            )

    TORCH_AVAILABLE = False


__all__ = [
    "TestModel",
    "TestModel2D",
    "TORCH_AVAILABLE",
]
