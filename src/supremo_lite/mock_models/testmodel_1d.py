"""
Mock 1D genomic prediction model for testing and demonstrations.

This module provides a simple PyTorch model that mimics realistic genomic deep learning
architectures. It is intended for:
1. Testing prediction alignment functionality
2. Providing runnable examples for users without trained models

**NOT for actual genomic predictions** - this model returns constant values and has
no learned parameters.

Model Architecture Characteristics:
- **Binning**: Predictions at lower resolution than input (bin_size parameter)
- **Cropping**: Edge bins removed to focus on central regions (crop_length parameter)
- **Output shape**: (batch_size, n_targets, n_final_bins)

Example:
    >>> from supremo_lite.mock_models import TestModel
    >>> import torch
    >>>
    >>> model = TestModel(seq_length=1024, bin_length=32, crop_length=128, n_targets=1)
    >>> x = torch.randn(8, 4, 1024)  # (batch, channels, length)
    >>> predictions = model(x)
    >>> predictions.shape
    torch.Size([8, 1, 28])  # (batch, targets, bins after cropping)

"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


if TORCH_AVAILABLE:
    class TestModel(nn.Module):
        """
        Mock 1D genomic prediction model.

        This model demonstrates typical genomic deep learning architecture features:
        - Accepts one-hot encoded DNA sequences
        - Outputs binned predictions at lower resolution
        - Applies edge cropping to focus on central regions

        **Warning**: Returns constant values (ones). Not for actual predictions.

        Parameters
        ----------
        seq_length : int
            Length of input sequences in base pairs
        bin_length : int
            Number of base pairs per prediction bin
        crop_length : int, optional
            Number of base pairs to crop from each edge (default: 0)
        n_targets : int, optional
            Number of prediction targets per bin (default: 1)

        Attributes
        ----------
        seq_length : int
            Input sequence length
        bin_length : int
            Bin size in base pairs
        crop_length : int
            Cropped bases per edge
        n_targets : int
            Number of prediction targets
        crop_bins : int
            Number of bins to crop from each edge
        n_initial_bins : int
            Total bins before cropping
        n_final_bins : int
            Final number of bins after cropping

        Examples
        --------
        Basic usage:

        >>> model = TestModel(seq_length=1024, bin_length=32)
        >>> x = torch.randn(4, 4, 1024)  # (batch, channels, length)
        >>> out = model(x)
        >>> out.shape
        torch.Size([4, 1, 32])

        With cropping:

        >>> model = TestModel(seq_length=2048, bin_length=64, crop_length=256)
        >>> model.n_initial_bins
        32
        >>> model.crop_bins
        4
        >>> model.n_final_bins
        24
        """

        def __init__(self, seq_length, bin_length, crop_length=0, n_targets=1):
            super().__init__()

            self.seq_length = seq_length
            self.bin_length = bin_length
            self.crop_length = crop_length
            self.n_targets = n_targets

            self.crop_bins = crop_length // bin_length
            self.n_initial_bins = seq_length // bin_length
            self.n_final_bins = self.n_initial_bins - 2 * self.crop_bins

        def forward(self, x):
            """
            Forward pass returning mock predictions.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, 4, seq_length)
                Channel dimension should be 4 (one-hot encoded A, C, G, T)

            Returns
            -------
            torch.Tensor
                Mock predictions of shape (batch_size, n_targets, n_final_bins)
                Contains all ones (not meaningful predictions)
            """
            assert x.shape[1] == 4, f"Expected 4 channels (one-hot), got {x.shape[1]}"
            assert x.shape[2] == self.seq_length, \
                f"Expected sequence length {self.seq_length}, got {x.shape[2]}"

            return torch.ones([x.shape[0], self.n_targets, self.n_final_bins])

        def training_step(self, batch, batch_idx):
            """
            Mock training step for demonstration purposes.

            Shows how cropping would be applied during training.
            This is for educational purposes only - the model has no learnable parameters.
            """
            x, y = batch
            # Crop target predictions to match model output
            y = y[:, :, self.crop_bins:-self.crop_bins]
            y_hat = self(x)
            return nn.functional.mse_loss(y_hat, y)

else:
    # Fallback when PyTorch not available
    class TestModel:
        """TestModel requires PyTorch. Please install with: pip install torch"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TestModel requires PyTorch. Install with: pip install torch\n"
                "See https://pytorch.org/get-started/locally/ for installation instructions."
            )


# Make TestModel available at module level
__all__ = ['TestModel', 'TORCH_AVAILABLE']


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    # Demonstration of model behavior
    batch_size = 8
    seq_length = 524288
    bin_length = 32
    crop_length = 163840
    n_targets = 2

    crop_bins = crop_length // bin_length
    n_initial_bins = seq_length // bin_length
    n_cropped_bins = n_initial_bins - 2 * crop_bins

    print(f"Model Configuration:")
    print(f"  Sequence length: {seq_length:,} bp")
    print(f"  Bin length: {bin_length} bp")
    print(f"  Crop length: {crop_length:,} bp")
    print(f"  Initial bins: {n_initial_bins}")
    print(f"  Crop bins per edge: {crop_bins}")
    print(f"  Final bins: {n_cropped_bins}")
    print(f"  Targets: {n_targets}")

    m = TestModel(seq_length, bin_length, crop_length, n_targets)
    x = torch.ones([batch_size, 4, seq_length])

    y_hat = m(x)
    assert y_hat.shape[0] == batch_size
    assert y_hat.shape[1] == n_targets
    assert y_hat.shape[2] == n_cropped_bins

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y_hat.shape}")
    print("✓ Model test passed!")
