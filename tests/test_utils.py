"""
Test utilities for supremo_lite tests.

This module previously contained MockModel and create_test_predictions_dataset(),
which have been replaced by the official mock models in supremo_lite.mock_models.

For testing with mock models, use:
    from supremo_lite.mock_models import TestModel, TestModel2D

The new mock models are now part of the package and serve dual purposes:
1. Testing infrastructure for prediction alignment
2. User-facing demonstrations and documentation
"""

# This file is intentionally minimal now that mock models have been moved
# to the main package. Additional test utilities can be added here as needed.
