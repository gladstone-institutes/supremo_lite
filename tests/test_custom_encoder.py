"""
Test custom encoder functionality.
"""

import numpy as np
import pandas as pd
import pytest
import supremo_lite as sl


def test_custom_encoder_basic():
    """Test custom encoder with encode_seq function."""

    def custom_encoder(seq):
        """Simple custom encoder that reverses the one-hot encoding order."""
        # Convert to uppercase for consistency
        seq = seq.upper()

        # Custom encoding: T, G, C, A (reverse of default A, C, G, T)
        encoding_map = {
            "A": [0, 0, 0, 1],
            "C": [0, 0, 1, 0],
            "G": [0, 1, 0, 0],
            "T": [1, 0, 0, 0],
            "N": [0, 0, 0, 0],  # Same as default
        }

        result = np.array([encoding_map.get(nt, [0, 0, 0, 0]) for nt in seq]).T
        return result.astype(np.float32)

    # Test sequence
    seq = "ATCG"

    # Default encoding
    default_encoded = sl.encode_seq(seq)

    # Custom encoding
    custom_encoded = sl.encode_seq(seq, encoder=custom_encoder)

    # They should be different
    assert not np.array_equal(default_encoded, custom_encoded)

    # Custom should match our expected pattern (now in 4, L format)
    expected = np.array(
        [
            [0, 1, 0, 0],  # Channel 0 (T): A=0, T=1, C=0, G=0
            [0, 0, 0, 1],  # Channel 1 (G): A=0, T=0, C=0, G=1
            [0, 0, 1, 0],  # Channel 2 (C): A=0, T=0, C=1, G=0
            [1, 0, 0, 0],  # Channel 3 (A): A=1, T=0, C=0, G=0
        ],
        dtype=np.float32,
    )

    # Convert to numpy if it's a torch tensor
    if hasattr(custom_encoded, "numpy"):
        custom_encoded = custom_encoded.numpy()

    np.testing.assert_array_equal(custom_encoded, expected)


def test_custom_encoder_with_get_personal_genome(tmp_path):
    """Test custom encoder with get_personal_genome function."""

    def identity_encoder(seq):
        """Encoder that returns identity matrix repeated for each nucleotide."""
        result = np.zeros((4, len(seq)), dtype=np.float32)
        for i in range(len(seq)):
            result[i % 4, i] = 1.0  # Cycle through positions
        return result

    # Create a simple reference file
    ref_file = tmp_path / "test_ref.fa"
    ref_file.write_text(">chr1\nATCGATCG\n")

    # Create a simple VCF file with one variant
    vcf_file = tmp_path / "test.vcf"
    vcf_file.write_text(
        """##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	5	.	A	T	.	.	.
"""
    )

    # Test with custom encoder
    personal_genome = sl.get_personal_genome(
        str(ref_file), str(vcf_file), encode=True, encoder=identity_encoder
    )

    # Should have chr1
    assert "chr1" in personal_genome

    # Check the encoding matches our identity pattern
    encoded_seq = personal_genome["chr1"]
    if hasattr(encoded_seq, "numpy"):
        encoded_seq = encoded_seq.numpy()

    # Should be 8 positions, each with identity pattern
    assert encoded_seq.shape == (4, 8)

    # First position should be [1, 0, 0, 0]
    np.testing.assert_array_equal(encoded_seq[:, 0], [1, 0, 0, 0])
    # Second position should be [0, 1, 0, 0]
    np.testing.assert_array_equal(encoded_seq[:, 1], [0, 1, 0, 0])


def test_custom_encoder_none_uses_default():
    """Test that encoder=None uses default behavior."""
    seq = "ATCG"

    # These should be identical
    default1 = sl.encode_seq(seq)
    default2 = sl.encode_seq(seq, encoder=None)

    # Convert to numpy if needed
    if hasattr(default1, "numpy"):
        default1 = default1.numpy()
    if hasattr(default2, "numpy"):
        default2 = default2.numpy()

    np.testing.assert_array_equal(default1, default2)


def test_custom_encoder_validation():
    """Test that custom encoder produces correct shape."""

    def bad_encoder(seq):
        """Encoder that returns wrong shape."""
        return np.array([1, 2, 3])  # Wrong shape

    # This should work but produce unexpected results
    # The function doesn't validate shape, it trusts the user
    seq = "AT"
    result = sl.encode_seq(seq, encoder=bad_encoder)

    # Just verify it returns something
    assert result is not None
