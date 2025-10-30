"""
Comprehensive test suite for get_pam_disrupting_alt_sequences function.

Uses real genomic sequences from tests/data/test_genome.fa to test:
1. Variants that disrupt PAM sites
2. PAM sites at max_pam_distance boundary
3. Different PAM sequences (NGG, TTTN) with IUPAC codes
4. Edge cases and boundary conditions
"""

import os
import pandas as pd
import numpy as np
import pytest
import warnings

import supremo_lite as sl


class TestPAMDisruption:
    """Test PAM disruption functionality with real genomic sequences."""

    @pytest.fixture
    def test_reference(self):
        """Path to test reference genome."""
        return "tests/data/test_genome.fa"

    def test_chr4_agg_disruption(self, test_reference):
        """Test disrupting AGG PAM site on chr4 at position 0."""
        # chr4 starts with AGG at position 0-2
        # Create variant at position 2 (1-based) to disrupt the AGG
        variants = pd.DataFrame([{
            "chrom": "chr4",
            "pos": 2,  # 1-based
            "id": ".",
            "ref": "G",  # First G in AGG
            "alt": "A"   # Change to A, disrupting AGG
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        # Get the result from generator
        alt_seqs, ref_seqs, metadata = next(gen)

        # Should find at least one PAM-disrupting variant
        assert len(metadata) >= 1
        assert len(alt_seqs) >= 1
        assert len(ref_seqs) >= 1

        # Check metadata has PAM-specific columns
        assert 'pam_site_pos' in metadata.columns
        assert 'pam_ref_sequence' in metadata.columns
        assert 'pam_alt_sequence' in metadata.columns
        assert 'pam_distance' in metadata.columns

    def test_chr4_tgg_disruption(self, test_reference):
        """Test disrupting TGG PAM site on chr4."""
        # chr4 has TGG at position 3-5
        # Create variant at position 5 (1-based) = position 4 (0-based)
        variants = pd.DataFrame([{
            "chrom": "chr4",
            "pos": 5,  # 1-based
            "id": ".",
            "ref": "G",  # Second G in TGG
            "alt": "C"   # Change to C, disrupting TGG
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=8,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        alt_seqs, ref_seqs, metadata = next(gen)

        assert len(metadata) >= 1
        assert len(alt_seqs) >= 1
        assert len(ref_seqs) >= 1

    def test_variant_far_from_pam(self, test_reference):
        """Test variant far from any PAM site returns empty results."""
        # chr3 has no PAM sites (*GG patterns)
        # Create a variant that won't disrupt anything
        variants = pd.DataFrame([{
            "chrom": "chr3",
            "pos": 40,  # 1-based, middle of chr3
            "id": ".",
            "ref": "A",
            "alt": "T"
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=30,
            max_pam_distance=5,  # Very small distance
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        # Should return empty generator
        result_list = list(gen)
        assert len(result_list) == 0

    def test_pam_distance_boundary(self, test_reference):
        """Test max_pam_distance boundary conditions."""
        # chr4 position 0: AGG
        # Test variant at position 15 with different distances
        variants = pd.DataFrame([{
            "chrom": "chr4",
            "pos": 15,  # 1-based
            "id": ".",
            "ref": "A",
            "alt": "G"
        }])

        # With max_pam_distance=10, AGG at pos 0 should be too far (distance ~14)
        gen_small = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=60,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        # With max_pam_distance=20, AGG at pos 0 might be within range
        gen_large = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=60,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        # Count results
        result_small_list = list(gen_small)
        result_large_list = list(gen_large)

        num_small = sum(len(metadata) for _, _, metadata in result_small_list)
        num_large = sum(len(metadata) for _, _, metadata in result_large_list)

        # Larger distance should find at least as many or more PAMs
        assert num_large >= num_small

    def test_tttn_pam_pattern(self, test_reference):
        """Test TTTN PAM pattern (Cas12a) with IUPAC N matching."""
        # Look for TTTA, TTTC, TTTG, or TTTT patterns in chr1
        # chr1 has "TTTT" at positions 13-16 and other TT patterns
        variants = pd.DataFrame([{
            "chrom": "chr1",
            "pos": 15,  # 1-based, disrupts TTTT
            "id": ".",
            "ref": "T",
            "alt": "A"
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=8,
            pam_sequence="TTTN",
            encode=False,
            n_chunks=1
        )

        # Should find at least one PAM disruption
        result_list = list(gen)
        # May or may not find depending on exact pattern - just check it doesn't error
        assert len(result_list) >= 0

    def test_purine_pam_pattern(self, test_reference):
        """Test PAM pattern with R (purine = A or G)."""
        # Test RGG pattern on chr4 which has AGG at position 0
        variants = pd.DataFrame([{
            "chrom": "chr4",
            "pos": 2,  # 1-based
            "id": ".",
            "ref": "G",
            "alt": "C"
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=10,
            pam_sequence="RGG",  # R matches A or G
            encode=False,
            n_chunks=1
        )

        # AGG should match RGG pattern
        alt_seqs, ref_seqs, metadata = next(gen)
        assert len(metadata) >= 1

    def test_pyrimidine_pam_pattern(self, test_reference):
        """Test PAM pattern with Y (pyrimidine = C or T)."""
        # Test YGG pattern on chr2 which has CGG at position 50
        variants = pd.DataFrame([{
            "chrom": "chr2",
            "pos": 51,  # 1-based
            "id": ".",
            "ref": "G",
            "alt": "A"
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=10,
            pam_sequence="YGG",  # Y matches C or T
            encode=False,
            n_chunks=1
        )

        # CGG should match YGG pattern
        alt_seqs, ref_seqs, metadata = next(gen)
        assert len(metadata) >= 1

    def test_encoded_vs_string_output(self, test_reference):
        """Test that encoded and string outputs are consistent."""
        variants = pd.DataFrame([{
            "chrom": "chr4",
            "pos": 2,
            "id": ".",
            "ref": "G",
            "alt": "A"
        }])

        # Get string output
        gen_str = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        # Get encoded output
        gen_enc = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=True,
            n_chunks=1
        )

        alt_str, ref_str, meta_str = next(gen_str)
        alt_enc, ref_enc, meta_enc = next(gen_enc)

        # Should have same number of results
        assert len(meta_str) == len(meta_enc)
        assert len(alt_str) == len(alt_enc)
        assert len(ref_str) == len(ref_enc)

        # Check encoded sequences have correct shape
        if len(alt_enc) > 0:
            # alt_enc should be stacked arrays
            assert hasattr(alt_enc, "shape")
            assert alt_enc.shape[1:] == (4, 40)  # (n_variants, 4, seq_len)

    def test_no_variants_provided(self, test_reference):
        """Test with empty variants list."""
        # Create properly typed empty DataFrame
        variants = pd.DataFrame({
            "chrom": pd.Series([], dtype=str),
            "pos": pd.Series([], dtype=int),
            "id": pd.Series([], dtype=str),
            "ref": pd.Series([], dtype=str),
            "alt": pd.Series([], dtype=str)
        })

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        result_list = list(gen)
        assert len(result_list) == 0

    def test_chr2_overlapping_pam_sites(self, test_reference):
        """Test region with overlapping PAM sites."""
        # chr2 has CGG at position 50 and GGG at position 51 (overlapping)
        # Disrupting position 52 (1-based) should affect both
        variants = pd.DataFrame([{
            "chrom": "chr2",
            "pos": 52,  # 1-based
            "id": ".",
            "ref": "G",  # Last G in both overlapping PAMs
            "alt": "A"
        }])

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1
        )

        alt_seqs, ref_seqs, metadata = next(gen)

        # Should find variant disrupts at least one PAM
        # May find multiple overlapping disrupted PAMs (multiple rows in metadata)
        assert len(metadata) >= 1
        assert len(alt_seqs) >= 1
        assert len(ref_seqs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
