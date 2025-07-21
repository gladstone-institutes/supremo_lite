"""
Comprehensive test suite for get_pam_disrupting_personal_sequences function.

Tests PAM disruption filtering and sequence generation across multiple scenarios:
1. Variants with no nearby PAM sites
2. PAM sites at max_pam_distance boundary (edge cases)
3. Variants with multiple PAM sites at varying distances
4. Single variant disrupting multiple PAM sites
5. Different PAM sequences (NGG, TTTN) with ambiguous nucleotides
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
import warnings

import supremo_lite as sl


def create_reference_with_pams(sequence_template, pam_positions, pam_sequence="NGG"):
    """
    Create a reference sequence with PAM sites at specified positions.

    Args:
        sequence_template: Base sequence template (will be repeated/extended as needed)
        pam_positions: List of 0-based positions where PAM sites should be placed
        pam_sequence: PAM sequence to insert (default: 'NGG')

    Returns:
        Dictionary with chromosome sequence containing PAM sites
    """
    # Determine required sequence length
    max_pos = max(pam_positions) if pam_positions else 0
    min_length = max_pos + len(pam_sequence) + 200  # Add larger buffer

    # Create base sequence by repeating template, avoiding NGG/TGG patterns
    safe_template = "ATCGACT"  # Template that won't accidentally create PAM sites
    repeat_count = (min_length // len(safe_template)) + 1
    base_seq = (safe_template * repeat_count)[:min_length]
    base_seq = list(base_seq.upper())

    # Insert PAM sequences at specified positions
    for pos in pam_positions:
        for i, nucleotide in enumerate(pam_sequence.upper()):
            if pos + i < len(base_seq):
                base_seq[pos + i] = nucleotide

    return {"chr1": "".join(base_seq)}


def create_variants_dataframe(variants_list):
    """
    Create a variants DataFrame from a list of variant tuples.

    Args:
        variants_list: List of tuples (chrom, pos, ref, alt)

    Returns:
        DataFrame with standard VCF columns
    """
    return pd.DataFrame(
        [
            {"chrom": chrom, "pos": pos, "id": ".", "ref": ref, "alt": alt}
            for chrom, pos, ref, alt in variants_list
        ]
    )


class TestPAMDisruption:
    """Test PAM disruption functionality comprehensively."""

    def test_no_nearby_pam_sites(self):
        """Test variants with no nearby PAM sites return empty results."""
        # Create reference with PAM sites far from variant
        # Variant at position 100, PAM sites at positions 10 and 500
        reference = create_reference_with_pams("ATCGATCG", [10, 500], "NGG")

        # Create variant at position 100 (far from both PAM sites)
        variants = create_variants_dataframe(
            [("chr1", 100, "A", "T")]  # 1-based position
        )

        # Test with max_pam_distance=20 (not enough to reach either PAM)
        result = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
        )

        # Should return empty lists
        assert len(result["variants"]) == 0
        assert len(result["pam_intact"]) == 0
        assert len(result["pam_disrupted"]) == 0

    def test_pam_sites_at_boundary(self):
        """Test PAM sites exactly at max_pam_distance boundary."""
        # Create reference with PAM sites at specific distances from variant
        # Variant at position 100, PAM sites at positions 75, 85, 115, 125
        reference = create_reference_with_pams(
            "ATCGATCGATCGATCG", [75, 85, 115, 125], "NGG"
        )

        variants = create_variants_dataframe(
            [("chr1", 100, "A", "T")]  # 1-based position (0-based: 99)
        )

        # Test with max_pam_distance=15
        # PAM at 85: distance = |85 - 99| = 14 (within boundary)
        # PAM at 115: distance = |115 - 99| = 16 (outside boundary)
        result_15 = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=100,
            max_pam_distance=15,
            pam_sequence="NGG",
            encode=False,
        )

        # Should find the PAM at position 85 but not 115
        assert len(result_15["variants"]) == 1
        assert len(result_15["pam_intact"]) == 1
        assert len(result_15["pam_disrupted"]) == 1  # Only one PAM site within distance

        # Test with max_pam_distance=16
        # Now PAM at 115 should also be included
        result_16 = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=100,
            max_pam_distance=16,
            pam_sequence="NGG",
            encode=False,
        )

        # Should find both PAM sites at positions 85 and 115
        assert len(result_16["variants"]) == 1
        assert len(result_16["pam_intact"]) == 1
        assert len(result_16["pam_disrupted"]) == 2  # Two PAM sites within distance

    def test_multiple_pam_sites_varying_distances(self):
        """Test variants with multiple PAM sites at varying distances."""
        # Create a carefully controlled reference sequence
        # Use a 300bp sequence with specific PAM sites at known positions
        base_seq = "A" * 300  # All A's to avoid accidental PAM sites
        base_seq = list(base_seq)

        # Insert NGG at specific positions: 180, 190, 210, 220, 250
        pam_positions = [180, 190, 210, 220, 250]
        for pos in pam_positions:
            if pos + 2 < len(base_seq):
                base_seq[pos] = "N"
                base_seq[pos + 1] = "G"
                base_seq[pos + 2] = "G"

        reference = {"chr1": "".join(base_seq)}

        # Variant at position 200 (1-based), which is position 199 (0-based)
        variants = create_variants_dataframe(
            [("chr1", 200, "A", "T")]  # 1-based position (0-based: 199)
        )

        # Test with max_pam_distance=25
        # Distances from variant (199): 180->19, 190->9, 210->11, 220->21, 250->51
        # Should include PAM sites at 180, 190, 210, 220 (all ≤25)
        result = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=200,
            max_pam_distance=25,
            pam_sequence="NGG",
            encode=False,
        )

        assert len(result["variants"]) == 1
        assert len(result["pam_intact"]) == 1
        assert (
            len(result["pam_disrupted"]) == 4
        )  # Exactly four PAM sites within distance

        # Test with smaller distance
        result_small = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=200,
            max_pam_distance=15,
            pam_sequence="NGG",
            encode=False,
        )

        # Should only include PAM sites at 190, 210 (distances 9, 11)
        assert len(result_small["variants"]) == 1
        assert len(result_small["pam_intact"]) == 1
        assert len(result_small["pam_disrupted"]) == 2

    def test_single_variant_multiple_pam_disruptions(self):
        """Test single variant disrupting multiple PAM sites generates separate entries."""
        # Create a carefully controlled reference sequence
        base_seq = "A" * 200  # All A's to avoid accidental PAM sites
        base_seq = list(base_seq)

        # Insert NGG at specific positions: 95, 105, 115
        pam_positions = [95, 105, 115]
        for pos in pam_positions:
            if pos + 2 < len(base_seq):
                base_seq[pos] = "N"
                base_seq[pos + 1] = "G"
                base_seq[pos + 2] = "G"

        reference = {"chr1": "".join(base_seq)}

        # Variant at position 100 (1-based), which is position 99 (0-based)
        variants = create_variants_dataframe(
            [("chr1", 100, "A", "T")]  # 1-based position (0-based: 99)
        )

        result = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=100,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
        )

        # Should have one variant that disrupts three PAM sites
        assert len(result["variants"]) == 1
        assert len(result["pam_intact"]) == 1
        assert len(result["pam_disrupted"]) == 3  # Three separate disrupted sequences

        # Verify that each disrupted sequence has different PAM positions disrupted
        disrupted_sequences = [seq[3] for seq in result["pam_disrupted"]]

        # Each sequence should be different (different PAM sites disrupted)
        assert (
            len(set(disrupted_sequences)) == 3
        ), "Each PAM disruption should create unique sequence"

    def test_different_pam_sequences(self):
        """Test different PAM sequences (NGG vs TTTN) and ambiguous nucleotides."""
        # Test NGG PAM sequence
        reference_ngg = create_reference_with_pams("ATCGATCGATCGATCG", [90, 110], "NGG")

        # Get the actual nucleotide at position 99 (0-based) to create valid variant
        ref_seq = reference_ngg["chr1"]
        ref_nucleotide = ref_seq[99]  # 0-based position 99
        alt_nucleotide = "T" if ref_nucleotide != "T" else "A"

        variants = create_variants_dataframe(
            [("chr1", 100, ref_nucleotide, alt_nucleotide)]
        )

        result_ngg = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference_ngg,
            variants_fn=variants,
            seq_len=80,
            max_pam_distance=15,
            pam_sequence="NGG",
            encode=False,
        )

        assert len(result_ngg["variants"]) == 1
        assert len(result_ngg["pam_disrupted"]) == 2

        # Test TTTN PAM sequence (Cas12a) with different N variants
        reference_tttn1 = create_reference_with_pams(
            "ATCGATCGATCGATCG", [90], "TTTA"  # One specific case of TTTN
        )

        reference_tttn2 = create_reference_with_pams(
            "ATCGATCGATCGATCG", [110], "TTTG"  # Another specific case of TTTN
        )

        # Combine both references manually
        combined_seq = list(reference_tttn1["chr1"])
        ref2_seq = reference_tttn2["chr1"]
        for i in range(110, min(110 + 4, len(ref2_seq), len(combined_seq))):
            combined_seq[i] = ref2_seq[i]

        combined_reference = {"chr1": "".join(combined_seq)}

        result_tttn = sl.get_pam_disrupting_personal_sequences(
            reference_fn=combined_reference,
            variants_fn=variants,
            seq_len=80,
            max_pam_distance=15,
            pam_sequence="TTTN",  # Should match both TTTA and TTTG
            encode=False,
        )

        assert len(result_tttn["variants"]) == 1
        assert len(result_tttn["pam_disrupted"]) == 2

    def test_ambiguous_nucleotides_in_pam(self):
        """Test PAM matching with ambiguous nucleotides like 'N'."""
        # Create reference with sequences that should match NGG pattern
        base_seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 10
        base_seq = list(base_seq)

        # Insert specific sequences at known positions
        # Position 90: AGG (should match NGG)
        # Position 110: TGG (should match NGG)
        # Position 130: ATC (should NOT match NGG)
        pam_sequences = ["AGG", "TGG", "ATC"]
        positions = [90, 110, 130]

        for pos, pam in zip(positions, pam_sequences):
            for i, nucleotide in enumerate(pam):
                if pos + i < len(base_seq):
                    base_seq[pos + i] = nucleotide

        reference = {"chr1": "".join(base_seq)}

        variants = create_variants_dataframe(
            [("chr1", 100, "A", "T")]  # 1-based position (0-based: 99)
        )

        result = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=100,
            max_pam_distance=15,
            pam_sequence="NGG",  # N should match A and T
            encode=False,
        )

        # Should match AGG and TGG but not ATC
        assert len(result["variants"]) == 1
        assert len(result["pam_disrupted"]) == 2  # Only AGG and TGG match NGG pattern

    def test_encoded_vs_string_output(self):
        """Test that encoded and string outputs are consistent."""
        reference = create_reference_with_pams("ATCGATCGATCGATCG", [95, 105], "NGG")

        variants = create_variants_dataframe([("chr1", 100, "A", "T")])

        # Get string output
        result_str = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Get encoded output
        result_enc = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=True,
        )

        # Should have same number of results
        assert len(result_str["variants"]) == len(result_enc["variants"])
        assert len(result_str["pam_intact"]) == len(result_enc["pam_intact"])
        assert len(result_str["pam_disrupted"]) == len(result_enc["pam_disrupted"])

        # Check that encoded sequences have correct shape (could be numpy or torch)
        if result_enc["pam_intact"]:
            intact_seq = result_enc["pam_intact"][0][3]  # First sequence
            # Could be numpy array or torch tensor depending on availability
            assert hasattr(intact_seq, "shape")
            assert intact_seq.shape == (50, 4)  # seq_len x 4 nucleotides

        if result_enc["pam_disrupted"]:
            disrupted_seq = result_enc["pam_disrupted"][0][3]  # First sequence
            assert hasattr(disrupted_seq, "shape")
            assert disrupted_seq.shape == (50, 4)

    def test_edge_case_sequence_boundaries(self):
        """Test edge cases where variants are near sequence boundaries."""
        # Create a short reference sequence
        reference = {"chr1": "ATCGATCGNGGTATCGATCGNGGAAA"}

        # Test variant very close to start
        variants_start = create_variants_dataframe(
            [("chr1", 2, "T", "A")]  # Near start of sequence
        )

        result_start = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_start,
            seq_len=20,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Should handle boundary conditions gracefully
        # NGG is at position 8, variant at position 1 (0-based), distance = 7
        assert len(result_start["variants"]) == 1

        # Test variant very close to end
        variants_end = create_variants_dataframe(
            [("chr1", 24, "A", "T")]  # Near end of sequence
        )

        result_end = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_end,
            seq_len=20,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Should handle boundary conditions gracefully
        # NGG is at position 19, variant at position 23 (0-based), distance = 4
        assert len(result_end["variants"]) == 1

    def test_no_pam_sites_in_sequence(self):
        """Test sequence with no PAM sites at all."""
        # Create reference without any PAM sites
        reference = {"chr1": "ATCGATCGATCGATCGATCGATCGATCGATCG"}

        variants = create_variants_dataframe([("chr1", 15, "A", "T")])

        result = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=30,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
        )

        # Should return empty results
        assert len(result["variants"]) == 0
        assert len(result["pam_intact"]) == 0
        assert len(result["pam_disrupted"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
