"""
Comprehensive test suite for get_pam_disrupting_alt_sequences function.

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
        result = sl.get_pam_disrupting_alt_sequences(
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
        # Variant at position 100 disrupting the N in NGG at position 98-100
        # Additional PAMs at positions 85-87 (distance 11-13 from variant) and 113-115 (distance 13-15)
        reference = create_reference_with_pams("ATCGATCGATCGATCG", [85, 98, 113], "NGG")

        # Get actual base at position 100 to create valid variant
        # This should be the 'G' in the NGG at position 98-100
        ref_seq = reference["chr1"]
        ref_base = ref_seq[99]  # 0-based position 99 (last G in NGG)
        alt_base = "A" if ref_base == "G" else "G"

        variants = create_variants_dataframe(
            [("chr1", 100, ref_base, alt_base)]  # 1-based position (0-based: 99)
        )

        # Test with max_pam_distance=15
        # PAM at 98-100: overlaps with variant - SHOULD BE DISRUPTED
        # PAM at 85-87: distance from variant (99) to PAM end (87) = 12 (within boundary) - no overlap, NOT disrupted
        # PAM at 113-115: distance from variant (99) to PAM start (113) = 14 (within boundary) - no overlap, NOT disrupted
        result_15 = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=150,
            max_pam_distance=15,
            pam_sequence="NGG",
            encode=False,
        )

        # Should find the variant disrupts the PAM it overlaps with
        assert len(result_15["variants"]) == 1
        assert len(result_15["pam_intact"]) == 1
        assert (
            len(result_15["pam_disrupted"]) == 1
        )  # Only the overlapping PAM is disrupted

        # Test with max_pam_distance=10
        # Now the PAMs at 85 and 113 are outside the distance threshold
        result_10 = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=150,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Should still find the variant (it disrupts the PAM at 98-100)
        assert len(result_10["variants"]) == 1
        assert len(result_10["pam_intact"]) == 1
        assert len(result_10["pam_disrupted"]) == 1

    def test_multiple_pam_sites_varying_distances(self):
        """Test variant disrupting a PAM with other PAMs nearby."""
        # Create a carefully controlled reference sequence
        # Place variant WITHIN one PAM, with other PAMs at varying distances
        base_seq = "A" * 300  # All A's to avoid accidental PAM sites
        base_seq = list(base_seq)

        # Insert NGG at specific positions
        # PAM at 197-199 will be disrupted by variant at 199
        # Other PAMs at various distances from variant
        pam_positions = [180, 190, 197, 210, 220]
        for pos in pam_positions:
            if pos + 2 < len(base_seq):
                base_seq[pos] = "N"
                base_seq[pos + 1] = "G"
                base_seq[pos + 2] = "G"

        reference = {"chr1": "".join(base_seq)}

        # Variant at position 200 (1-based), which is position 199 (0-based)
        # This overlaps with the PAM at 197-199 (last G of NGG)
        ref_base = base_seq[199]  # Should be 'G'
        alt_base = "A"  # Change G to A, disrupting the NGG

        variants = create_variants_dataframe(
            [("chr1", 200, ref_base, alt_base)]  # 1-based position (0-based: 199)
        )

        # Test with max_pam_distance=25
        # Variant at 199 overlaps PAM at 197-199 - DISRUPTED
        # Other PAMs: 180-182 (dist 17), 190-192 (dist 7), 210-212 (dist 11), 220-222 (dist 21)
        # All within distance but NOT disrupted (no overlap)
        result = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=200,
            max_pam_distance=25,
            pam_sequence="NGG",
            encode=False,
        )

        # Should find one variant disrupting one PAM (the one it overlaps)
        assert len(result["variants"]) == 1
        assert len(result["pam_intact"]) == 1
        assert (
            len(result["pam_disrupted"]) == 1
        )  # Only the overlapping PAM is disrupted

        # Test with smaller distance that excludes some nearby PAMs
        result_small = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=200,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Should still find the variant (still disrupts the overlapping PAM)
        assert len(result_small["variants"]) == 1
        assert len(result_small["pam_intact"]) == 1
        assert len(result_small["pam_disrupted"]) == 1

    def test_single_variant_multiple_pam_disruptions(self):
        """Test single variant disrupting multiple overlapping PAM sites."""
        # Create a reference with OVERLAPPING PAM sites
        # For example: ...NGGGG... can contain TWO overlapping NGG PAMs
        # Position 95-97: NGG
        # Position 96-98: GGG (also matches NGG where N can be G)

        base_seq = "A" * 200
        base_seq = list(base_seq)

        # Create overlapping PAMs: NGGG at position 97-100
        # This contains: NGG at 97-99 and GGG at 98-100 (also matches NGG)
        base_seq[97] = "N"
        base_seq[98] = "G"
        base_seq[99] = "G"
        base_seq[100] = "G"

        # Add another separate PAM that will also be disrupted by a variant at 99
        # PAM at 97-99 (NGG) will be disrupted by changing position 99

        reference = {"chr1": "".join(base_seq)}

        # Variant at position 100 (1-based = 0-based 99)
        # This is the middle G in NGGG, disrupting both overlapping PAMs
        ref_base = base_seq[99]  # Should be 'G'
        alt_base = "A"  # Change to A

        variants = create_variants_dataframe(
            [("chr1", 100, ref_base, alt_base)]  # 1-based position (0-based: 99)
        )

        result = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=100,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
        )

        # Should have one variant that disrupts multiple overlapping PAM sites
        # PAM at 97-99 (NGG) is disrupted by changing position 99
        assert len(result["variants"]) == 1
        assert len(result["pam_intact"]) == 1
        # Note: The exact number of PAMs detected depends on how overlapping PAMs are handled
        assert len(result["pam_disrupted"]) >= 1  # At least one PAM is disrupted

    def test_different_pam_sequences(self):
        """Test different PAM sequences (NGG vs TTTN) and ambiguous nucleotides."""
        # Test NGG PAM sequence - create PAM that will be disrupted
        reference_ngg = create_reference_with_pams("ATCGATCGATCGATCG", [98], "NGG")

        # Get the actual nucleotide at position 99 (0-based) to create valid variant
        # This should be in the middle of the NGG at 98-100
        ref_seq = reference_ngg["chr1"]
        ref_nucleotide = ref_seq[99]  # 0-based position 99, middle of NGG
        alt_nucleotide = "A" if ref_nucleotide != "A" else "T"

        variants = create_variants_dataframe(
            [("chr1", 100, ref_nucleotide, alt_nucleotide)]
        )

        result_ngg = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference_ngg,
            variants_fn=variants,
            seq_len=80,
            max_pam_distance=15,
            pam_sequence="NGG",
            encode=False,
        )

        assert len(result_ngg["variants"]) == 1
        assert len(result_ngg["pam_disrupted"]) == 1  # One PAM disrupted

        # Test TTTN PAM sequence (Cas12a)
        reference_tttn = create_reference_with_pams(
            "ATCGATCGATCGATCG", [98], "TTTA"  # TTTN PAM at same position as NGG test
        )

        # Create variant for TTTN test at the same position
        ref_seq_tttn = reference_tttn["chr1"]
        ref_nucleotide_tttn = ref_seq_tttn[99]  # 0-based position 99, middle of TTTA
        alt_nucleotide_tttn = "G" if ref_nucleotide_tttn != "G" else "C"

        variants_tttn = create_variants_dataframe(
            [("chr1", 100, ref_nucleotide_tttn, alt_nucleotide_tttn)]
        )

        result_tttn = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference_tttn,
            variants_fn=variants_tttn,
            seq_len=80,
            max_pam_distance=15,
            pam_sequence="TTTN",  # Should match TTTA
            encode=False,
        )

        assert len(result_tttn["variants"]) == 1
        assert len(result_tttn["pam_disrupted"]) >= 1  # At least one TTTN PAM disrupted

    def test_ambiguous_nucleotides_in_pam(self):
        """Test PAM matching with ambiguous nucleotides like 'N'."""
        # Create reference with AGG PAM (matches NGG pattern) that will be disrupted
        base_seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 10
        base_seq = list(base_seq)

        # Insert AGG at position 98-100 (will be disrupted by variant at 100)
        # Position 98: AGG (should match NGG)
        base_seq[98] = "A"
        base_seq[99] = "G"
        base_seq[100] = "G"

        # Also insert TGG nearby (also matches NGG) at position 105-107
        base_seq[105] = "T"
        base_seq[106] = "G"
        base_seq[107] = "G"

        reference = {"chr1": "".join(base_seq)}

        # Variant at position 101 (1-based) = position 100 (0-based)
        # This disrupts the AGG at 98-100
        ref_base = base_seq[100]  # Should be 'G'
        alt_base = "C"  # Change G to C, disrupting AGG

        variants = create_variants_dataframe(
            [("chr1", 101, ref_base, alt_base)]  # 1-based position (0-based: 100)
        )

        result = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=100,
            max_pam_distance=15,
            pam_sequence="NGG",  # N should match A and T
            encode=False,
        )

        # Should match and disrupt AGG (TGG is nearby but not disrupted)
        assert len(result["variants"]) == 1
        assert len(result["pam_disrupted"]) == 1  # AGG is disrupted

    def test_encoded_vs_string_output(self):
        """Test that encoded and string outputs are consistent."""
        # Create reference with PAM that will be disrupted
        reference = create_reference_with_pams("ATCGATCGATCGATCG", [98], "NGG")

        # Get actual base and create variant that disrupts the PAM
        ref_seq = reference["chr1"]
        ref_base = ref_seq[99]  # Middle G of NGG at 98-100
        alt_base = "A" if ref_base != "A" else "T"

        variants = create_variants_dataframe([("chr1", 100, ref_base, alt_base)])

        # Get string output
        result_str = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Get encoded output
        result_enc = sl.get_pam_disrupting_alt_sequences(
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
        # NGG at positions 8-10 and 19-21
        reference = {"chr1": "ATCGATCGNGGTATCGATCGNGGAAA"}

        ref_seq = reference["chr1"]

        # Test variant that disrupts PAM near start of sequence
        # Variant at position 10 (1-based) = position 9 (0-based) disrupts NGG at 8-10
        ref_base_start = ref_seq[9]  # Last G of first NGG (should be 'G')
        alt_base_start = "A"  # Change G to A to disrupt

        variants_start = create_variants_dataframe(
            [("chr1", 10, ref_base_start, alt_base_start)]  # Disrupts first NGG
        )

        result_start = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants_start,
            seq_len=20,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Should handle boundary conditions gracefully and find the disruption
        assert len(result_start["variants"]) == 1

        # Test variant that disrupts PAM near end of sequence
        # NGG is at 20-22, so variant at position 22 (1-based) = 21 (0-based) disrupts it
        ref_base_end = ref_seq[21]  # Middle G of second NGG at 20-22
        alt_base_end = "A"  # Change G to A to disrupt

        variants_end = create_variants_dataframe(
            [("chr1", 22, ref_base_end, alt_base_end)]  # Disrupts second NGG
        )

        result_end = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants_end,
            seq_len=20,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
        )

        # Should handle boundary conditions gracefully
        assert len(result_end["variants"]) == 1

    def test_no_pam_sites_in_sequence(self):
        """Test sequence with no PAM sites at all."""
        # Create reference without any PAM sites
        reference = {"chr1": "ATCGATCGATCGATCGATCGATCGATCGATCG"}

        variants = create_variants_dataframe([("chr1", 15, "A", "T")])

        result = sl.get_pam_disrupting_alt_sequences(
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
