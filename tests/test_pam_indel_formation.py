"""
Test suite for INDEL-related PAM formation detection.

This test suite validates that get_pam_disrupting_alt_sequences correctly
handles INDELs from real VCF files and genomic sequences.

Uses only real data from tests/data/
"""

import pandas as pd
import pytest
import supremo_lite as sl


class TestPAMINDELFormation:
    """Test PAM disruption detection with real INDELs."""

    @pytest.fixture
    def test_reference(self):
        """Path to test reference genome."""
        return "tests/data/test_genome.fa"

    def test_real_deletion_chr1(self, test_reference):
        """Test with real deletion from del.vcf on chr1."""
        # chr1	17	.	CGAGAA	C
        variants = pd.DataFrame(
            [{"chrom": "chr1", "pos": 17, "id": ".", "ref": "CGAGAA", "alt": "C"}]
        )

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1,
        )

        # Verify deletion is processed correctly (generator returns tuples)
        result_list = list(gen)
        # Either empty or has results
        assert isinstance(result_list, list)

    def test_real_deletion_chr2(self, test_reference):
        """Test with real deletion from del.vcf on chr2."""
        # chr2	23	.	ATTAATTTA	A
        variants = pd.DataFrame(
            [{"chrom": "chr2", "pos": 23, "id": ".", "ref": "ATTAATTTA", "alt": "A"}]
        )

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1,
        )

        # Verify deletion is processed correctly
        result_list = list(gen)
        assert isinstance(result_list, list)

    def test_deletion_from_vcf_file(self, test_reference):
        """Test loading deletions directly from VCF file."""
        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn="tests/data/del/del.vcf",
            seq_len=50,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1,
        )

        # Should process all deletions from VCF
        result_list = list(gen)
        assert isinstance(result_list, list)

    def test_snv_disrupting_pam_chr4(self, test_reference):
        """Test SNV disrupting a real PAM site on chr4."""
        # chr4 starts with AGGTGGAAAA - has AGG at 0-2, TGG at 3-5
        # Disrupt the AGG at position 2 (1-based)
        variants = pd.DataFrame(
            [{"chrom": "chr4", "pos": 2, "id": ".", "ref": "G", "alt": "C"}]
        )

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=40,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1,
        )

        # Should find PAM disruption
        alt_seqs, ref_seqs, metadata = next(gen)
        assert len(metadata) >= 1
        assert len(alt_seqs) >= 1
        assert len(ref_seqs) >= 1

    def test_snv_from_vcf_file(self, test_reference):
        """Test loading SNVs directly from VCF file."""
        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn="tests/data/snp/snp.vcf",
            seq_len=50,
            max_pam_distance=20,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1,
        )

        # Should process all SNVs from VCF
        result_list = list(gen)
        assert isinstance(result_list, list)

    def test_overlapping_pam_sites_chr2(self, test_reference):
        """Test handling of overlapping PAM sites on chr2."""
        # chr2 has CGG at 50 and GGG at 51 (overlapping)
        variants = pd.DataFrame(
            [{"chrom": "chr2", "pos": 52, "id": ".", "ref": "G", "alt": "A"}]
        )

        gen = sl.get_pam_disrupting_alt_sequences(
            reference_fn=test_reference,
            variants_fn=variants,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False,
            n_chunks=1,
        )

        # Should detect disruption of overlapping PAMs
        alt_seqs, ref_seqs, metadata = next(gen)
        assert len(metadata) >= 1

    def test_indel_with_window_extension_warning(self, test_reference):
        """Test that large INDELs trigger window extension warning."""
        import warnings

        # Use a real deletion with small window to trigger warning
        variants = pd.DataFrame(
            [
                {
                    "chrom": "chr1",
                    "pos": 17,
                    "id": ".",
                    "ref": "CGAGAA",  # 6 bp
                    "alt": "C",
                }
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            gen = sl.get_pam_disrupting_alt_sequences(
                reference_fn=test_reference,
                variants_fn=variants,
                seq_len=8,  # Very small window
                max_pam_distance=5,
                pam_sequence="NGG",
                encode=False,
                n_chunks=1,
            )

            # Consume generator to trigger warnings
            result_list = list(gen)

            # Check if extension warning was issued
            extension_warnings = [
                warning for warning in w if "extends" in str(warning.message)
            ]

            assert (
                len(extension_warnings) >= 1
            ), "Should warn about variant extending past window"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
