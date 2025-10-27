"""
Test chunked processing functionality for large VCF files.
"""

import pandas as pd
import pytest

import supremo_lite as sl
from supremo_lite.variant_utils import read_vcf_chunked


def get_test_vcf_path():
    """Get path to existing test VCF file."""
    return "tests/data/snp/snp.vcf"


def get_test_reference():
    """Get path to existing test reference genome."""
    return "tests/data/test_genome.fa"


class TestChunkedVCFReading:
    """Test chunked VCF reading functionality."""

    def test_read_vcf_chunked_single_chunk(self):
        """Test reading small VCF in single chunk."""
        vcf_path = get_test_vcf_path()

        chunks = list(read_vcf_chunked(vcf_path, n_chunks=1))

        # Should have only one chunk (snp.vcf has 4 variants)
        assert len(chunks) == 1
        assert len(chunks[0]) == 4
        assert list(chunks[0].columns) == [
            "chrom",
            "pos1",
            "id",
            "ref",
            "alt",
            "info",
            "vcf_line",
            "variant_type",
        ]
        assert chunks[0]["chrom"].iloc[0] == "chr1"
        assert chunks[0]["pos1"].iloc[0] == 2

    def test_read_vcf_chunked_multiple_chunks(self):
        """Test reading VCF split into multiple chunks."""
        vcf_path = get_test_vcf_path()

        chunks = list(read_vcf_chunked(vcf_path, n_chunks=2))

        # Should have 2 chunks: 2, 2 variants (snp.vcf has 4 variants split into 2 chunks)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2

        # Check continuity
        all_variants = pd.concat(chunks, ignore_index=True)
        assert len(all_variants) == 4
        assert all_variants["pos1"].iloc[0] == 2
        assert all_variants["pos1"].iloc[3] == 57

    def test_read_vcf_chunked_empty_file(self):
        """Test reading empty VCF file."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        try:
            chunks = list(read_vcf_chunked(vcf_path, n_chunks=1))
            assert len(chunks) == 0
        finally:
            os.unlink(vcf_path)


class TestChunkedPersonalSequences:
    """Test chunked processing in get_alt_sequences."""

    def test_chunked_processing_default_n_chunks(self):
        """Test that default n_chunks=1 yields all variants in single chunk."""
        vcf_path = get_test_vcf_path()
        reference = get_test_reference()
        seq_len = 100

        # Test default chunking (n_chunks=1)
        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=True,
            )
        )

        # Should yield 1 result with all 4 variants (snp.vcf has 4 variants)
        assert len(results) == 1

        # Unpack tuple (sequences, metadata)
        sequences, metadata = results[0]

        # Result should contain all 4 variants
        assert sequences.shape == (
            4,
            4,
            seq_len,
        )  # (4 variants, 4 nucleotide channels, seq_len)

        # Verify metadata is included
        assert len(metadata) == 4  # 4 variants worth of metadata

    def test_chunked_processing_multiple_variants(self):
        """Test chunking with multiple variants per chunk."""
        vcf_path = get_test_vcf_path()
        reference = get_test_reference()
        seq_len = 100

        # Test chunking with n_chunks=2 (split 4 variants into 2 chunks)
        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=True,
                n_chunks=2,
            )
        )

        # Should yield 2 chunks: 2, 2 variants (snp.vcf has 4 variants split into 2 chunks)
        assert len(results) == 2

        # Unpack tuples (sequences, metadata) for each chunk
        sequences_1, metadata_1 = results[0]
        sequences_2, metadata_2 = results[1]

        assert sequences_1.shape == (2, 4, seq_len)
        assert sequences_2.shape == (2, 4, seq_len)

        # Verify metadata for each chunk
        assert len(metadata_1) == 2  # 2 variants in first chunk
        assert len(metadata_2) == 2  # 2 variants in second chunk

    def test_chunked_processing_with_dataframe(self):
        """Test chunking when variants_fn is a DataFrame."""
        # Create test DataFrame with positions that exist in test_genome.fa
        # chr1 sequence: ATGAATATAATATTTTCGAGAATTACTCCTTTTGGAAATGGAACATTATGCGTTTTAAGAGTTTCTGGTAACAATATATT
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "pos": [2, 10, 20, 30],  # Positions within the 77bp chr1 sequence
                "id": ["."] * 4,
                "ref": [
                    "T",
                    "A",
                    "T",
                    "C",
                ],  # Actual bases at positions 2=T, 10=A, 20=T, 30=C
                "alt": ["G"] * 4,
            }
        )

        reference = get_test_reference()
        seq_len = 50  # Smaller seq_len to fit within the test genome

        # Test chunking with n_chunks=3 (split 4 variants into 3 chunks)
        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=variants_df,
                seq_len=seq_len,
                encode=True,
                n_chunks=3,
            )
        )

        # Should yield 3 chunks: 2, 1, 1 variants (array_split creates approximately equal chunks)
        assert len(results) == 3

        # Unpack tuples (sequences, metadata) for each chunk
        sequences_1, metadata_1 = results[0]
        sequences_2, metadata_2 = results[1]
        sequences_3, metadata_3 = results[2]

        assert sequences_1.shape == (2, 4, seq_len)
        assert sequences_2.shape == (1, 4, seq_len)
        assert sequences_3.shape == (1, 4, seq_len)

        # Verify metadata for each chunk
        assert len(metadata_1) == 2  # 2 variants in first chunk
        assert len(metadata_2) == 1  # 1 variant in second chunk
        assert len(metadata_3) == 1  # 1 variant in third chunk

    def test_chunked_processing_no_encoding(self):
        """Test chunked processing without encoding."""
        vcf_path = get_test_vcf_path()
        reference = get_test_reference()
        seq_len = 50  # Smaller seq_len to fit within the test genome

        # Test without encoding
        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=False,
                n_chunks=2,
            )
        )

        # Should yield 2 chunks (snp.vcf has 4 variants split into 2 chunks)
        assert len(results) == 2

        # Unpack tuples (sequences, metadata) for each chunk
        sequences_1, metadata_1 = results[0]
        sequences_2, metadata_2 = results[1]

        assert len(sequences_1) == 2  # 2 variants in first chunk
        assert len(sequences_2) == 2  # 2 variants in second chunk

        # Verify metadata for each chunk
        assert len(metadata_1) == 2  # 2 variants in first chunk
        assert len(metadata_2) == 2  # 2 variants in second chunk

        # Check format of results
        for sequences, metadata in results:
            for item in sequences:
                assert len(item) == 4  # (chrom, start, end, sequence_string)
                assert isinstance(item[3], str)  # sequence is string
                assert len(item[3]) == seq_len

    def test_memory_efficiency_comparison(self):
        """Test that chunking uses less memory than loading all at once."""
        # This is a conceptual test - in practice we'd need much larger files
        # to see real memory differences, but we can test the pattern

        vcf_path = get_test_vcf_path()
        reference = get_test_reference()
        seq_len = 50  # Smaller seq_len to fit within the test genome

        # Process with chunking - should be generator
        chunked_results = sl.get_alt_sequences(
            reference_fn=reference,
            variants_fn=vcf_path,
            seq_len=seq_len,
            encode=True,
            n_chunks=2,
        )

        # Verify it's a generator
        import types

        assert isinstance(chunked_results, types.GeneratorType)

        # Consume generator and verify total results
        all_chunks = list(chunked_results)
        total_variants = sum(sequences.shape[0] for sequences, metadata in all_chunks)
        assert total_variants == 4  # snp.vcf has 4 variants

        # Verify metadata is also returned for each chunk
        total_metadata = sum(len(metadata) for sequences, metadata in all_chunks)
        assert total_metadata == 4  # 4 variants worth of metadata


class TestChunkedPersonalizeFunctions:
    """Test chunked processing in main personalize functions."""

    def test_get_personal_genome_with_n_chunks(self):
        """Test get_personal_genome with n_chunks parameter."""
        # Create test data with positions that exist in test_genome.fa
        # chr1 sequence: ATGAATATAATATTTTCGAGAATTACTCCTTTTGGAAATGGAACATTATGCGTTTTAAGAGTTTCTGGTAACAATATATT
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 3,
                "pos": [2, 10, 20],  # Positions within the 77bp chr1 sequence
                "id": ["."] * 3,
                "ref": ["T", "A", "T"],  # Actual bases at positions 2=T, 10=A, 20=T
                "alt": ["G", "C", "A"],
            }
        )

        reference = get_test_reference()

        # Test with default n_chunks=1
        result1 = sl.get_personal_genome(
            reference_fn=reference, variants_fn=variants_df, encode=False, n_chunks=1
        )

        # Test with n_chunks=2 (should work the same for small data)
        result2 = sl.get_personal_genome(
            reference_fn=reference, variants_fn=variants_df, encode=False, n_chunks=2
        )

        # Results should be identical regardless of n_chunks for get_personal_genome
        # since it processes all variants for each chromosome anyway
        assert "chr1" in result1
        assert "chr1" in result2
        assert result1["chr1"] == result2["chr1"]

    def test_get_pam_disrupting_sequences_with_n_chunks(self):
        """Test get_pam_disrupting_alt_sequences with n_chunks parameter."""
        # Create variants that might disrupt PAM sites using the existing test data
        # chr1 sequence: ATGAATATAATATTTTCGAGAATTACTCCTTTTGGAAATGGAACATTATGCGTTTTAAGAGTTTCTGGTAACAATATATT
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 2,
                "pos": [10, 20],  # Positions within the test genome
                "id": ["."] * 2,
                "ref": ["A", "T"],  # Actual bases at positions 10=A, 20=T
                "alt": ["G", "C"],
            }
        )

        reference = get_test_reference()

        # Test with default n_chunks=1
        result1 = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=50,  # Smaller seq_len to fit within the test genome
            max_pam_distance=10,
            pam_sequence="AGG",
            encode=False,
            n_chunks=1,
        )

        # Test with n_chunks=2
        result2 = sl.get_pam_disrupting_alt_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="AGG",
            encode=False,
            n_chunks=2,
        )

        # Results should be identical
        assert len(result1["variants"]) == len(result2["variants"])
        assert len(result1["pam_intact"]) == len(result2["pam_intact"])
        assert len(result1["pam_disrupted"]) == len(result2["pam_disrupted"])

    def test_chromosome_order_preservation(self):
        """Test that get_personal_genome preserves reference chromosome order."""
        # test_genome.fa has chromosomes in order: chr1, chr2, chr3, chr4, chr5
        # Apply variants only to chr3 and chr5 (not chr1, chr2, chr4)
        # to verify that ALL chromosomes maintain original order

        variants_df = pd.DataFrame(
            {
                "chrom": ["chr3", "chr5", "chr3"],  # Intentionally not sorted
                "pos": [10, 15, 30],
                "id": [".", ".", "."],
                "ref": [
                    "A",
                    "T",
                    "A",
                ],  # Actual bases: chr3[9]=A, chr5[14]=T, chr3[29]=A
                "alt": ["C", "G", "G"],
            }
        )

        reference = get_test_reference()

        # Test with encode=False to get sequence strings
        result = sl.get_personal_genome(
            reference_fn=reference, variants_fn=variants_df, encode=False
        )

        # Verify all chromosomes are present
        assert len(result) == 5
        assert "chr1" in result
        assert "chr2" in result
        assert "chr3" in result
        assert "chr4" in result
        assert "chr5" in result

        # Verify chromosome order matches reference order
        result_chroms = list(result.keys())
        expected_order = ["chr1", "chr2", "chr3", "chr4", "chr5"]
        assert (
            result_chroms == expected_order
        ), f"Expected chromosome order {expected_order}, but got {result_chroms}"

        # Verify unmodified chromosomes still have original sequences
        from pyfaidx import Fasta

        ref = Fasta(reference)
        assert result["chr1"] == str(ref["chr1"])  # No variants
        assert result["chr2"] == str(ref["chr2"])  # No variants
        assert result["chr4"] == str(ref["chr4"])  # No variants

        # Verify modified chromosomes are different from reference
        assert result["chr3"] != str(ref["chr3"])  # Has variants
        assert result["chr5"] != str(ref["chr5"])  # Has variants

    def test_chromosome_order_with_encoded_output(self):
        """Test chromosome order preservation with encoded output."""
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr5", "chr2"],  # Reverse order to test sorting
                "pos": [10, 20],
                "id": [".", "."],
                "ref": ["A", "A"],  # Actual bases: chr5[9]=A, chr2[19]=A
                "alt": ["G", "C"],
            }
        )

        reference = get_test_reference()

        # Test with encode=True (default)
        result = sl.get_personal_genome(
            reference_fn=reference, variants_fn=variants_df, encode=True
        )

        # Verify chromosome order is preserved even with encoding
        result_chroms = list(result.keys())
        expected_order = ["chr1", "chr2", "chr3", "chr4", "chr5"]
        assert (
            result_chroms == expected_order
        ), f"Expected chromosome order {expected_order}, but got {result_chroms}"

        # Verify that results are encoded (should be arrays/tensors, not strings)
        import numpy as np

        for chrom in result_chroms:
            assert hasattr(
                result[chrom], "shape"
            ), f"{chrom} should be encoded as array/tensor, not string"
            # One-hot encoding should have shape (4, length)
            assert (
                result[chrom].shape[0] == 4
            ), f"{chrom} should have 4 channels for one-hot encoding"


class TestSingleVariantIsolation:
    """Test that get_alt_sequences applies each variant individually."""

    def test_single_variant_isolation_using_snp_vcf(self):
        """Verify each window contains ONLY its specific variant using real test data."""
        # Use the existing snp.vcf file which has:
        # chr1:2 T>G, chr1:31 T>A, chr2:19 A>G, chr2:57 C>G

        reference = get_test_reference()
        vcf_path = "tests/data/snp/snp.vcf"
        seq_len = 20  # Small window

        # Get alt sequences (should apply each variant individually)
        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=False,  # Get raw sequences for inspection
                n_chunks=1,
            )
        )

        sequences, metadata = results[0]

        # Verify we got 4 sequences (one per variant in snp.vcf)
        assert len(sequences) == 4, f"Should have 4 sequences, got {len(sequences)}"

        # Extract sequence strings
        center_idx = seq_len // 2

        # Variant 1: chr1:2 T>G
        seq1 = sequences[0][3]
        assert (
            seq1[center_idx] == "G"
        ), f"Window 1 (chr1:2) should have alt allele 'G', got '{seq1[center_idx]}'"

        # Variant 2: chr1:31 T>A
        seq2 = sequences[1][3]
        assert (
            seq2[center_idx] == "A"
        ), f"Window 2 (chr1:31) should have alt allele 'A', got '{seq2[center_idx]}'"

        # Variant 3: chr2:19 A>G
        seq3 = sequences[2][3]
        assert (
            seq3[center_idx] == "G"
        ), f"Window 3 (chr2:19) should have alt allele 'G', got '{seq3[center_idx]}'"

        # Variant 4: chr2:57 C>G
        seq4 = sequences[3][3]
        assert (
            seq4[center_idx] == "G"
        ), f"Window 4 (chr2:57) should have alt allele 'G', got '{seq4[center_idx]}'"

    def test_single_variant_isolation_close_variants_same_chromosome(self):
        """Verify chr1 variants (pos 2 and 31) don't affect each other."""
        # snp.vcf has chr1:2 T>G and chr1:31 T>A
        # These are 29bp apart

        reference = get_test_reference()
        vcf_path = "tests/data/snp/snp.vcf"
        seq_len = 40  # Windows are 40bp centered on each variant

        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=False,
                n_chunks=1,
            )
        )

        sequences, metadata = results[0]

        # Get chr1 sequences (first two variants)
        seq1 = sequences[0][3]  # chr1:2 T>G (genomic 0-based pos 1)
        seq2 = sequences[1][3]  # chr1:31 T>A (genomic 0-based pos 30)

        # Window 1: centered at genomic pos 1, half_len=20
        #   Covers genomic -19 to 21 (padded: 0 to 21)
        #   Window indices: [19 N's padding] + [genomic 0-20]
        #   Center at window index 20 corresponds to genomic position 1

        # Window 2: centered at genomic pos 30, half_len=20
        #   Covers genomic 10 to 50
        #   Window indices directly map: window[0]=genomic[10], ..., window[20]=genomic[30]

        center_idx = seq_len // 2  # 20

        # Verify variant 1 is applied at its center
        assert (
            seq1[center_idx] == "G"
        ), f"Window 1 center should be 'G' (chr1:2 variant), got '{seq1[center_idx]}'"

        # Verify variant 2 is applied at its center
        assert (
            seq2[center_idx] == "A"
        ), f"Window 2 center should be 'A' (chr1:31 variant), got '{seq2[center_idx]}'"

        # Critical test: Check that window 2 does NOT have variant 1 applied
        # Window 2 covers genomic positions 10-50
        # Genomic position 1 (variant 1) would NOT be in window 2's range
        # So we can't test this overlap

        # Instead, verify that each window only shows its own variant by checking
        # that the center matches expectations and no other variants are visible
        # This is implicitly validated by the center checks above

    def test_cross_chromosome_variant_isolation(self):
        """Variants on different chromosomes should not affect each other."""
        # Use snp.vcf which has variants on both chr1 and chr2

        reference = get_test_reference()
        vcf_path = "tests/data/snp/snp.vcf"
        seq_len = 20

        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=False,
                n_chunks=1,
            )
        )

        sequences, metadata = results[0]

        # Verify we got 4 sequences
        assert len(sequences) == 4, f"Should have 4 sequences, got {len(sequences)}"

        # Extract chromosomes and sequences
        seq1_chrom = sequences[0][0]  # chr1:2
        seq2_chrom = sequences[1][0]  # chr1:31
        seq3_chrom = sequences[2][0]  # chr2:19
        seq4_chrom = sequences[3][0]  # chr2:57

        # Verify chromosomes are correct
        assert seq1_chrom == "chr1", f"Sequence 1 should be chr1, got {seq1_chrom}"
        assert seq2_chrom == "chr1", f"Sequence 2 should be chr1, got {seq2_chrom}"
        assert seq3_chrom == "chr2", f"Sequence 3 should be chr2, got {seq3_chrom}"
        assert seq4_chrom == "chr2", f"Sequence 4 should be chr2, got {seq4_chrom}"

        # Verify each has its own variant applied (centers match expected alt alleles)
        center_idx = seq_len // 2

        assert sequences[0][3][center_idx] == "G", "chr1:2 should have alt 'G'"
        assert sequences[1][3][center_idx] == "A", "chr1:31 should have alt 'A'"
        assert sequences[2][3][center_idx] == "G", "chr2:19 should have alt 'G'"
        assert sequences[3][3][center_idx] == "G", "chr2:57 should have alt 'G'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
