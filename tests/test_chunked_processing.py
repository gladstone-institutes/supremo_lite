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
        assert list(chunks[0].columns) == ['chrom', 'pos1', 'id', 'ref', 'alt', 'info', 'variant_type']
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
            seq_len,
            4,
        )  # (4 variants, seq_len, 4 nucleotides)
        
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
        
        assert sequences_1.shape == (2, seq_len, 4)
        assert sequences_2.shape == (2, seq_len, 4)
        
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
                "ref": ["T", "A", "T", "C"],  # Actual bases at positions 2=T, 10=A, 20=T, 30=C
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
        
        assert sequences_1.shape == (2, seq_len, 4)
        assert sequences_2.shape == (1, seq_len, 4)
        assert sequences_3.shape == (1, seq_len, 4)
        
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
        """Test get_pam_disrupting_personal_sequences with n_chunks parameter."""
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
        result1 = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=50,  # Smaller seq_len to fit within the test genome
            max_pam_distance=10,
            pam_sequence="AGG",
            encode=False,
            n_chunks=1,
        )

        # Test with n_chunks=2
        result2 = sl.get_pam_disrupting_personal_sequences(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
