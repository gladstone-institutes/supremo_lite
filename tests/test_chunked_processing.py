"""
Test chunked processing functionality for large VCF files.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
import warnings

import supremo_lite as sl
from supremo_lite.variant_utils import read_vcf_chunked


def create_test_vcf(path, n_variants=10):
    """Create a test VCF file with specified number of variants."""
    with open(path, "w") as f:
        # Write VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##contig=<ID=chr1,length=248956422>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Write variants
        for i in range(n_variants):
            pos = 1000 + i * 100  # Space variants 100bp apart
            f.write(f"chr1\t{pos}\t.\tA\tG\t.\t.\t.\n")


def create_test_reference():
    """Create a simple test reference genome."""
    # Create 10kb of sequence for testing
    seq = "ATCG" * 2500  # 10,000 bp
    return {"chr1": seq}


class TestChunkedVCFReading:
    """Test chunked VCF reading functionality."""

    def test_read_vcf_chunked_single_chunk(self):
        """Test reading small VCF in single chunk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            create_test_vcf(vcf_path, n_variants=5)

        try:
            chunks = list(read_vcf_chunked(vcf_path, chunk_size=10))

            # Should have only one chunk
            assert len(chunks) == 1
            assert len(chunks[0]) == 5
            assert list(chunks[0].columns) == ["chrom", "pos1", "id", "ref", "alt"]
            assert chunks[0]["chrom"].iloc[0] == "chr1"
            assert chunks[0]["pos1"].iloc[0] == 1000
        finally:
            os.unlink(vcf_path)

    def test_read_vcf_chunked_multiple_chunks(self):
        """Test reading VCF split into multiple chunks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            create_test_vcf(vcf_path, n_variants=7)

        try:
            chunks = list(read_vcf_chunked(vcf_path, chunk_size=3))

            # Should have 3 chunks: 3, 3, 1 variants
            assert len(chunks) == 3
            assert len(chunks[0]) == 3
            assert len(chunks[1]) == 2  # array_split creates approximately equal chunks
            assert len(chunks[2]) == 2

            # Check continuity
            all_variants = pd.concat(chunks, ignore_index=True)
            assert len(all_variants) == 7
            assert all_variants["pos1"].iloc[0] == 1000
            assert all_variants["pos1"].iloc[6] == 1600
        finally:
            os.unlink(vcf_path)

    def test_read_vcf_chunked_empty_file(self):
        """Test reading empty VCF file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        try:
            chunks = list(read_vcf_chunked(vcf_path, chunk_size=3))
            assert len(chunks) == 0
        finally:
            os.unlink(vcf_path)


class TestChunkedPersonalSequences:
    """Test chunked processing in get_alt_sequences."""

    def test_chunked_processing_default_chunk_size(self):
        """Test that default chunk_size=1 yields individual variants."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            create_test_vcf(vcf_path, n_variants=3)

        reference = create_test_reference()
        seq_len = 100

        try:
            # Test default chunking (chunk_size=1)
            results = list(
                sl.get_alt_sequences(
                    reference_fn=reference,
                    variants_fn=vcf_path,
                    seq_len=seq_len,
                    encode=True,
                )
            )

            # Should yield 3 individual results
            assert len(results) == 3

            # Each result should be a single sequence
            for result in results:
                assert result.shape == (
                    1,
                    seq_len,
                    4,
                )  # (1 variant, seq_len, 4 nucleotides)
        finally:
            os.unlink(vcf_path)

    def test_chunked_processing_multiple_variants(self):
        """Test chunking with multiple variants per chunk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            create_test_vcf(vcf_path, n_variants=5)

        reference = create_test_reference()
        seq_len = 100

        try:
            # Test chunking with chunk_size=2
            results = list(
                sl.get_alt_sequences(
                    reference_fn=reference,
                    variants_fn=vcf_path,
                    seq_len=seq_len,
                    encode=True,
                    chunk_size=2,
                )
            )

            # Should yield 3 chunks: 2, 2, 1 variants
            assert len(results) == 3
            assert results[0].shape == (2, seq_len, 4)
            assert results[1].shape == (
                2,
                seq_len,
                4,
            )  # array_split creates approximately equal chunks
            assert results[2].shape == (1, seq_len, 4)
        finally:
            os.unlink(vcf_path)

    def test_chunked_processing_with_dataframe(self):
        """Test chunking when variants_fn is a DataFrame."""
        # Create test DataFrame
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "pos": [1000, 1100, 1200, 1300],
                "id": ["."] * 4,
                "ref": ["A"] * 4,
                "alt": ["G"] * 4,
            }
        )

        reference = create_test_reference()
        seq_len = 100

        # Test chunking with chunk_size=3
        results = list(
            sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=variants_df,
                seq_len=seq_len,
                encode=True,
                chunk_size=3,
            )
        )

        # Should yield 2 chunks: 2, 2 variants (array_split creates approximately equal chunks)
        assert len(results) == 2
        assert results[0].shape == (2, seq_len, 4)
        assert results[1].shape == (2, seq_len, 4)

    def test_chunked_processing_no_encoding(self):
        """Test chunked processing without encoding."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            create_test_vcf(vcf_path, n_variants=3)

        reference = create_test_reference()
        seq_len = 100

        try:
            # Test without encoding
            results = list(
                sl.get_alt_sequences(
                    reference_fn=reference,
                    variants_fn=vcf_path,
                    seq_len=seq_len,
                    encode=False,
                    chunk_size=2,
                )
            )

            # Should yield 2 chunks
            assert len(results) == 2
            assert len(results[0]) == 2  # 2 variants in first chunk
            assert len(results[1]) == 1  # 1 variant in second chunk

            # Check format of results
            for chunk in results:
                for item in chunk:
                    assert len(item) == 4  # (chrom, start, end, sequence_string)
                    assert isinstance(item[3], str)  # sequence is string
                    assert len(item[3]) == seq_len
        finally:
            os.unlink(vcf_path)

    def test_memory_efficiency_comparison(self):
        """Test that chunking uses less memory than loading all at once."""
        # This is a conceptual test - in practice we'd need much larger files
        # to see real memory differences, but we can test the pattern

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            vcf_path = f.name
            create_test_vcf(vcf_path, n_variants=10)

        reference = create_test_reference()
        seq_len = 100

        try:
            # Process with chunking - should be generator
            chunked_results = sl.get_alt_sequences(
                reference_fn=reference,
                variants_fn=vcf_path,
                seq_len=seq_len,
                encode=True,
                chunk_size=3,
            )

            # Verify it's a generator
            import types

            assert isinstance(chunked_results, types.GeneratorType)

            # Consume generator and verify total results
            all_chunks = list(chunked_results)
            total_variants = sum(chunk.shape[0] for chunk in all_chunks)
            assert total_variants == 10
        finally:
            os.unlink(vcf_path)


class TestChunkedPersonalizeFunctions:
    """Test chunked processing in main personalize functions."""

    def test_get_personal_genome_with_chunk_size(self):
        """Test get_personal_genome with chunk_size parameter."""
        # Create test data
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 3,
                "pos": [1000, 1100, 1200],
                "id": ["."] * 3,
                "ref": ["A", "A", "A"],
                "alt": ["G", "T", "C"],
            }
        )

        reference = create_test_reference()

        # Test with default chunk_size=1
        result1 = sl.get_personal_genome(
            reference_fn=reference, variants_fn=variants_df, encode=False, chunk_size=1
        )

        # Test with chunk_size=2 (should work the same for small data)
        result2 = sl.get_personal_genome(
            reference_fn=reference, variants_fn=variants_df, encode=False, chunk_size=2
        )

        # Results should be identical regardless of chunk_size for get_personal_genome
        # since it processes all variants for each chromosome anyway
        assert "chr1" in result1
        assert "chr1" in result2
        assert result1["chr1"] == result2["chr1"]

    def test_get_pam_disrupting_sequences_with_chunk_size(self):
        """Test get_pam_disrupting_personal_sequences with chunk_size parameter."""
        # Create variants near PAM sites
        variants_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 2,
                "pos": [1000, 1100],
                "id": ["."] * 2,
                "ref": ["A", "A"],
                "alt": ["G", "T"],
            }
        )

        # Create reference with PAM sites near variant positions
        reference = {
            "chr1": "A" * 950 + "AAAGGAAA" + "A" * 50 + "AAAGGAAA" + "A" * 8942
        }

        # Test with default chunk_size=1
        result1 = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=100,
            max_pam_distance=10,
            pam_sequence="AGG",
            encode=False,
            chunk_size=1,
        )

        # Test with chunk_size=2
        result2 = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=100,
            max_pam_distance=10,
            pam_sequence="AGG",
            encode=False,
            chunk_size=2,
        )

        # Results should be identical
        assert len(result1["variants"]) == len(result2["variants"])
        assert len(result1["pam_intact"]) == len(result2["pam_intact"])
        assert len(result1["pam_disrupted"]) == len(result2["pam_disrupted"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
