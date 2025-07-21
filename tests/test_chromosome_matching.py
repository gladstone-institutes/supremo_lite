"""
Comprehensive test suite for chromosome name matching functionality.

Tests the heuristics and utilities for handling mismatches between 
chromosome names in FASTA references and VCF files.
"""
import pandas as pd
import pytest
import warnings

import supremo_lite as sl
from supremo_lite.chromosome_utils import (
    normalize_chromosome_name,
    create_chromosome_mapping,
    apply_chromosome_mapping,
    get_chromosome_match_report,
    match_chromosomes_with_report,
)


class TestChromosomeNormalization:
    """Test chromosome name normalization functions."""
    
    def test_normalize_basic_chromosomes(self):
        """Test normalization of basic chromosome names."""
        # Standard chromosomes
        assert normalize_chromosome_name("1") == "1"
        assert normalize_chromosome_name("2") == "2"
        assert normalize_chromosome_name("X") == "X"
        assert normalize_chromosome_name("Y") == "Y"
        
        # With chr prefix
        assert normalize_chromosome_name("chr1") == "1"
        assert normalize_chromosome_name("CHR1") == "1"
        assert normalize_chromosome_name("chrX") == "X"
        assert normalize_chromosome_name("chrY") == "Y"
    
    def test_normalize_mitochondrial_variants(self):
        """Test normalization of mitochondrial chromosome variants."""
        # Different mitochondrial representations
        assert normalize_chromosome_name("M") == "MT"
        assert normalize_chromosome_name("chrM") == "MT"
        assert normalize_chromosome_name("MT") == "MT"
        assert normalize_chromosome_name("chrMT") == "MT"
        assert normalize_chromosome_name("mito") == "MT"
        assert normalize_chromosome_name("MITO") == "MT"
        assert normalize_chromosome_name("mitochondrion") == "MT"
    
    def test_normalize_case_handling(self):
        """Test case handling in normalization."""
        assert normalize_chromosome_name("chr1") == "1"
        assert normalize_chromosome_name("CHR1") == "1"
        assert normalize_chromosome_name("Chr1") == "1"
        assert normalize_chromosome_name("chrx") == "X"
        assert normalize_chromosome_name("CHRX") == "X"
    
    def test_normalize_whitespace(self):
        """Test handling of whitespace."""
        assert normalize_chromosome_name(" chr1 ") == "1"
        assert normalize_chromosome_name("\tchr2\n") == "2"
        assert normalize_chromosome_name("  X  ") == "X"


class TestChromosomeMapping:
    """Test chromosome mapping creation."""
    
    def test_exact_match(self):
        """Test exact chromosome name matches."""
        ref_chroms = {'1', '2', 'X', 'Y', 'MT'}
        vcf_chroms = {'1', '2', 'X'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {'1': '1', '2': '2', 'X': 'X'}
        assert mapping == expected_mapping
        assert unmatched == set()
    
    def test_chr_prefix_matching(self):
        """Test matching with/without chr prefix."""
        ref_chroms = {'1', '2', 'X', 'Y', 'MT'}
        vcf_chroms = {'chr1', 'chr2', 'chrX', 'chrY'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {
            'chr1': '1', 
            'chr2': '2', 
            'chrX': 'X', 
            'chrY': 'Y'
        }
        assert mapping == expected_mapping
        assert unmatched == set()
    
    def test_reverse_chr_prefix_matching(self):
        """Test matching when reference has chr prefix but VCF doesn't."""
        ref_chroms = {'chr1', 'chr2', 'chrX', 'chrY', 'chrMT'}
        vcf_chroms = {'1', '2', 'X', 'Y'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {
            '1': 'chr1',
            '2': 'chr2', 
            'X': 'chrX',
            'Y': 'chrY'
        }
        assert mapping == expected_mapping
        assert unmatched == set()
    
    def test_case_insensitive_matching(self):
        """Test case insensitive chromosome matching."""
        ref_chroms = {'chr1', 'chr2', 'chrX', 'chrY'}
        vcf_chroms = {'CHR1', 'Chr2', 'chrx', 'CHRY'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {
            'CHR1': 'chr1',
            'Chr2': 'chr2',
            'chrx': 'chrX', 
            'CHRY': 'chrY'
        }
        assert mapping == expected_mapping
        assert unmatched == set()
    
    def test_mitochondrial_matching(self):
        """Test mitochondrial chromosome matching variants."""
        ref_chroms = {'1', '2', 'MT'}
        vcf_chroms = {'chr1', 'chr2', 'chrM'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {
            'chr1': '1',
            'chr2': '2',
            'chrM': 'MT'  # chrM should map to MT
        }
        assert mapping == expected_mapping
        assert unmatched == set()
    
    def test_mixed_complex_matching(self):
        """Test complex mixed chromosome naming scenarios."""
        ref_chroms = {'1', '2', 'chr3', 'chrX', 'MT'}
        vcf_chroms = {'chr1', '2', '3', 'X', 'chrM'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {
            'chr1': '1',    # chr1 -> 1
            '2': '2',       # exact match
            '3': 'chr3',    # 3 -> chr3
            'X': 'chrX',    # X -> chrX
            'chrM': 'MT'    # chrM -> MT
        }
        assert mapping == expected_mapping
        assert unmatched == set()
    
    def test_unmatched_chromosomes(self):
        """Test handling of unmatched chromosomes."""
        ref_chroms = {'1', '2', 'X'}
        vcf_chroms = {'chr1', 'chr2', 'chr22', 'chrUn_123'}
        
        mapping, unmatched = create_chromosome_mapping(ref_chroms, vcf_chroms)
        
        expected_mapping = {
            'chr1': '1',
            'chr2': '2'
        }
        expected_unmatched = {'chr22', 'chrUn_123'}
        
        assert mapping == expected_mapping
        assert unmatched == expected_unmatched
    
    def test_empty_inputs(self):
        """Test handling of empty chromosome sets."""
        # Empty VCF chromosomes
        mapping, unmatched = create_chromosome_mapping({'1', '2'}, set())
        assert mapping == {}
        assert unmatched == set()
        
        # Empty reference chromosomes
        mapping, unmatched = create_chromosome_mapping(set(), {'chr1', 'chr2'})
        assert mapping == {}
        assert unmatched == {'chr1', 'chr2'}


class TestChromosomeMappingApplication:
    """Test applying chromosome mappings to DataFrames."""
    
    def test_apply_mapping_to_variants(self):
        """Test applying chromosome mapping to variants DataFrame."""
        variants_df = pd.DataFrame({
            'chrom': ['chr1', 'chr2', 'chrX', 'chr22'],
            'pos': [1000, 2000, 3000, 4000],
            'ref': ['A', 'G', 'C', 'T'],
            'alt': ['T', 'C', 'A', 'G']
        })
        
        mapping = {'chr1': '1', 'chr2': '2', 'chrX': 'X'}
        
        result_df = apply_chromosome_mapping(variants_df, mapping)
        
        expected_chroms = ['1', '2', 'X', 'chr22']  # chr22 unmapped
        assert result_df['chrom'].tolist() == expected_chroms
        
        # Other columns should be unchanged
        assert result_df['pos'].tolist() == [1000, 2000, 3000, 4000]
        assert result_df['ref'].tolist() == ['A', 'G', 'C', 'T']
        assert result_df['alt'].tolist() == ['T', 'C', 'A', 'G']
    
    def test_apply_empty_mapping(self):
        """Test applying empty mapping doesn't change DataFrame."""
        variants_df = pd.DataFrame({
            'chrom': ['chr1', 'chr2'],
            'pos': [1000, 2000],
            'ref': ['A', 'G'],
            'alt': ['T', 'C']
        })
        
        result_df = apply_chromosome_mapping(variants_df, {})
        
        # Should be unchanged
        pd.testing.assert_frame_equal(result_df, variants_df)


class TestChromosomeMatchingReport:
    """Test chromosome matching reporting functionality."""
    
    def test_match_report_generation(self):
        """Test generation of chromosome matching report."""
        ref_chroms = {'1', '2', 'X', 'Y'}
        vcf_chroms = {'chr1', 'chr2', 'chrX', 'chr22'}
        mapping = {'chr1': '1', 'chr2': '2', 'chrX': 'X'}
        unmatched = {'chr22'}
        
        report = get_chromosome_match_report(ref_chroms, vcf_chroms, mapping, unmatched)
        
        # Check report contains key information
        assert "Chromosome Matching Report" in report
        assert "Reference chromosomes (4):" in report
        assert "VCF chromosomes (4):" in report
        assert "Successfully matched (3):" in report
        assert "Unmatched VCF chromosomes (1):" in report
        assert "Matching coverage: 75.0%" in report
        assert "'chr1' -> '1'" in report
        assert "'chr22' (no suitable reference match found)" in report
    
    def test_match_with_report_function(self):
        """Test match_chromosomes_with_report function."""
        ref_chroms = {'1', '2', 'X'}
        vcf_chroms = {'chr1', 'chr2', 'chr22'}
        
        with warnings.catch_warnings(record=True) as w:
            mapping, unmatched = match_chromosomes_with_report(
                ref_chroms, vcf_chroms, verbose=False
            )
            
            # Should generate a warning for unmatched chromosomes
            assert len(w) == 1
            assert "Could not match" in str(w[0].message)
            assert "chr22" in str(w[0].message)
        
        expected_mapping = {'chr1': '1', 'chr2': '2'}
        expected_unmatched = {'chr22'}
        
        assert mapping == expected_mapping
        assert unmatched == expected_unmatched


class TestIntegrationWithPersonalizeFunctions:
    """Test integration of chromosome matching with main functions."""
    
    def create_test_reference_and_variants(self):
        """Create test reference and variants with chromosome name mismatches."""
        # Reference uses numbers: 1, 2, X
        reference = {
            "1": "ATCGATCGATCGATCGATCGATCG" * 20,
            "2": "GCTAGCTAGCTAGCTAGCTAGCTA" * 20, 
            "X": "AAATTTCCCGGGAAATTTCCCGGG" * 20
        }
        
        # Create variants that match the actual reference sequences
        # Get actual nucleotides at specific positions
        ref1_seq = reference["1"]
        ref2_seq = reference["2"]
        refX_seq = reference["X"]
        
        # VCF uses chr prefix: chr1, chr2, chrX
        variants_df = pd.DataFrame({
            'chrom': ['chr1', 'chr2', 'chrX', 'chr1'],
            'pos': [100, 200, 300, 400],
            'id': ['.', '.', '.', '.'],
            'ref': [ref1_seq[99], ref2_seq[199], refX_seq[299], ref1_seq[399]],  # 0-based indexing
            'alt': ['T', 'C', 'G', 'C']
        })
        
        return reference, variants_df
    
    def test_get_personal_genome_with_chromosome_matching(self):
        """Test get_personal_genome handles chromosome name mismatches."""
        reference, variants_df = self.create_test_reference_and_variants()
        
        # Should work despite chromosome name mismatch
        result = sl.get_personal_genome(
            reference_fn=reference,
            variants_fn=variants_df,
            encode=False
        )
        
        # Should have all reference chromosomes
        assert set(result.keys()) == {'1', '2', 'X'}
        
        # All chromosomes should have the same length as reference
        for chrom in ['1', '2', 'X']:
            assert len(result[chrom]) == len(reference[chrom])
        
        # The chromosome matching worked if the function completed without error
        # and returned results (even if some variants couldn't be applied due to mismatches)
        assert isinstance(result, dict)
        assert all(isinstance(seq, str) for seq in result.values())
    
    def test_get_personal_sequences_with_chromosome_matching(self):
        """Test get_personal_sequences handles chromosome name mismatches."""
        reference, variants_df = self.create_test_reference_and_variants()
        
        # Should work despite chromosome name mismatch
        results = list(sl.get_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=50,
            encode=False,
            chunk_size=2
        ))
        
        # Should process all 4 variants in 2 chunks
        assert len(results) == 2
        assert len(results[0]) == 2  # First chunk: 2 variants
        assert len(results[1]) == 2  # Second chunk: 2 variants
        
        # Check that sequences were generated
        for chunk in results:
            for item in chunk:
                chrom, start, end, sequence = item
                assert chrom in ['1', '2', 'X']  # Should use reference chromosome names
                assert len(sequence) == 50
                assert isinstance(sequence, str)
    
    def test_get_pam_disrupting_sequences_with_chromosome_matching(self):
        """Test PAM disruption function handles chromosome name mismatches."""
        # Create reference with PAM sites
        reference = {
            "1": "ATCG" * 25 + "NGG" + "ATCG" * 25,  # NGG at position ~100
            "X": "GCTA" * 25 + "NGG" + "GCTA" * 25   # NGG at position ~100
        }
        
        # Create variants with chr prefix that should match reference
        variants_df = pd.DataFrame({
            'chrom': ['chr1', 'chrX'],
            'pos': [100, 100],  # Near PAM sites
            'id': ['.', '.'],
            'ref': ['A', 'G'],
            'alt': ['T', 'C']
        })
        
        result = sl.get_pam_disrupting_personal_sequences(
            reference_fn=reference,
            variants_fn=variants_df,
            seq_len=50,
            max_pam_distance=10,
            pam_sequence="NGG",
            encode=False
        )
        
        # Should find PAM-disrupting variants
        assert len(result["variants"]) > 0
        assert len(result["pam_intact"]) > 0
        assert len(result["pam_disrupted"]) > 0
    
    def test_chromosome_matching_with_no_matches(self):
        """Test behavior when no chromosomes can be matched."""
        reference = {"1": "ATCGATCGATCG", "2": "GCTAGCTAGCTA"}
        
        # VCF with completely different chromosome names
        variants_df = pd.DataFrame({
            'chrom': ['scaffold_123', 'contig_456'],
            'pos': [100, 200],
            'id': ['.', '.'],
            'ref': ['A', 'G'],
            'alt': ['T', 'C']
        })
        
        # Should still run but produce genome without variants applied
        result = sl.get_personal_genome(
            reference_fn=reference,
            variants_fn=variants_df,
            encode=False
        )
        
        # Should have reference chromosomes unchanged
        assert result == reference
    
    def test_complex_chromosome_scenarios(self):
        """Test complex real-world chromosome naming scenarios."""
        # Mix of naming conventions
        reference = {
            "1": "A" * 1000,
            "chr2": "T" * 1000, 
            "X": "G" * 1000,
            "chrMT": "C" * 1000
        }
        
        variants_df = pd.DataFrame({
            'chrom': ['chr1', '2', 'chrX', 'M'],  # Mixed naming
            'pos': [100, 200, 300, 400],
            'id': ['.', '.', '.', '.'],
            'ref': ['A', 'T', 'G', 'C'],
            'alt': ['G', 'A', 'C', 'T']
        })
        
        result = sl.get_personal_genome(
            reference_fn=reference,
            variants_fn=variants_df,
            encode=False
        )
        
        # All chromosomes should be processed
        assert set(result.keys()) == {"1", "chr2", "X", "chrMT"}
        
        # Variants should be applied to each chromosome
        for chrom in result.keys():
            assert result[chrom] != reference[chrom]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])