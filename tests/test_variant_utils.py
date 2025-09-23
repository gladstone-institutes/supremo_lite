"""
Comprehensive tests for variant_utils module.

Tests VCF 4.2 specification-compliant variant classification,
INFO field parsing, and VCF reading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

from supremo_lite.variant_utils import (
    classify_variant_type,
    parse_vcf_info,
    read_vcf,
    read_vcf_chunked,
    read_vcf_chromosome,
    read_vcf_chromosomes_chunked
)


class TestVariantClassification:
    """Test VCF 4.2 specification-compliant variant classification."""
    
    def test_snv_classification(self):
        """Test SNV (Single Nucleotide Variant) classification."""
        # Standard SNVs
        assert classify_variant_type('A', 'G') == 'SNV'
        assert classify_variant_type('T', 'C') == 'SNV'
        assert classify_variant_type('G', 'A') == 'SNV'
        assert classify_variant_type('C', 'T') == 'SNV'
        
        # Mixed case
        assert classify_variant_type('a', 'g') == 'SNV'
        assert classify_variant_type('T', 'c') == 'SNV'
        
    def test_ins_classification(self):
        """Test INS variant classification."""
        # Simple INSs
        assert classify_variant_type('T', 'TGGG') == 'INS'
        assert classify_variant_type('A', 'ATCG') == 'INS'
        assert classify_variant_type('C', 'CAAAA') == 'INS'
        
        # Single base INS
        assert classify_variant_type('G', 'GT') == 'INS'
        
        # Mixed case
        assert classify_variant_type('a', 'atcg') == 'INS'
        
    def test_del_classification(self):
        """Test DEL variant classification."""
        # Simple DELs  
        assert classify_variant_type('CGAGAA', 'C') == 'DEL'
        assert classify_variant_type('ATCG', 'A') == 'DEL'
        assert classify_variant_type('TTTT', 'T') == 'DEL'
        
        # Single base DEL
        assert classify_variant_type('AT', 'A') == 'DEL'
        
        # Mixed case
        assert classify_variant_type('CGAGAA', 'c') == 'DEL'
        
    def test_complex_variant_classification(self):
        """Test complex variant classification."""
        # Multi-nucleotide variants of same length
        assert classify_variant_type('ATCG', 'GCTA') == 'MNV'
        
        # Mixed case
        assert classify_variant_type('atcg', 'GCTA') == 'MNV'
        
    def test_missing_allele_classification(self):
        """Test missing/upstream DEL allele classification."""
        assert classify_variant_type('A', '*') == 'missing'
        assert classify_variant_type('TCGA', '*') == 'missing'
        
    def test_structural_variant_classification(self):
        """Test structural variant classification based on INFO field."""
        # Inversions
        info_inv = {'SVTYPE': 'INV', 'END': 1000, 'SVLEN': 500}
        assert classify_variant_type('A', '<INV>', info_inv) == 'SV_INV'
        
        # Duplications
        info_dup = {'SVTYPE': 'DUP', 'END': 2000, 'SVLEN': 1000}
        assert classify_variant_type('G', '<DUP>', info_dup) == 'SV_DUP'
        
        # Structural DELs
        info_del = {'SVTYPE': 'DEL', 'END': 1500, 'SVLEN': -1000}
        assert classify_variant_type('T', '<DEL>', info_del) == 'SV_DEL'
        
        # Structural INSs
        info_ins = {'SVTYPE': 'INS', 'SVLEN': 500}
        assert classify_variant_type('C', '<INS>', info_ins) == 'SV_INS'
        
        # Copy number variants
        info_cnv = {'SVTYPE': 'CNV', 'END': 3000, 'SVLEN': 2000}
        assert classify_variant_type('A', '<CNV>', info_cnv) == 'SV_CNV'
        
    def test_breakend_classification(self):
        """Test breakend (BND) variant classification."""
        # Standard breakend notation (no insertions)
        assert classify_variant_type('A', 'A[chr2:1000[') == 'SV_BND'
        assert classify_variant_type('T', ']chr1:100]T') == 'SV_BND'
        assert classify_variant_type('G', 'G]chr3:500]') == 'SV_BND'
        assert classify_variant_type('C', '[chrX:200[C') == 'SV_BND'

        # Breakend with insertions should be classified as SV_BND_INS
        assert classify_variant_type('T', ']chr2:20]ATCGT') == 'SV_BND_INS'
        assert classify_variant_type('GCAT', 'GCAT[chr5:100[') == 'SV_BND_INS'
        assert classify_variant_type('A', '[chr1:50[ANNNCAT') == 'SV_BND_INS'

        # With INFO field
        info_bnd = {'SVTYPE': 'BND', 'CHR2': 'chr2', 'POS2': 1000}
        assert classify_variant_type('A', 'A[chr2:1000[', info_bnd) == 'SV_BND'
        
    def test_symbolic_allele_fallback(self):
        """Test symbolic alleles without proper INFO field classification."""
        # Symbolic alleles without INFO should be classified based on symbol
        assert classify_variant_type('A', '<INV>') == 'SV_INV'
        assert classify_variant_type('T', '<DUP>') == 'SV_DUP'
        assert classify_variant_type('G', '<DEL>') == 'SV_DEL'
        assert classify_variant_type('C', '<INS>') == 'SV_INS'
        assert classify_variant_type('A', '<CNV>') == 'SV_CNV'
        
        # Unknown symbolic alleles
        assert classify_variant_type('A', '<UNKNOWN>') == 'unknown'
        
    def test_edge_cases(self):
        """Test edge cases and unusual inputs."""
        # Empty strings
        assert classify_variant_type('', 'A') == 'unknown'
        assert classify_variant_type('A', '') == 'unknown'
        
        # None values
        assert classify_variant_type(None, 'A') == 'unknown'
        assert classify_variant_type('A', None) == 'unknown'
        
        # Ambiguous nucleotides
        assert classify_variant_type('N', 'A') == 'SNV'
        assert classify_variant_type('A', 'N') == 'SNV'
        
        # Multiple alternative alleles (should be complex)
        assert classify_variant_type('A', 'G,T') == 'complex'
        assert classify_variant_type('T', 'TGGG,C') == 'complex'
        
    def test_classification_priority(self):
        """Test that structural variants take priority over sequence variants."""
        # Structural variant info should override sequence-based classification
        info_sv = {'SVTYPE': 'INV', 'END': 1000}
        assert classify_variant_type('ATCG', 'GCTA', info_sv) == 'SV_INV'
        
        # Breakend notation should override sequence-based classification
        # Standard BND without insertion
        assert classify_variant_type('G', 'G]chr2:1000]') == 'SV_BND'
        # BND with insertion should be classified as SV_BND_INS
        assert classify_variant_type('ATCG', 'ATCG[chr2:1000[') == 'SV_BND_INS'


class TestInfoFieldParsing:
    """Test VCF INFO field parsing functionality."""
    
    def test_basic_info_parsing(self):
        """Test basic INFO field parsing."""
        info_str = "DP=100;AF=0.5;AN=2"
        result = parse_vcf_info(info_str)
        
        expected = {'DP': 100, 'AF': 0.5, 'AN': 2}
        assert result == expected
        
    def test_structural_variant_info(self):
        """Test parsing structural variant INFO fields."""
        info_str = "SVTYPE=INV;END=1000;SVLEN=500;CHR2=chr2;POS2=1500"
        result = parse_vcf_info(info_str)
        
        expected = {
            'SVTYPE': 'INV',
            'END': 1000,
            'SVLEN': 500,
            'CHR2': 'chr2',
            'POS2': 1500
        }
        assert result == expected
        
    def test_boolean_flags(self):
        """Test parsing boolean flags in INFO field."""
        info_str = "DP=50;SOMATIC;VALIDATED;AF=0.3"
        result = parse_vcf_info(info_str)
        
        expected = {
            'DP': 50,
            'SOMATIC': True,
            'VALIDATED': True,
            'AF': 0.3
        }
        assert result == expected
        
    def test_string_values(self):
        """Test parsing string values in INFO field."""
        info_str = "ID=rs123456;GENE=BRCA1;CONSEQUENCE=missense_variant"
        result = parse_vcf_info(info_str)
        
        expected = {
            'ID': 'rs123456',
            'GENE': 'BRCA1',
            'CONSEQUENCE': 'missense_variant'
        }
        assert result == expected
        
    def test_mixed_types(self):
        """Test parsing mixed data types in INFO field."""
        info_str = "DP=100;AF=0.5,0.3;VALIDATED;GENE=TP53;COUNT=5"
        result = parse_vcf_info(info_str)
        
        expected = {
            'DP': 100,
            'AF': [0.5,0.3],  # Comma-separated values
            'VALIDATED': True,
            'GENE': 'TP53',
            'COUNT': 5
        }
        assert result == expected
        
    def test_empty_info_field(self):
        """Test parsing empty or missing INFO fields."""
        assert parse_vcf_info('') == {}
        assert parse_vcf_info('.') == {}
        assert parse_vcf_info(None) == {}
        
    def test_malformed_info_field(self):
        """Test parsing malformed INFO fields."""
        # Missing value
        info_str = "DP=;AF=0.5"
        result = parse_vcf_info(info_str)
        assert 'DP' in result  # Should still parse, value might be empty string
        assert result['AF'] == 0.5
        
        # Multiple equals signs
        info_str = "DESC=key=value;DP=100"
        result = parse_vcf_info(info_str)
        assert result['DESC'] == 'key=value'
        assert result['DP'] == 100


class TestVCFReading:
    """Test VCF reading functions with variant classification."""
    
    def create_test_vcf(self, variants_data):
        """Helper to create test VCF content."""
        header = "##fileformat=VCFv4.2\n"
        header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        
        lines = [header]
        for variant in variants_data:
            line = f"{variant['chrom']}\t{variant['pos']}\t{variant.get('id', '.')}\t"
            line += f"{variant['ref']}\t{variant['alt']}\t{variant.get('qual', '.')}\t"
            line += f"{variant.get('filter', 'PASS')}\t{variant.get('info', '.')}\n"
            lines.append(line)
            
        return ''.join(lines)
  
    def test_read_vcf_without_classification(self, tmp_path):
        """Test reading VCF without variant classification."""
        variants = [
            {'chrom': 'chr1', 'pos': 100, 'ref': 'A', 'alt': 'G', 'info': 'DP=50'},
        ]
        
        vcf_content = self.create_test_vcf(variants)
        vcf_file = tmp_path / "test.vcf"
        vcf_file.write_text(vcf_content)
        
        # Read without classification
        df = read_vcf(str(vcf_file), classify_variants=False)
        
        # Should not have classification or info dict columns
        assert 'variant_type' not in df.columns
        assert 'info_dict' not in df.columns
        
        # Should still have basic INFO column
        assert 'info' in df.columns
        assert df.iloc[0]['info'] == 'DP=50'

    def test_read_vcf_chunked_with_classification(self, tmp_path):
        """Test chunked VCF reading with classification."""
        # Create larger test file
        variants = []
        for i in range(50):  # 50 variants
            variants.append({
                'chrom': f'chr{(i % 3) + 1}',
                'pos': 1000 + i * 100,
                'ref': 'A',
                'alt': 'G' if i % 2 == 0 else 'AGGG',
                'info': f'DP={50 + i}'
            })
            
        vcf_content = self.create_test_vcf(variants)
        vcf_file = tmp_path / "test_chunked.vcf"
        vcf_file.write_text(vcf_content)
        
        # Read in chunks
        chunks = []
        for chunk_df in read_vcf_chunked(str(vcf_file), n_chunks=5, classify_variants=True):
            chunks.append(chunk_df)
            
        # Should have 5 chunks of ~10 variants each (50 variants split into 5 chunks)
        assert len(chunks) == 5
        
        # Combine and check
        combined_df = pd.concat(chunks, ignore_index=True)
        assert len(combined_df) == 50
        assert 'variant_type' in combined_df.columns
        
        # Check that SNVs and INSs are properly classified
        snv_count = sum(combined_df['variant_type'] == 'SNV')
        INS_count = sum(combined_df['variant_type'] == 'INS')
        assert snv_count == 25  # Even indices
        assert INS_count == 25  # Odd indices
        
    def test_read_vcf_chromosome_specific(self, tmp_path):
        """Test reading specific chromosomes with classification."""
        variants = [
            {'chrom': 'chr1', 'pos': 100, 'ref': 'A', 'alt': 'G', 'info': 'DP=50'},
            {'chrom': 'chr2', 'pos': 200, 'ref': 'T', 'alt': 'TGGG', 'info': 'DP=60'},
            {'chrom': 'chr1', 'pos': 300, 'ref': 'CGAA', 'alt': 'C', 'info': 'DP=70'},
        ]
        
        vcf_content = self.create_test_vcf(variants)
        vcf_file = tmp_path / "test_chr.vcf"
        vcf_file.write_text(vcf_content)
        
        # Read only chr1
        df = read_vcf_chromosome(str(vcf_file), 'chr1', classify_variants=True)
        
        # Should have 2 chr1 variants
        assert len(df) == 2
        assert all(df['chrom'] == 'chr1')
        assert df.iloc[0]['variant_type'] == 'SNV'
        assert df.iloc[1]['variant_type'] == 'DEL'

if __name__ == '__main__':
    # Run tests if executed directly
    pytest.main([__file__, '-v'])