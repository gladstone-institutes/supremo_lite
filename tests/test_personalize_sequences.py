"""
Tests for the personalized sequence functions in supremo_lite.

This file tests the functions for creating personalized genomes and sequence windows.
"""

import unittest
import pandas as pd
import supremo_lite as sl


class MockReferenceGenome:
    """Mock reference genome for testing."""
    
    def __init__(self):
        self.sequences = {
            'chr1': 'A' * 100 + 'C' * 100 + 'G' * 100 + 'T' * 100,
            'chr2': 'G' * 100 + 'C' * 100 + 'A' * 100 + 'T' * 100
        }
    
    def __getitem__(self, chrom):
        return self.sequences[chrom]


class TestPersonalizedSequences(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create a mock reference genome
        self.reference = MockReferenceGenome()
        
        # Create sample variants
        self.variants = pd.DataFrame({
            'chrom': ['chr1', 'chr1', 'chr2'],
            'pos': [50, 150, 250],
            'id': ['.', 'rs123', '.'],
            'ref': ['A', 'C', 'A'],
            'alt': ['G', 'T', 'AGG']
        })
    
    def test_get_personal_genome(self):
        """Test creating a personalized genome."""
        personalized = sl.get_personal_genome(self.reference, self.variants)
        
        # Check that we have entries for both chromosomes
        self.assertEqual(len(personalized), 2)
        self.assertIn('chr1', personalized)
        self.assertIn('chr2', personalized)
        
        # Check chromosome 1 variants
        chr1_seq = personalized['chr1']
        
        # First variant: A->G at position 50
        self.assertEqual(chr1_seq[49], 'G')  # 0-based index
        
        # Second variant: C->T at position 150
        self.assertEqual(chr1_seq[149], 'T')
        
        # Check chromosome 2 variant
        chr2_seq = personalized['chr2']
        
        # Third variant: A->AGG at position 250
        self.assertEqual(chr2_seq[249:252], 'AGG')
        
        # Check that lengths are as expected
        # chr1 should be unchanged in length (substitutions only)
        self.assertEqual(len(chr1_seq), len(self.reference['chr1']))
        
        # chr2 should be 2 bases longer due to insertion
        self.assertEqual(len(chr2_seq), len(self.reference['chr2']) + 2)
    
    def test_get_personal_sequences(self):
        """Test creating sequence windows centered on variants."""
        seq_len = 20
        sequences = sl.get_personal_sequences(self.reference, self.variants, seq_len=seq_len)
        
        # Check that we have the expected number of sequences
        self.assertEqual(len(sequences), 3)
        
        # Check the structure of the first sequence
        chrom, start, end, seq = sequences[0]
        self.assertEqual(chrom, 'chr1')
        self.assertEqual(start, 50 - seq_len // 2)
        self.assertEqual(end, start + seq_len)
        self.assertEqual(len(seq), seq_len)
        
        # Check a case where the window would extend beyond the start of the chromosome
        # Create a variant near the start
        start_variant = pd.DataFrame({
            'chrom': ['chr1'],
            'pos': [5],  # Very close to start
            'id': ['.'],
            'ref': ['A'],
            'alt': ['G']
        })
        
        start_sequences = sl.get_personal_sequences(self.reference, start_variant, seq_len=20)
        chrom, start, end, seq = start_sequences[0]
        
        # Start should be clamped to 0
        self.assertEqual(start, 0)
        self.assertEqual(len(seq), seq_len)
    
    def test_get_pam_disrupting_personal_sequences(self):
        """Test finding and modifying PAM sites."""
        # Create a reference with NGG PAM sites
        class PamReference:
            def __getitem__(self, chrom):
                if chrom == 'chr1':
                    # Create a sequence with NGG sites at specific positions
                    return 'A' * 10 + 'AGG' + 'A' * 10 + 'CGG' + 'A' * 10
                return ''
        
        # Create variants that could disrupt PAM sites
        pam_variants = pd.DataFrame({
            'chrom': ['chr1', 'chr1'],
            'pos': [12, 25],  # Second G of the first PAM and first G of the second PAM
            'id': ['.', '.'],
            'ref': ['G', 'G'],
            'alt': ['T', 'C']
        })
        
        result = sl.get_pam_disrupting_personal_sequences(
            PamReference(), 
            pam_variants, 
            seq_len=20, 
            max_pam_distance=5,
            pam_sequence='NGG'
        )
        
        # Check that we identified the PAM disrupting variants
        self.assertGreater(len(result['variants']), 0)
        
        # Check that we have intact and disrupted sequences
        self.assertGreater(len(result['pam_intact']), 0)
        self.assertGreater(len(result['pam_disrupted']), 0)


if __name__ == '__main__':
    unittest.main()