"""
Tests for the personalized sequence functions in supremo_lite.

This file tests the get_personal_genome function using real test data files
for SNPs, insertions, and deletions.
"""

import unittest
import os
import pandas as pd
from pyfaidx import Fasta
import supremo_lite as sl
import warnings


class TestPersonalizeGenome(unittest.TestCase):
    """Test cases for genome personalization using VCF files."""

    def setUp(self):
        """Set up test data paths."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

    def test_snp_variants(self):
        """Test personalization with SNP variants."""
        # Paths for SNP test
        snp_vcf = os.path.join(self.data_dir, "snp", "snp.vcf")
        snp_expected = os.path.join(self.data_dir, "snp", "snp_expected_output.fa")

        # Create personalized genome
        personalized = sl.get_personal_genome(self.reference_fa, snp_vcf, encode=False)

        # Load expected output
        expected = Fasta(snp_expected)

        # Compare each chromosome
        for chrom in expected.keys():
            self.assertIn(chrom, personalized)
            self.assertEqual(
                personalized[chrom],
                str(expected[chrom]),
                f"Mismatch in {chrom} for SNP variants",
            )

    def test_insertion_variants(self):
        """Test personalization with insertion variants."""
        # Paths for insertion test
        ins_vcf = os.path.join(self.data_dir, "ins", "ins.vcf")
        ins_expected = os.path.join(self.data_dir, "ins", "ins_expected_output.fa")

        # Create personalized genome
        personalized = sl.get_personal_genome(self.reference_fa, ins_vcf, encode=False)

        # Load expected output
        expected = Fasta(ins_expected)

        # Compare each chromosome
        for chrom in expected.keys():
            self.assertIn(chrom, personalized)
            self.assertEqual(
                personalized[chrom],
                str(expected[chrom]),
                f"Mismatch in {chrom} for insertion variants",
            )

    def test_deletion_variants(self):
        """Test personalization with deletion variants."""
        # Paths for deletion test
        del_vcf = os.path.join(self.data_dir, "del", "del.vcf")
        del_expected = os.path.join(self.data_dir, "del", "del_expected_output.fa")

        # Create personalized genome
        personalized = sl.get_personal_genome(self.reference_fa, del_vcf, encode=False)

        # Load expected output
        expected = Fasta(del_expected)

        # Compare each chromosome
        for chrom in expected.keys():
            self.assertIn(chrom, personalized)
            self.assertEqual(
                personalized[chrom],
                str(expected[chrom]),
                f"Mismatch in {chrom} for deletion variants",
            )

    def test_multi_overlapping_variants(self):
        """
        Test personalization with multiple overlapping variants where some should be skipped.
        The frozen region tracker should prevent overlapping variants from being applied.
        """
        # Paths for multi-variant test
        multi_vcf = os.path.join(self.data_dir, "multi", "multi.vcf")
        multi_expected = os.path.join(
            self.data_dir, "multi", "multi_expected_output.fa"
        )

        # Load the variants to understand what we're testing
        variants = sl.read_vcf(multi_vcf)
        chr1_variants = variants[variants["chrom"] == "chr1"].sort_values("pos")

        # Verify we have the expected overlapping variants
        self.assertTrue(
            len(chr1_variants) >= 6, "Should have multiple variants for testing"
        )

        # Check for overlapping variants at positions 11
        pos_11_variants = chr1_variants[chr1_variants["pos"] == 11]
        self.assertTrue(
            len(pos_11_variants) >= 2, "Should have overlapping variants at position 11"
        )

        # Create personalized genome - warnings will be visible in test output
        # but we don't need to catch them for this test to pass
        personalized = sl.get_personal_genome(
            self.reference_fa, multi_vcf, encode=False
        )

        # Load expected output and compare
        expected = Fasta(multi_expected)

        # Compare chr1 (the chromosome with overlapping variants)
        self.assertIn("chr1", personalized)
        self.assertEqual(
            personalized["chr1"],
            str(expected["chr1"]),
            f"Multi-variant chr1 sequence doesn't match expected output.\n"
            f"Expected: {str(expected['chr1'])}\n"
            f"Got:      {personalized['chr1']}",
        )

        # Verify other chromosomes remain unchanged
        ref_sequences = Fasta(self.reference_fa)
        for chrom in ["chr2", "chr3", "chr4", "chr5"]:
            self.assertEqual(
                personalized[chrom],
                str(ref_sequences[chrom]),
                f"Chromosome {chrom} should remain unchanged in multi-variant test",
            )

    def test_variant_details(self):
        """Test specific variant applications to verify correctness."""
        # Test SNP variant details
        snp_vcf = os.path.join(self.data_dir, "snp", "snp.vcf")
        personalized = sl.get_personal_genome(self.reference_fa, snp_vcf, encode=False)
        expected = Fasta(os.path.join(self.data_dir, "snp", "snp_expected_output.fa"))

        # Just verify the output matches expected
        self.assertEqual(personalized["chr1"], str(expected["chr1"]))
        self.assertEqual(personalized["chr2"], str(expected["chr2"]))

    def test_insertion_details(self):
        """Test specific insertion applications."""
        ins_vcf = os.path.join(self.data_dir, "ins", "ins.vcf")
        personalized = sl.get_personal_genome(self.reference_fa, ins_vcf, encode=False)
        expected = Fasta(os.path.join(self.data_dir, "ins", "ins_expected_output.fa"))

        # Just verify the output matches expected
        self.assertEqual(personalized["chr1"], str(expected["chr1"]))
        self.assertEqual(personalized["chr2"], str(expected["chr2"]))

    def test_deletion_details(self):
        """Test specific deletion applications."""
        del_vcf = os.path.join(self.data_dir, "del", "del.vcf")
        variants = sl.read_vcf(del_vcf)
        personalized = sl.get_personal_genome(self.reference_fa, variants, encode=False)

        # Get reference and expected for comparison
        ref_sequences = Fasta(self.reference_fa)
        expected = Fasta(os.path.join(self.data_dir, "del", "del_expected_output.fa"))

        # Just verify the final result matches expected
        self.assertEqual(
            personalized["chr1"],
            str(expected["chr1"]),
            "Deletion on chr1 did not produce expected result",
        )

    def test_variant_outside_bounds(self):
        """Test handling of variants that fall outside chromosome bounds."""
        # Create a variant past the end of chr1
        invalid_variants = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [1000],  # chr1 is only 80 bases long
                "id": ["."],
                "ref": ["A"],
                "alt": ["G"],
            }
        )

        # Should handle gracefully without error
        personalized = sl.get_personal_genome(
            self.reference_fa, invalid_variants, encode=False
        )

        # Original sequence should be unchanged
        ref_sequences = Fasta(self.reference_fa)
        self.assertEqual(personalized["chr1"], str(ref_sequences["chr1"]))

    def test_reference_mismatch(self):
        """Test handling of variants where reference allele doesn't match."""
        # Create a variant with wrong reference allele
        mismatch_variants = pd.DataFrame(
            {
                "chrom": ["chr1"],
                "pos": [5],  # Position 5 is 'A' in reference
                "id": ["."],
                "ref": ["G"],  # Wrong reference allele
                "alt": ["T"],
            }
        )

        # Should handle gracefully, skip the variant
        with self.assertWarns(Warning):
            personalized = sl.get_personal_genome(
                self.reference_fa, mismatch_variants, encode=False
            )

        # Original sequence should be unchanged
        ref_sequences = Fasta(self.reference_fa)
        self.assertEqual(personalized["chr1"], str(ref_sequences["chr1"]))

    def test_chromosome_not_in_vcf(self):
        """Test that chromosomes not in VCF remain unchanged."""
        # Use SNP VCF which only has variants for chr1 and chr2
        snp_vcf = os.path.join(self.data_dir, "snp", "snp.vcf")
        personalized = sl.get_personal_genome(self.reference_fa, snp_vcf, encode=False)

        # chr3, chr4, chr5 should remain unchanged
        ref_sequences = Fasta(self.reference_fa)

        # These chromosomes should be in the personalized genome but unchanged
        for chrom in ["chr3", "chr4", "chr5"]:
            self.assertIn(chrom, personalized)
            self.assertEqual(personalized[chrom], str(ref_sequences[chrom]))

    def test_variant_on_missing_chromosome(self):
        """Test handling of variants on chromosomes not in reference."""
        # The deletion VCF has a variant on chr9 which doesn't exist in test_genome.fa
        del_vcf = os.path.join(self.data_dir, "del", "del.vcf")

        # Should handle gracefully without error
        personalized = sl.get_personal_genome(self.reference_fa, del_vcf, encode=False)

        # Should only have chromosomes that exist in reference
        self.assertNotIn("chr9", personalized)

        # Other chromosomes should still be processed correctly
        expected = Fasta(os.path.join(self.data_dir, "del", "del_expected_output.fa"))
        for chrom in ["chr1", "chr2"]:
            self.assertEqual(personalized[chrom], str(expected[chrom]))

    def test_chromosomes_in_personalized_genome(self):
        """Test that all reference chromosomes appear in personalized genome."""
        # Even with no variants, all chromosomes should be present
        empty_variants = pd.DataFrame(
            {"chrom": [], "pos": [], "id": [], "ref": [], "alt": []}
        )

        personalized = sl.get_personal_genome(
            self.reference_fa, empty_variants, encode=False
        )
        ref_sequences = Fasta(self.reference_fa)

        # All chromosomes from reference should be in personalized
        for (
            chrom
        ) in ref_sequences.keys():  # Use .keys() to get chromosome names as strings
            self.assertIn(chrom, personalized)
            self.assertEqual(personalized[chrom], str(ref_sequences[chrom]))


if __name__ == "__main__":
    unittest.main()
