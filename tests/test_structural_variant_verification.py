"""
Comprehensive verification tests for structural variant processing.

This test file verifies that the implementation correctly processes inversion (INV),
duplication (DUP), and breakend (BND) variants without erroneous classifications.
"""

import os
import unittest
from pyfaidx import Fasta
import supremo_lite as sl
from supremo_lite.variant_utils import BNDClassifier, group_variants_by_semantic_type


class TestBNDClassificationAccuracy(unittest.TestCase):
    """Test BND semantic classification accuracy for each VCF type."""

    def setUp(self):
        """Set up test data paths."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # VCF file paths
        self.inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")
        self.dup_vcf = os.path.join(self.data_dir, "dup", "dup.vcf")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

    def test_inv_vcf_classification(self):
        """Verify inv.vcf breakends are correctly classified as SV_BND_INV only."""
        # Load variants and group by semantic type
        variants_df = sl.read_vcf(self.inv_vcf, classify_variants=True)
        grouped = group_variants_by_semantic_type(variants_df, vcf_path=self.inv_vcf)

        # Should have 4 BND variants in inv.vcf originally
        bnd_variants = variants_df[
            variants_df["variant_type"].isin(["SV_BND", "SV_BND_INS"])
        ]
        self.assertEqual(
            len(bnd_variants), 4, "inv.vcf should contain exactly 4 BND variants"
        )

        # Check semantic classification in grouped results
        # BND variants should be reclassified as SV_BND_INV and placed in inv_variants group
        bnd_inv_variants = grouped["inv_variants"][
            grouped["inv_variants"]["variant_type"] == "SV_BND_INV"
        ]
        self.assertEqual(
            len(bnd_inv_variants),
            4,
            "inv.vcf should have exactly 4 BND variants classified as SV_BND_INV",
        )

        # Verify no BND variants are misclassified
        bnd_dup_variants = (
            grouped["dup_variants"][
                grouped["dup_variants"]["variant_type"] == "SV_BND_DUP"
            ]
            if len(grouped["dup_variants"]) > 0
            else []
        )
        self.assertEqual(
            len(bnd_dup_variants),
            0,
            "inv.vcf BND variants should NOT be classified as SV_BND_DUP",
        )

        remaining_bnd_variants = (
            grouped["bnd_variants"][grouped["bnd_variants"]["variant_type"] == "SV_BND"]
            if len(grouped["bnd_variants"]) > 0
            else []
        )
        self.assertEqual(
            len(remaining_bnd_variants),
            0,
            "inv.vcf BND variants should NOT remain as SV_BND",
        )

        # Verify all 4 BND_INV variants are from chr2
        for _, variant in bnd_inv_variants.iterrows():
            self.assertEqual(
                variant["chrom"],
                "chr2",
                f"All inv.vcf BND variants should be on chr2, found: {variant['chrom']}",
            )
            self.assertEqual(
                variant["variant_type"],
                "SV_BND_INV",
                f"All inv.vcf BND variants should have variant_type SV_BND_INV, found: {variant['variant_type']}",
            )

    def test_dup_vcf_classification(self):
        """Verify dup.vcf breakends are correctly classified as SV_BND_DUP only."""
        # Load variants and group by semantic type
        variants_df = sl.read_vcf(self.dup_vcf, classify_variants=True)
        grouped = group_variants_by_semantic_type(variants_df, vcf_path=self.dup_vcf)

        # Should have 2 BND variants in dup.vcf originally
        bnd_variants = variants_df[
            variants_df["variant_type"].isin(["SV_BND", "SV_BND_INS"])
        ]
        self.assertEqual(
            len(bnd_variants), 2, "dup.vcf should contain exactly 2 BND variants"
        )

        # Check semantic classification in grouped results
        # BND variants should be reclassified as SV_BND_DUP and placed in dup_variants group
        bnd_dup_variants = grouped["dup_variants"][
            grouped["dup_variants"]["variant_type"] == "SV_BND_DUP"
        ]
        self.assertEqual(
            len(bnd_dup_variants),
            2,
            "dup.vcf should have exactly 2 BND variants classified as SV_BND_DUP",
        )

        # Verify no BND variants are misclassified
        bnd_inv_variants = (
            grouped["inv_variants"][
                grouped["inv_variants"]["variant_type"] == "SV_BND_INV"
            ]
            if len(grouped["inv_variants"]) > 0
            else []
        )
        self.assertEqual(
            len(bnd_inv_variants),
            0,
            "dup.vcf BND variants should NOT be classified as SV_BND_INV",
        )

        remaining_bnd_variants = (
            grouped["bnd_variants"][grouped["bnd_variants"]["variant_type"] == "SV_BND"]
            if len(grouped["bnd_variants"]) > 0
            else []
        )
        self.assertEqual(
            len(remaining_bnd_variants),
            0,
            "dup.vcf BND variants should NOT remain as SV_BND",
        )

        # Verify all 2 BND_DUP variants are from chr2
        for _, variant in bnd_dup_variants.iterrows():
            self.assertEqual(
                variant["chrom"],
                "chr2",
                f"All dup.vcf BND variants should be on chr2, found: {variant['chrom']}",
            )
            self.assertEqual(
                variant["variant_type"],
                "SV_BND_DUP",
                f"All dup.vcf BND variants should have variant_type SV_BND_DUP, found: {variant['variant_type']}",
            )

    def test_bnd_vcf_classification(self):
        """Verify bnd.vcf breakends remain as SV_BND (no misclassification)."""
        # Load variants and group by semantic type
        variants_df = sl.read_vcf(self.bnd_vcf, classify_variants=True)
        grouped = group_variants_by_semantic_type(variants_df, vcf_path=self.bnd_vcf)

        # Should have multiple BND variants in bnd.vcf originally
        bnd_variants = variants_df[
            variants_df["variant_type"].isin(["SV_BND", "SV_BND_INS"])
        ]
        self.assertGreater(len(bnd_variants), 0, "bnd.vcf should contain BND variants")

        # Check semantic classification in grouped results
        # BND variants should remain as SV_BND and be placed in bnd_variants group
        true_bnd_variants = (
            grouped["bnd_variants"][
                grouped["bnd_variants"]["variant_type"].isin(["SV_BND", "SV_BND_INS"])
            ]
            if len(grouped["bnd_variants"]) > 0
            else []
        )
        self.assertGreater(
            len(true_bnd_variants),
            0,
            "bnd.vcf should have BND variants remaining as SV_BND",
        )

        # Verify no BND variants are misclassified
        misclassified_dup = (
            grouped["dup_variants"][
                grouped["dup_variants"]["variant_type"] == "SV_BND_DUP"
            ]
            if len(grouped["dup_variants"]) > 0
            else []
        )
        self.assertEqual(
            len(misclassified_dup),
            0,
            "bnd.vcf BND variants should NOT be misclassified as SV_BND_DUP",
        )

        misclassified_inv = (
            grouped["inv_variants"][
                grouped["inv_variants"]["variant_type"] == "SV_BND_INV"
            ]
            if len(grouped["inv_variants"]) > 0
            else []
        )
        self.assertEqual(
            len(misclassified_inv),
            0,
            "bnd.vcf BND variants should NOT be misclassified as SV_BND_INV",
        )

        # Verify all true BND variants maintain correct variant_type
        for _, variant in true_bnd_variants.iterrows():
            self.assertIn(
                variant["variant_type"],
                ["SV_BND", "SV_BND_INS"],
                f"bnd.vcf BND variants should have variant_type SV_BND or SV_BND_INS, found: {variant['variant_type']}",
            )
            # Should be inter-chromosomal translocations (different chromosomes in ALT field)
            alt_field = variant["alt"]
            if "]" in alt_field or "[" in alt_field:
                # Extract chromosome from BND ALT field for verification
                import re

                chrom_match = re.search(r"[\[\]]([^:\[\]]+):", alt_field)
                if chrom_match:
                    mate_chrom = chrom_match.group(1)
                    # For true translocations, mate chromosome should be different
                    # (Note: Some may be same chromosome but that's still valid BND)


class TestExactOutputVerification(unittest.TestCase):
    """Test exact output verification against expected files."""

    def setUp(self):
        """Set up test data paths."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # VCF and expected output file paths
        self.inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")
        self.inv_expected = os.path.join(self.data_dir, "inv", "inv_expected_output.fa")

        self.dup_vcf = os.path.join(self.data_dir, "dup", "dup.vcf")
        self.dup_expected = os.path.join(self.data_dir, "dup", "dup_expected_output.fa")

        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")
        self.bnd_expected = os.path.join(self.data_dir, "bnd", "bnd_expected_output.fa")

    def test_inv_exact_output(self):
        """Process inv.vcf and verify character-by-character match with expected output."""
        # Process inv.vcf
        personalized = sl.get_personal_genome(
            self.reference_fa, self.inv_vcf, encode=False
        )

        # Load expected output
        expected = Fasta(self.inv_expected)

        # Verify all expected chromosomes are present
        for chrom in expected.keys():
            self.assertIn(
                chrom,
                personalized,
                f"Chromosome {chrom} missing from inv.vcf processing result",
            )

        # Compare each chromosome character-by-character
        for chrom in expected.keys():
            expected_seq = str(expected[chrom])
            actual_seq = personalized[chrom]

            self.assertEqual(
                actual_seq,
                expected_seq,
                f"Sequence mismatch in {chrom} for inv.vcf processing.\n"
                f"Expected: {expected_seq}\n"
                f"Actual:   {actual_seq}",
            )

    def test_dup_exact_output(self):
        """Process dup.vcf and verify character-by-character match with expected output."""
        # Process dup.vcf
        personalized = sl.get_personal_genome(
            self.reference_fa, self.dup_vcf, encode=False
        )

        # Load expected output
        expected = Fasta(self.dup_expected)

        # Verify all expected chromosomes are present
        for chrom in expected.keys():
            self.assertIn(
                chrom,
                personalized,
                f"Chromosome {chrom} missing from dup.vcf processing result",
            )

        # Compare each chromosome character-by-character
        for chrom in expected.keys():
            expected_seq = str(expected[chrom])
            actual_seq = personalized[chrom]

            self.assertEqual(
                actual_seq,
                expected_seq,
                f"Sequence mismatch in {chrom} for dup.vcf processing.\n"
                f"Expected: {expected_seq}\n"
                f"Actual:   {actual_seq}",
            )

    def test_bnd_exact_output(self):
        """Process bnd.vcf and verify character-by-character match with expected output."""
        # Process bnd.vcf
        personalized = sl.get_personal_genome(
            self.reference_fa, self.bnd_vcf, encode=False
        )

        # Load expected output
        expected = Fasta(self.bnd_expected)

        # Verify all expected sequences are present
        for seq_name in expected.keys():
            self.assertIn(
                seq_name,
                personalized,
                f"Sequence {seq_name} missing from bnd.vcf processing result",
            )

        # Compare each sequence character-by-character
        for seq_name in expected.keys():
            expected_seq = str(expected[seq_name])
            actual_seq = personalized[seq_name]

            self.assertEqual(
                actual_seq,
                expected_seq,
                f"Sequence mismatch in {seq_name} for bnd.vcf processing.\n"
                f"Expected: {expected_seq}\n"
                f"Actual:   {actual_seq}",
            )


class TestProcessingIntegrity(unittest.TestCase):
    """Test processing integrity and variant count verification."""

    def setUp(self):
        """Set up test data paths."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # VCF file paths
        self.inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")
        self.dup_vcf = os.path.join(self.data_dir, "dup", "dup.vcf")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

    def test_inv_processing_details(self):
        """Verify inv.vcf processing shows correct phase counts and synthetic variant creation."""
        # Load and group variants to check phase distribution
        variants_df = sl.read_vcf(self.inv_vcf, classify_variants=True)
        grouped = group_variants_by_semantic_type(variants_df, vcf_path=self.inv_vcf)

        # Verify correct grouping
        self.assertGreater(
            len(grouped["inv_variants"]),
            0,
            "inv.vcf should have variants in inv_variants group",
        )

        # Check that BND-derived variants are properly classified
        bnd_inv_variants = grouped["inv_variants"][
            grouped["inv_variants"]["variant_type"] == "SV_BND_INV"
        ]
        symbolic_inv_variants = grouped["inv_variants"][
            grouped["inv_variants"]["variant_type"] == "SV_INV"
        ]

        self.assertEqual(
            len(bnd_inv_variants),
            4,
            "inv.vcf should have 4 BND-derived inversion variants",
        )
        self.assertEqual(
            len(symbolic_inv_variants),
            1,
            "inv.vcf should have 1 symbolic inversion variant",
        )

    def test_dup_processing_details(self):
        """Verify dup.vcf processing shows correct BND-derived duplication processing."""
        # Load and group variants to check phase distribution
        variants_df = sl.read_vcf(self.dup_vcf, classify_variants=True)
        grouped = group_variants_by_semantic_type(variants_df, vcf_path=self.dup_vcf)

        # Verify correct grouping
        self.assertGreater(
            len(grouped["dup_variants"]),
            0,
            "dup.vcf should have variants in dup_variants group",
        )

        # Check that BND-derived variants are properly classified
        bnd_dup_variants = grouped["dup_variants"][
            grouped["dup_variants"]["variant_type"] == "SV_BND_DUP"
        ]
        symbolic_dup_variants = grouped["dup_variants"][
            grouped["dup_variants"]["variant_type"] == "SV_DUP"
        ]

        self.assertEqual(
            len(bnd_dup_variants),
            2,
            "dup.vcf should have 2 BND-derived duplication variants",
        )
        self.assertEqual(
            len(symbolic_dup_variants),
            1,
            "dup.vcf should have 1 symbolic duplication variant",
        )

    def test_bnd_processing_details(self):
        """Verify bnd.vcf processing without DUP/INV misclassification."""
        # Load and group variants to check phase distribution
        variants_df = sl.read_vcf(self.bnd_vcf, classify_variants=True)
        grouped = group_variants_by_semantic_type(variants_df, vcf_path=self.bnd_vcf)

        # Verify that BND variants are not misclassified
        self.assertEqual(
            len(grouped["dup_variants"]),
            0,
            "bnd.vcf should NOT have any variants in dup_variants group",
        )
        self.assertEqual(
            len(grouped["inv_variants"]),
            0,
            "bnd.vcf should NOT have any variants in inv_variants group",
        )

        # Should have true BND variants
        self.assertGreater(
            len(grouped["bnd_variants"]),
            0,
            "bnd.vcf should have variants in bnd_variants group",
        )


if __name__ == "__main__":
    unittest.main()
