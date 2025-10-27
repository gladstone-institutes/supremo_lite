"""
Tests for BND (breakend) variant support in supremo_lite.

This file tests all aspects of BND variant processing including:
- Parsing of breakend ALT fields
- Creation of breakend pairs
- Application of novel adjacencies
- Integration with get_personal_genome
"""

import unittest
import os
import supremo_lite as sl
from supremo_lite.variant_utils import (
    parse_breakend_alt,
    create_breakend_pairs,
    load_breakend_variants,
    BreakendVariant,
    BreakendPair,
    read_vcf,
    Breakend,
)
from supremo_lite.personalize import ChimericSequenceBuilder
import warnings


class TestBreakendParsing(unittest.TestCase):
    """Test breakend ALT field parsing functions."""

    def setUp(self):
        """Set up test data paths."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, "data")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

        # Load the actual BND variants from the VCF file
        self.variants_df = read_vcf(
            self.bnd_vcf, include_info=True, classify_variants=True
        )
        self.bnd_variants = self.variants_df[
            self.variants_df["variant_type"].isin(["SV_BND", "SV_BND_INS"])
        ]

    def test_parse_breakend_alt_patterns(self):
        """Test parsing of different breakend ALT patterns using actual VCF data."""
        # Test t]p] format using bnd_W: A]chr5:7]
        bnd_w = self.bnd_variants[self.bnd_variants["id"] == "bnd_W"].iloc[0]
        result = parse_breakend_alt(bnd_w["alt"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["mate_chrom"], "chr5")
        self.assertEqual(result["mate_pos"], 7)
        self.assertEqual(result["orientation"], "t]p]")
        self.assertEqual(result["inserted_seq"], "")

        # Test ]p]t format using bnd_V: ]chr3:20]T
        bnd_v = self.bnd_variants[self.bnd_variants["id"] == "bnd_V"].iloc[0]
        result = parse_breakend_alt(bnd_v["alt"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["mate_chrom"], "chr3")
        self.assertEqual(result["mate_pos"], 20)
        self.assertEqual(result["orientation"], "]p]t")
        self.assertEqual(result["inserted_seq"], "")

        # Test t[p[ format using bnd_U: A[chr1:11[
        bnd_u = self.bnd_variants[self.bnd_variants["id"] == "bnd_U"].iloc[0]
        result = parse_breakend_alt(bnd_u["alt"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["mate_chrom"], "chr1")
        self.assertEqual(result["mate_pos"], 11)
        self.assertEqual(result["orientation"], "t[p[")
        self.assertEqual(result["inserted_seq"], "")

        # Test [p[t format using bnd_X: [chr5:8[A
        bnd_x = self.bnd_variants[self.bnd_variants["id"] == "bnd_X"].iloc[0]
        result = parse_breakend_alt(bnd_x["alt"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["mate_chrom"], "chr5")
        self.assertEqual(result["mate_pos"], 8)
        self.assertEqual(result["orientation"], "[p[t")
        self.assertEqual(result["inserted_seq"], "")

    def test_parse_breakend_alt_with_insertions(self):
        """Test parsing breakend ALT fields with inserted sequences using actual VCF data."""
        # Test with inserted sequence using bnd_B: ]chr2:20]ATCGT
        bnd_b = self.bnd_variants[self.bnd_variants["id"] == "bnd_B"].iloc[0]
        result = parse_breakend_alt(bnd_b["alt"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["mate_chrom"], "chr2")
        self.assertEqual(result["mate_pos"], 20)
        self.assertEqual(
            result["inserted_seq"], "ATCG"
        )  # Removes last base (reference)

    def test_parse_breakend_alt_invalid(self):
        """Test parsing of invalid breakend ALT fields."""
        # Invalid format
        result = parse_breakend_alt("INVALID")
        self.assertFalse(result["is_valid"])

        # Empty string
        result = parse_breakend_alt("")
        self.assertFalse(result["is_valid"])

        # None input
        result = parse_breakend_alt(None)
        self.assertFalse(result["is_valid"])

    def test_orientation_naming_scheme(self):
        """Test that orientation names match the VCF 4.2 specification using actual VCF data."""
        # Test all four breakend patterns with VCF 4.2 naming scheme using actual variants

        # t[p[ pattern should be 't[p[' - using bnd_U
        bnd_u = self.bnd_variants[self.bnd_variants["id"] == "bnd_U"].iloc[0]
        result = parse_breakend_alt(bnd_u["alt"])
        self.assertEqual(result["orientation"], "t[p[")

        # t]p] pattern should be 't]p]' - using bnd_W
        bnd_w = self.bnd_variants[self.bnd_variants["id"] == "bnd_W"].iloc[0]
        result = parse_breakend_alt(bnd_w["alt"])
        self.assertEqual(result["orientation"], "t]p]")

        # ]p]t pattern should be ']p]t' - using bnd_V
        bnd_v = self.bnd_variants[self.bnd_variants["id"] == "bnd_V"].iloc[0]
        result = parse_breakend_alt(bnd_v["alt"])
        self.assertEqual(result["orientation"], "]p]t")

        # [p[t pattern should be '[p[t' - using bnd_X
        bnd_x = self.bnd_variants[self.bnd_variants["id"] == "bnd_X"].iloc[0]
        result = parse_breakend_alt(bnd_x["alt"])
        self.assertEqual(result["orientation"], "[p[t")


class TestBreakendDataStructures(unittest.TestCase):
    """Test BreakendVariant and BreakendPair data structures."""

    def setUp(self):
        """Set up test data paths."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, "data")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

        # Load the actual BND variants from the VCF file
        self.variants_df = read_vcf(
            self.bnd_vcf, include_info=True, classify_variants=True
        )
        self.bnd_variants = self.variants_df[
            self.variants_df["variant_type"] == "SV_BND"
        ]

    def test_breakend_variant_creation(self):
        """Test creation of BreakendVariant objects using actual VCF data."""
        # Use actual bnd_W from VCF: chr1:10 A]chr5:7]
        bnd_w_row = self.bnd_variants[self.bnd_variants["id"] == "bnd_W"].iloc[0]
        bnd = BreakendVariant(
            id=bnd_w_row["id"],
            chrom=bnd_w_row["chrom"],
            pos=bnd_w_row["pos1"],
            ref=bnd_w_row["ref"],
            alt=bnd_w_row["alt"],
            mate_id="bnd_Y",
            mate_chrom="chr5",
            mate_pos=7,
            orientation="t]p]",
            inserted_seq="",
            info=bnd_w_row["info"],
            variant_type=bnd_w_row["variant_type"],
        )

        self.assertEqual(bnd.id, "bnd_W")
        self.assertEqual(bnd.chrom, "chr1")
        self.assertEqual(bnd.pos, 10)
        self.assertEqual(bnd.variant_type, "SV_BND")

    def test_breakend_variant_validation(self):
        """Test validation in BreakendVariant creation."""
        # Test empty ID validation
        with self.assertRaises(ValueError):
            BreakendVariant(
                "",
                "chr1",
                20,
                "G",
                "G]chr2:40]",
                "bnd_V",
                "chr2",
                40,
                "ref_then_rc_mate",
                "",
                "",
                "SV_BND",
            )

        # Test position validation
        with self.assertRaises(ValueError):
            BreakendVariant(
                "bnd_W",
                "chr1",
                0,
                "G",
                "G]chr2:40]",
                "bnd_V",
                "chr2",
                40,
                "ref_then_rc_mate",
                "",
                "",
                "SV_BND",
            )

    def test_breakend_pair_creation(self):
        """Test creation and validation of BreakendPair objects using actual VCF data."""
        # Use actual paired variants from VCF: bnd_W and bnd_Y
        bnd_w_row = self.bnd_variants[self.bnd_variants["id"] == "bnd_W"].iloc[0]
        bnd_y_row = self.bnd_variants[self.bnd_variants["id"] == "bnd_Y"].iloc[0]

        bnd1 = BreakendVariant(
            bnd_w_row["id"],
            bnd_w_row["chrom"],
            bnd_w_row["pos1"],
            bnd_w_row["ref"],
            bnd_w_row["alt"],
            "bnd_Y",
            "chr5",
            7,
            "t]p]",
            "",
            bnd_w_row["info"],
            "SV_BND",
        )
        bnd2 = BreakendVariant(
            bnd_y_row["id"],
            bnd_y_row["chrom"],
            bnd_y_row["pos1"],
            bnd_y_row["ref"],
            bnd_y_row["alt"],
            "bnd_W",
            "chr1",
            10,
            "t]p]",
            "",
            bnd_y_row["info"],
            "SV_BND",
        )

        pair = BreakendPair(bnd1, bnd2)
        self.assertTrue(pair.is_valid)
        self.assertEqual(len(pair.validation_errors), 0)


class TestBreakendPairing(unittest.TestCase):
    """Test breakend pairing functionality."""

    def setUp(self):
        """Set up test VCF data using actual bnd.vcf file."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, "data")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

        # Load the actual BND variants from the VCF file
        self.test_df = read_vcf(self.bnd_vcf, include_info=True, classify_variants=True)
        self.bnd_variants = self.test_df[self.test_df["variant_type"] == "SV_BND"]

    def test_create_breakend_pairs(self):
        """Test creation of breakend pairs from DataFrame using actual VCF data."""
        pairs = create_breakend_pairs(self.bnd_variants)

        self.assertEqual(
            len(pairs), 3
        )  # Should create 3 pairs from 6 BND variants in the VCF

        # Check that all pairs are valid
        for pair in pairs:
            self.assertTrue(pair.is_valid)
            self.assertEqual(len(pair.validation_errors), 0)

    def test_load_breakend_variants(self):
        """Test loading and separation of BND and standard variants."""
        standard_variants, breakend_pairs = load_breakend_variants(self.bnd_vcf)

        # Check that BND variants are processed into pairs
        self.assertEqual(len(standard_variants), 0)  # No standard variants in BND VCF
        self.assertGreater(len(breakend_pairs), 0)  # Should have BND pairs

        # Check that we get the expected number of pairs
        # The VCF has 7 BND variants (6 with mates + 1 with inferred mate)
        # This should result in 4 pairs total
        self.assertEqual(
            len(breakend_pairs), 4
        )  # 4 pairs total including inferred mate


class TestBreakendApplication(unittest.TestCase):
    """Test application of breakend variants to sequences."""

    def setUp(self):
        """Set up test sequences and breakend pairs using actual VCF data."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, "data")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

        # Load actual reference sequences
        self.test_sequences = {
            "chr1": "ATGAATATAATATTTTCGAGAATTACTCCTTTTGGAAATGGAACATTATGCGTTTTAAGAGTTTCTGGTAACAATATATT",
            "chr2": "TATTTCTTATTCGTTTAAAAAAATTAATTTATTTATTTCTAAATTAAAAACGGGAACACCTCCCAGATTGGATTTAAATT",
            "chr3": "TTCTTTAATAAATACTATTAATAAATTAAAAAAATTTAATTGTTTAGATTATACAATTATTATCGCTTCGACTGCTTCAG",
            "chr4": "AGGTGGAAAAATAGGATTATTTGGTGGAGCAGGAGTTGGAAAAACAATAAATATGATGGAATTAATTAGAAATATTGCAA",
            "chr5": "ATTTTGTAAAATATTAAAAAAAAAAAAACAAAAATATCGCAAATTCCAAGTGGCATTCCCGGATGACCTGAATTTGCTTT",
        }

        # Load actual BND variants and create a simple breakend pair
        variants_df = read_vcf(self.bnd_vcf, include_info=True, classify_variants=True)
        bnd_variants = variants_df[variants_df["variant_type"] == "SV_BND"]

        # Use actual bnd_W and bnd_Y as a paired example
        bnd_w_row = bnd_variants[bnd_variants["id"] == "bnd_W"].iloc[0]
        bnd_y_row = bnd_variants[bnd_variants["id"] == "bnd_Y"].iloc[0]

        self.bnd1 = BreakendVariant(
            bnd_w_row["id"],
            bnd_w_row["chrom"],
            bnd_w_row["pos1"],
            bnd_w_row["ref"],
            bnd_w_row["alt"],
            "bnd_Y",
            "chr5",
            7,
            "t]p]",
            "",
            bnd_w_row["info"],
            "SV_BND",
        )
        self.bnd2 = BreakendVariant(
            bnd_y_row["id"],
            bnd_y_row["chrom"],
            bnd_y_row["pos1"],
            bnd_y_row["ref"],
            bnd_y_row["alt"],
            "bnd_W",
            "chr1",
            10,
            "t]p]",
            "",
            bnd_y_row["info"],
            "SV_BND",
        )
        self.test_pair = BreakendPair(self.bnd1, self.bnd2)

    def test_chimeric_sequence_builder_creation(self):
        """Test creation of ChimericSequenceBuilder."""
        builder = ChimericSequenceBuilder(self.test_sequences)

        self.assertEqual(
            len(builder.reference_sequences), 5
        )  # We now have 5 chromosomes
        self.assertEqual(len(builder.chimeric_sequences), 0)
        self.assertEqual(len(builder.sequence_segments), 0)

    def test_create_fusion_from_pair(self):
        """Test creation of fusion sequence from breakend pair using actual VCF data."""
        builder = ChimericSequenceBuilder(self.test_sequences)

        # Convert BreakendVariants to Breakend objects for compatibility
        bnd1 = Breakend.from_breakend_variant(self.bnd1, "paired")
        bnd2 = Breakend.from_breakend_variant(self.bnd2, "paired")
        pair = (bnd1, bnd2)

        fusion_name, _ = builder.create_fusion_from_pair(pair)

        self.assertIsInstance(fusion_name, str)
        self.assertIn("fusion", fusion_name)
        # Check that fusion name contains the actual chromosome names from our data
        self.assertTrue("chr1" in fusion_name or "chr5" in fusion_name)

    def test_sequence_segments_tracking(self):
        """Test that sequence segments are tracked correctly using actual VCF data."""
        builder = ChimericSequenceBuilder(self.test_sequences)

        # Convert BreakendVariants to Breakend objects for compatibility
        bnd1 = Breakend.from_breakend_variant(self.bnd1, "paired")
        bnd2 = Breakend.from_breakend_variant(self.bnd2, "paired")
        pair = (bnd1, bnd2)

        fusion_name, _ = builder.create_fusion_from_pair(pair)
        segments = builder.get_sequence_segments(fusion_name)

        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        # Just check that we have segments - the internal structure may vary
        # so we'll just verify that the basic functionality works
        self.assertTrue(len(segments) > 0)


class TestBNDIntegration(unittest.TestCase):
    """Test integration of BND variants with get_personal_genome."""

    def setUp(self):
        """Set up test data paths."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

    def test_get_personal_genome_with_bnd(self):
        """Test get_personal_genome function with BND variants."""
        # Test that the function runs without errors
        try:
            personalized = sl.get_personal_genome(
                self.reference_fa, self.bnd_vcf, encode=False, verbose=False
            )

            # Check that we get sequences back
            self.assertIsInstance(personalized, dict)
            self.assertGreater(len(personalized), 0)

            # Check that fusion sequences are present
            fusion_sequences = [
                name for name in personalized.keys() if "_fusion_" in name
            ]
            self.assertEqual(len(fusion_sequences), 4)  # Expect 4 fusion sequences

            # Check that leftover sequences are present (chr2, chr4)
            self.assertIn("chr2", personalized)
            self.assertIn("chr4", personalized)

            # Check that original chromosomes that were consumed by fusions are NOT present
            self.assertNotIn("chr1", personalized)  # chr1 consumed by fusions
            self.assertNotIn("chr3", personalized)  # chr3 consumed by fusions
            self.assertNotIn("chr5", personalized)  # chr5 consumed by fusions

        except Exception as e:
            self.fail(f"get_personal_genome with BND variants failed: {e}")

    def test_bnd_vcf_loading(self):
        """Test that BND VCF file loads correctly."""
        # Load the test BND VCF
        variants_df = read_vcf(self.bnd_vcf, include_info=True, classify_variants=True)

        # Check that BND variants are classified correctly
        bnd_variants = variants_df[variants_df["variant_type"] == "SV_BND"]
        self.assertGreater(len(bnd_variants), 0)

        # Check that all BND variants have required fields
        for _, variant in bnd_variants.iterrows():
            self.assertIsNotNone(variant["id"])
            self.assertIsNotNone(variant["alt"])
            self.assertTrue(
                variant["alt"].count("[") > 0 or variant["alt"].count("]") > 0
            )

    def test_exact_expected_output(self):
        """Test that BND processing produces exact expected output."""
        # Load expected output
        expected_output_fa = os.path.join(
            self.data_dir, "bnd", "bnd_expected_output.fa"
        )
        expected_sequences = {}

        with open(expected_output_fa, "r") as f:
            current_name = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name:
                        expected_sequences[current_name] = "".join(current_seq)
                    current_name = line[1:]  # Remove '>'
                    current_seq = []
                elif line:
                    current_seq.append(line)
            if current_name:
                expected_sequences[current_name] = "".join(current_seq)

        # Generate actual output
        actual_sequences = sl.get_personal_genome(
            self.reference_fa, self.bnd_vcf, encode=False, verbose=True
        )

        # Compare expected vs actual
        self.assertEqual(len(actual_sequences), len(expected_sequences))

        for name, expected_seq in expected_sequences.items():
            self.assertIn(name, actual_sequences, f"Missing sequence: {name}")
            actual_seq = actual_sequences[name]
            self.assertEqual(
                actual_seq,
                expected_seq,
                f"Sequence mismatch for {name}:\nExpected: {expected_seq}\nActual: {actual_seq}",
            )


if __name__ == "__main__":
    # Suppress warnings during testing
    warnings.filterwarnings("ignore")
    unittest.main()
