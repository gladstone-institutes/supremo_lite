"""
Test BND sequence generation functionality in get_alt_sequences, get_ref_sequences, and get_alt_ref_sequences.
Uses existing VCF files in tests/data for all test data.
"""

import supremo_lite as sl


class TestBNDSequenceGeneration:
    """Test BND sequence generation in main sequence functions."""

    def setup_method(self):
        """Set up test data for BND sequence generation."""
        # Use existing test files
        self.reference = "tests/data/test_genome.fa"
        self.bnd_vcf_path = "tests/data/bnd/bnd.vcf"
        self.snp_vcf_path = "tests/data/snp/snp.vcf"

    def test_get_alt_sequences_with_bnds(self):
        """Test get_alt_sequences generates fusion sequences for BND variants."""
        results = list(sl.get_alt_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=False
        ))

        # Should get results for BND variants
        assert len(results) >= 1

        # Check that we get fusion sequences
        sequences, metadata = results[-1]  # BNDs are yielded last
        assert len(sequences) > 0
        assert len(metadata) > 0

        # Check metadata contains BND information
        assert 'variant_type' in metadata.columns
        assert any(metadata['variant_type'] == 'SV_BND')

        # For encode=False, sequences are list of strings (fusion sequences)
        assert isinstance(sequences, list)
        assert all(isinstance(seq, str) for seq in sequences)

    def test_get_alt_sequences_with_bnds_encoded(self):
        """Test get_alt_sequences with encoding for BND variants."""
        results = list(sl.get_alt_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=True
        ))

        # Should get results for BND variants
        assert len(results) >= 1

        sequences, metadata = results[-1]  # BNDs are yielded last

        # Check that sequences are encoded (numpy arrays or tensors)
        assert hasattr(sequences, 'shape')
        assert sequences.shape[-1] == 4  # One-hot encoding dimension
        assert sequences.shape[-2] == 50  # Sequence length

    def test_get_ref_sequences_with_bnds(self):
        """Test get_ref_sequences generates dual reference sequences for BND variants."""
        results = list(sl.get_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=False
        ))

        # Should get results for BND variants
        assert len(results) >= 1

        sequences, metadata = results[-1]  # BNDs are yielded last

        # For BNDs, sequences should be a tuple (left_refs, right_refs)
        assert isinstance(sequences, tuple)
        assert len(sequences) == 2

        left_refs, right_refs = sequences
        assert len(left_refs) > 0
        assert len(right_refs) > 0

        # Check that we have corresponding metadata
        assert len(metadata) > 0
        assert any(metadata['variant_type'] == 'SV_BND')

    def test_get_ref_sequences_with_bnds_encoded(self):
        """Test get_ref_sequences with encoding for BND variants."""
        results = list(sl.get_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=True
        ))

        # Should get results for BND variants
        assert len(results) >= 1

        sequences, metadata = results[-1]  # BNDs are yielded last

        # For BNDs, sequences should be a tuple (left_refs, right_refs)
        assert isinstance(sequences, tuple)
        assert len(sequences) == 2

        left_refs, right_refs = sequences

        # Check that sequences are encoded
        assert hasattr(left_refs, 'shape')
        assert hasattr(right_refs, 'shape')
        assert left_refs.shape[-1] == 4  # One-hot encoding
        assert right_refs.shape[-1] == 4
        assert left_refs.shape[-2] == 50  # Sequence length
        assert right_refs.shape[-2] == 50

    def test_get_alt_ref_sequences_with_bnds(self):
        """Test get_alt_ref_sequences handles BND dual reference structure."""
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=False
        ))

        # Should get results for BND variants
        assert len(results) >= 1

        alt_sequences, ref_sequences, metadata = results[-1]  # BNDs are yielded last

        # Check ALT sequences (fusion sequences)
        assert len(alt_sequences) > 0

        # Check REF sequences (dual references)
        assert isinstance(ref_sequences, tuple)
        assert len(ref_sequences) == 2

        left_refs, right_refs = ref_sequences
        assert len(left_refs) > 0
        assert len(right_refs) > 0

        # Check metadata
        assert len(metadata) > 0
        assert any(metadata['variant_type'] == 'SV_BND')

    def test_get_alt_ref_sequences_backward_compatibility(self):
        """Test that standard variants still work with existing behavior."""
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.snp_vcf_path,
            seq_len=50,
            encode=False
        ))

        assert len(results) == 1

        alt_sequences, ref_sequences, metadata = results[0]

        # For standard variants, ref_sequences should NOT be a tuple
        assert not isinstance(ref_sequences, tuple)
        assert len(alt_sequences) > 0
        assert len(ref_sequences) > 0
        assert len(metadata) > 0

        # Check that all variants are standard types (not BND)
        assert all(metadata['variant_type'] != 'SV_BND')

    def test_bnd_sequence_lengths(self):
        """Test that BND sequences have correct lengths."""
        seq_len = 60
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=seq_len,
            encode=False
        ))

        assert len(results) >= 1

        alt_sequences, ref_sequences, metadata = results[-1]  # BNDs are yielded last

        # Check ALT sequences length (fusion sequences)
        assert all(len(seq) == seq_len for seq in alt_sequences)

        # Check REF sequences length (dual references)
        assert isinstance(ref_sequences, tuple)
        left_refs, right_refs = ref_sequences

        assert all(len(seq) == seq_len for seq in left_refs)
        assert all(len(seq) == seq_len for seq in right_refs)

    def test_bnd_n_padding(self):
        """Test that BND sequences are properly handled at boundaries."""
        results = list(sl.get_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=False
        ))

        assert len(results) >= 1
        sequences, metadata = results[-1]  # BNDs are yielded last

        assert isinstance(sequences, tuple)
        left_refs, right_refs = sequences

        # Check that all sequences have the correct length
        assert all(len(seq) == 50 for seq in left_refs)
        assert all(len(seq) == 50 for seq in right_refs)

        # Check that sequences contain only valid nucleotides or N
        valid_chars = set('ACGTN')
        for seq in left_refs:
            assert set(seq).issubset(valid_chars)
        for seq in right_refs:
            assert set(seq).issubset(valid_chars)

    def test_empty_bnd_dataset(self):
        """Test handling of datasets with no BND variants."""
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=self.snp_vcf_path,
            seq_len=50,
            encode=False
        ))

        assert len(results) == 1

        alt_sequences, ref_sequences, metadata = results[0]

        # Should be standard format (not tuple for ref_sequences)
        assert not isinstance(ref_sequences, tuple)
        assert len(alt_sequences) > 0
        assert len(ref_sequences) > 0
        assert all(metadata['variant_type'] != 'SV_BND')

    def test_bnd_metadata_completeness(self):
        """Test that BND metadata contains required fields."""
        results = list(sl.get_alt_sequences(
            reference_fn=self.reference,
            variants_fn=self.bnd_vcf_path,
            seq_len=50,
            encode=False
        ))

        assert len(results) >= 1
        sequences, metadata = results[-1]  # BNDs are yielded last

        # Check that BND metadata has expected columns
        required_columns = [
            'chrom', 'variant_pos1', 'variant_type',
            'mate_chrom', 'mate_pos', 'orientation_1', 'orientation_2'
        ]

        for col in required_columns:
            assert col in metadata.columns, f"Missing column: {col}"

        # Check that we have BND variants
        bnd_rows = metadata[metadata['variant_type'] == 'SV_BND']
        assert len(bnd_rows) > 0

        # Check that mate information is populated
        assert all(bnd_rows['mate_chrom'].notna())
        assert all(bnd_rows['mate_pos'].notna())