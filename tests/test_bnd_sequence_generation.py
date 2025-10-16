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

    def test_get_alt_ref_sequences_with_derived_inv(self):
        """Test get_alt_ref_sequences with derived SV_INV variants from BND preprocessing."""
        # Use inv.vcf which contains BNDs that get preprocessed into synthetic SV_INV variants
        inv_vcf_path = "tests/data/inv/inv.vcf"

        # Test with encoding disabled first
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=inv_vcf_path,
            seq_len=50,
            encode=False
        ))

        # Should get results (standard variants chunk)
        assert len(results) >= 1

        # Find the chunk with SV_INV variants (derived from BND preprocessing)
        sv_inv_chunk = None
        for alt_sequences, ref_sequences, metadata in results:
            if any(metadata['variant_type'] == 'SV_INV'):
                sv_inv_chunk = (alt_sequences, ref_sequences, metadata)
                break

        assert sv_inv_chunk is not None, "Should find chunk with SV_INV variants derived from BND preprocessing"

        alt_sequences, ref_sequences, metadata = sv_inv_chunk

        # Check ALT sequences (should be regular list for SV_INV)
        assert len(alt_sequences) > 0
        assert isinstance(alt_sequences, list)
        # Extract sequence strings from tuples
        assert all(isinstance(seq, tuple) for seq in alt_sequences)
        alt_strings = [seq[3] for seq in alt_sequences]
        assert all(isinstance(seq, str) for seq in alt_strings)
        assert all(len(seq) == 50 for seq in alt_strings)

        # Check REF sequences (should be single list, NOT tuple for synthetic SV_INV)
        assert not isinstance(ref_sequences, tuple), "Synthetic SV_INV should have single reference structure"
        assert isinstance(ref_sequences, list)
        assert len(ref_sequences) > 0
        # Extract sequence strings from tuples
        assert all(isinstance(seq, tuple) for seq in ref_sequences)
        ref_strings = [seq[3] for seq in ref_sequences]
        assert all(isinstance(seq, str) for seq in ref_strings)
        assert all(len(seq) == 50 for seq in ref_strings)

        # Check metadata contains SV_INV variants
        assert 'variant_type' in metadata.columns
        inv_variants = metadata[metadata['variant_type'] == 'SV_INV']
        assert len(inv_variants) > 0, "Should have SV_INV variants in metadata"

        # Test with encoding enabled
        results_encoded = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=inv_vcf_path,
            seq_len=50,
            encode=True
        ))

        # Find the encoded SV_INV chunk
        encoded_sv_inv_chunk = None
        for alt_sequences, ref_sequences, metadata in results_encoded:
            if any(metadata['variant_type'] == 'SV_INV'):
                encoded_sv_inv_chunk = (alt_sequences, ref_sequences, metadata)
                break

        assert encoded_sv_inv_chunk is not None
        alt_encoded, ref_encoded, metadata_encoded = encoded_sv_inv_chunk

        # Check encoded ALT sequences
        assert hasattr(alt_encoded, 'shape')
        assert alt_encoded.shape[-1] == 4  # One-hot encoding dimension
        assert alt_encoded.shape[-2] == 50  # Sequence length

        # Check encoded REF sequences (should be single array, NOT tuple)
        assert not isinstance(ref_encoded, tuple), "Encoded synthetic SV_INV should have single reference structure"
        assert hasattr(ref_encoded, 'shape')
        assert ref_encoded.shape[-1] == 4  # One-hot encoding
        assert ref_encoded.shape[-2] == 50  # Sequence length

    def test_get_alt_ref_sequences_with_derived_dup(self):
        """Test get_alt_ref_sequences with derived SV_DUP variants from BND preprocessing."""
        # Use dup.vcf which contains BNDs that get preprocessed into synthetic SV_DUP variants
        dup_vcf_path = "tests/data/dup/dup.vcf"

        # Test with encoding disabled first
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=dup_vcf_path,
            seq_len=50,
            encode=False
        ))

        # Should get results (standard variants chunk)
        assert len(results) >= 1

        # Find the chunk with SV_DUP variants (derived from BND preprocessing)
        sv_dup_chunk = None
        for alt_sequences, ref_sequences, metadata in results:
            if any(metadata['variant_type'] == 'SV_DUP'):
                sv_dup_chunk = (alt_sequences, ref_sequences, metadata)
                break

        assert sv_dup_chunk is not None, "Should find chunk with SV_DUP variants derived from BND preprocessing"

        alt_sequences, ref_sequences, metadata = sv_dup_chunk

        # Check ALT sequences (should be regular list for SV_DUP)
        assert len(alt_sequences) > 0
        assert isinstance(alt_sequences, list)
        # Extract sequence strings from tuples
        assert all(isinstance(seq, tuple) for seq in alt_sequences)
        alt_strings = [seq[3] for seq in alt_sequences]
        assert all(isinstance(seq, str) for seq in alt_strings)
        assert all(len(seq) == 50 for seq in alt_strings)

        # Check REF sequences (should be single list, NOT tuple for synthetic SV_DUP)
        assert not isinstance(ref_sequences, tuple), "Synthetic SV_DUP should have single reference structure"
        assert isinstance(ref_sequences, list)
        assert len(ref_sequences) > 0
        # Extract sequence strings from tuples
        assert all(isinstance(seq, tuple) for seq in ref_sequences)
        ref_strings = [seq[3] for seq in ref_sequences]
        assert all(isinstance(seq, str) for seq in ref_strings)
        assert all(len(seq) == 50 for seq in ref_strings)

        # Check metadata contains SV_DUP variants
        assert 'variant_type' in metadata.columns
        dup_variants = metadata[metadata['variant_type'] == 'SV_DUP']
        assert len(dup_variants) > 0, "Should have SV_DUP variants in metadata"

        # Test with encoding enabled
        results_encoded = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference,
            variants_fn=dup_vcf_path,
            seq_len=50,
            encode=True
        ))

        # Find the encoded SV_DUP chunk
        encoded_sv_dup_chunk = None
        for alt_sequences, ref_sequences, metadata in results_encoded:
            if any(metadata['variant_type'] == 'SV_DUP'):
                encoded_sv_dup_chunk = (alt_sequences, ref_sequences, metadata)
                break

        assert encoded_sv_dup_chunk is not None
        alt_encoded, ref_encoded, metadata_encoded = encoded_sv_dup_chunk

        # Check encoded ALT sequences
        assert hasattr(alt_encoded, 'shape')
        assert alt_encoded.shape[-1] == 4  # One-hot encoding dimension
        assert alt_encoded.shape[-2] == 50  # Sequence length

        # Check encoded REF sequences (should be single array, NOT tuple)
        assert not isinstance(ref_encoded, tuple), "Encoded synthetic SV_DUP should have single reference structure"
        assert hasattr(ref_encoded, 'shape')
        assert ref_encoded.shape[-1] == 4  # One-hot encoding
        assert ref_encoded.shape[-2] == 50  # Sequence length

    def test_get_alt_ref_sequences_sv_types_comparison(self):
        """Test get_alt_ref_sequences across all SV types to verify consistent behavior."""
        # Test all three SV types and compare their output structures
        test_cases = [
            ("tests/data/inv/inv.vcf", "SV_INV", False),  # Synthetic from BND preprocessing, single ref structure
            ("tests/data/dup/dup.vcf", "SV_DUP", False),  # Synthetic from BND preprocessing, single ref structure
            ("tests/data/bnd/bnd.vcf", "SV_BND", True)    # True breakends, dual ref structure
        ]

        results_summary = {}

        for vcf_path, expected_variant_type, should_have_dual_refs in test_cases:
            # Test with encode=False
            results = list(sl.get_alt_ref_sequences(
                reference_fn=self.reference,
                variants_fn=vcf_path,
                seq_len=60,  # Use different length to test consistency
                encode=False
            ))

            # Find the chunk with the expected variant type
            target_chunk = None
            for alt_sequences, ref_sequences, metadata in results:
                if any(metadata['variant_type'] == expected_variant_type):
                    target_chunk = (alt_sequences, ref_sequences, metadata)
                    break

            assert target_chunk is not None, f"Should find chunk with {expected_variant_type} variants in {vcf_path}"

            alt_sequences, ref_sequences, metadata = target_chunk

            # Verify ALT sequences structure is consistent
            assert len(alt_sequences) > 0
            assert isinstance(alt_sequences, list)
            if expected_variant_type == "SV_BND":
                # True BND variants return fusion sequences as strings
                assert all(isinstance(seq, str) for seq in alt_sequences)
                assert all(len(seq) == 60 for seq in alt_sequences)
            else:
                # Standard variants return tuples, extract sequence strings
                assert all(isinstance(seq, tuple) for seq in alt_sequences)
                seq_strings = [seq[3] for seq in alt_sequences]  # Extract sequence strings
                assert all(isinstance(seq, str) for seq in seq_strings)
                assert all(len(seq) == 60 for seq in seq_strings)

            # Verify REF sequences structure based on variant type
            if should_have_dual_refs:
                assert isinstance(ref_sequences, tuple), f"{expected_variant_type} should have dual reference structure"
                assert len(ref_sequences) == 2, f"{expected_variant_type} should have exactly 2 reference arrays"
                left_refs, right_refs = ref_sequences
                assert len(left_refs) > 0
                assert len(right_refs) > 0
                # For BND, ref sequences are strings
                assert all(isinstance(seq, str) for seq in left_refs)
                assert all(isinstance(seq, str) for seq in right_refs)
                assert all(len(seq) == 60 for seq in left_refs)
                assert all(len(seq) == 60 for seq in right_refs)
                ref_structure = "dual"
            else:
                assert not isinstance(ref_sequences, tuple), f"{expected_variant_type} should NOT have dual reference structure"
                assert len(ref_sequences) > 0
                # For standard variants, extract sequence strings from tuples
                assert all(isinstance(seq, tuple) for seq in ref_sequences)
                ref_strings = [seq[3] for seq in ref_sequences]
                assert all(isinstance(seq, str) for seq in ref_strings)
                assert all(len(seq) == 60 for seq in ref_strings)
                ref_structure = "single"

            # Verify metadata contains expected variant types
            variant_types = set(metadata['variant_type'])
            assert expected_variant_type in variant_types, f"Should have {expected_variant_type} in metadata"

            # Store results for summary verification
            results_summary[expected_variant_type] = {
                'vcf_path': vcf_path,
                'alt_count': len(alt_sequences),
                'ref_structure': ref_structure,
                'variant_count': len(metadata[metadata['variant_type'] == expected_variant_type]),
                'metadata_columns': list(metadata.columns)
            }

        # Verify consistency across variant types (metadata columns can differ between BND and standard variants)

        # Verify dual reference structure is only for true SV_BND (translocations)
        assert results_summary['SV_INV']['ref_structure'] == 'single'  # Synthetic from BND preprocessing
        assert results_summary['SV_DUP']['ref_structure'] == 'single'  # Synthetic from BND preprocessing
        assert results_summary['SV_BND']['ref_structure'] == 'dual'    # True breakends/translocations

        # Test with encoding enabled for consistency
        for vcf_path, expected_variant_type, should_have_dual_refs in test_cases:
            results_encoded = list(sl.get_alt_ref_sequences(
                reference_fn=self.reference,
                variants_fn=vcf_path,
                seq_len=60,
                encode=True
            ))

            # Find encoded chunk
            encoded_chunk = None
            for alt_sequences, ref_sequences, metadata in results_encoded:
                if any(metadata['variant_type'] == expected_variant_type):
                    encoded_chunk = (alt_sequences, ref_sequences, metadata)
                    break

            assert encoded_chunk is not None
            alt_encoded, ref_encoded, metadata_encoded = encoded_chunk

            # Verify ALT encoding consistency
            assert hasattr(alt_encoded, 'shape')
            assert alt_encoded.shape[-1] == 4  # One-hot encoding
            assert alt_encoded.shape[-2] == 60  # Sequence length

            # Verify REF encoding structure matches expected pattern
            if should_have_dual_refs:
                assert isinstance(ref_encoded, tuple), f"Encoded {expected_variant_type} should maintain dual reference structure"
                left_encoded, right_encoded = ref_encoded
                assert hasattr(left_encoded, 'shape')
                assert hasattr(right_encoded, 'shape')
                assert left_encoded.shape[-1] == 4
                assert right_encoded.shape[-1] == 4
                assert left_encoded.shape[-2] == 60
                assert right_encoded.shape[-2] == 60
            else:
                assert not isinstance(ref_encoded, tuple), f"Encoded {expected_variant_type} should NOT have dual reference structure"
                assert hasattr(ref_encoded, 'shape')
                assert ref_encoded.shape[-1] == 4
                assert ref_encoded.shape[-2] == 60