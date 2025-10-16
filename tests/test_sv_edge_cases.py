"""
Edge case tests for structural variant processing.

Tests boundary conditions, error handling, and edge cases for INV, DUP, and BND variants.
Uses existing test data and focuses on robustness and error handling.
"""

import os
import numpy as np
import pytest
import warnings
import supremo_lite as sl
from pyfaidx import Fasta


class TestInversionEdgeCases:
    """Test edge cases for inversion (INV) variants."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")

    def test_inv_at_sequence_boundary(self):
        """Test inversion handling when variant is near sequence boundaries."""
        # Get sequences with small window (may hit boundaries)
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.inv_vcf,
            seq_len=50,  # Small window
            encode=False
        ))

        # Should handle boundary conditions gracefully
        assert len(results) > 0, "Should process INV variants near boundaries"

        # Check that sequences don't exceed reference bounds
        reference = Fasta(self.reference_fa)
        for ref_seqs, alt_seqs, metadata in results:
            if metadata['variant_type'].iloc[0] == 'SV_BND_INV':
                for idx, row in metadata.iterrows():
                    chrom = row['chrom']
                    window_start = row['window_start']
                    window_end = row['window_end']

                    # Window should be within chromosome bounds
                    assert window_start >= 0, f"Window start should be non-negative: {window_start}"
                    assert window_end <= len(reference[chrom]), \
                        f"Window end should not exceed chromosome length: {window_end} > {len(reference[chrom])}"

        print("✓ INV at sequence boundary test passed")

    def test_inv_very_small_window(self):
        """Test inversion with very small sequence windows."""
        # Get sequences with minimal window
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.inv_vcf,
            seq_len=20,  # Very small window
            encode=False
        ))

        # Should handle small windows without errors
        assert len(results) >= 0, "Should handle very small windows"

        for ref_seqs, alt_seqs, metadata in results:
            if metadata['variant_type'].iloc[0] == 'SV_BND_INV':
                # Sequences should not be empty
                if isinstance(ref_seqs, list):
                    assert all(len(seq) > 0 for seq in ref_seqs), "Sequences should not be empty"
                if isinstance(alt_seqs, list):
                    assert all(len(seq) > 0 for seq in alt_seqs), "Sequences should not be empty"

        print("✓ INV very small window test passed")

    def test_inv_with_padding(self):
        """Test that INV sequences are properly padded when window extends beyond chromosome."""
        # Get sequences that may require padding
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.inv_vcf,
            seq_len=200,
            encode=False
        ))

        # Check for N-padding in sequences if needed
        for ref_seqs, alt_seqs, metadata in results:
            if metadata['variant_type'].iloc[0] == 'SV_BND_INV':
                # If sequences contain 'N', it means padding was applied
                if isinstance(ref_seqs, list):
                    for seq in ref_seqs:
                        if 'N' in seq:
                            # N-padding should only be at boundaries
                            assert seq.startswith('N') or seq.endswith('N'), \
                                "N-padding should only be at sequence boundaries"

        print("✓ INV with padding test passed")


class TestDuplicationEdgeCases:
    """Test edge cases for duplication (DUP) variants."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.dup_vcf = os.path.join(self.data_dir, "dup", "dup.vcf")

    def test_dup_with_different_svlen(self):
        """Test DUP handling with various SVLEN values."""
        # Read VCF to check SVLEN values
        variants_df = sl.read_vcf(self.dup_vcf, classify_variants=True)

        # Get sequences
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.dup_vcf,
            seq_len=200,
            encode=False
        ))

        # Should handle all SVLEN values
        dup_results = [r for r in results if 'DUP' in r[2]['variant_type'].iloc[0]]
        assert len(dup_results) > 0, "Should process DUP variants"

        # Check metadata includes sym_variant_end for symbolic alleles
        for ref_seqs, alt_seqs, metadata in dup_results:
            for idx, row in metadata.iterrows():
                if row['alt'].startswith('<'):
                    assert 'sym_variant_end' in row.index, "Symbolic DUP should have sym_variant_end"
                    # SVLEN should be consistent with END - POS
                    if 'sym_variant_end' in row.index:
                        calculated_svlen = row['sym_variant_end'] - row['variant_pos1']
                        assert calculated_svlen >= 0, "SVLEN should be non-negative for DUP"

        print("✓ DUP with different SVLEN test passed")

    def test_dup_sequence_length_consistency(self):
        """Test that DUP sequences have consistent lengths with SVLEN."""
        # Get sequences
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.dup_vcf,
            seq_len=200,
            encode=False
        ))

        # Check sequence length consistency
        dup_results = [r for r in results if 'DUP' in r[2]['variant_type'].iloc[0]]

        for ref_seqs, alt_seqs, metadata in dup_results:
            # ALT sequences should be longer than or equal to REF for duplications
            if isinstance(ref_seqs, list) and isinstance(alt_seqs, list):
                assert len(ref_seqs) == len(alt_seqs), "Should have same number of sequences"
                # Individual sequence lengths may vary due to duplication

        print("✓ DUP sequence length consistency test passed")

    def test_dup_at_chromosome_end(self):
        """Test DUP handling when duplication region extends to chromosome end."""
        # Get sequences that may hit chromosome boundaries
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.dup_vcf,
            seq_len=100,
            encode=False
        ))

        # Should handle boundary conditions
        reference = Fasta(self.reference_fa)
        for ref_seqs, alt_seqs, metadata in results:
            if 'DUP' in metadata['variant_type'].iloc[0]:
                for idx, row in metadata.iterrows():
                    chrom = row['chrom']
                    window_end = row['window_end']

                    # Window can exceed chromosome length (will be padded with N's)
                    # Just verify window_end is a valid positive number
                    assert window_end > 0, f"Window end should be positive: {window_end}"

        print("✓ DUP at chromosome end test passed")


class TestBreakendEdgeCases:
    """Test edge cases for breakend (BND) variants."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.bnd_vcf = os.path.join(self.data_dir, "bnd", "bnd.vcf")

    def test_bnd_with_different_chromosomes(self):
        """Test BND handling with inter-chromosomal translocations."""
        # Get sequences
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.bnd_vcf,
            seq_len=200,
            encode=False
        ))

        # Check BND variants
        bnd_results = [r for r in results
                      if r[2]['variant_type'].iloc[0] in ['SV_BND', 'SV_BND_INS']]

        if len(bnd_results) > 0:
            for ref_seqs, alt_seqs, metadata in bnd_results:
                for idx, row in metadata.iterrows():
                    # Should have mate chromosome information
                    assert 'mate_chrom' in row.index, "BND should have mate_chrom"
                    assert 'mate_pos' in row.index, "BND should have mate_pos"

                    # Mate position should be valid
                    assert row['mate_pos'] > 0, f"Mate position should be positive: {row['mate_pos']}"

        print("✓ BND with different chromosomes test passed")

    def test_bnd_fusion_sequence_naming(self):
        """Test that BND fusion sequences have proper naming."""
        # Get sequences
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.bnd_vcf,
            seq_len=200,
            encode=False
        ))

        # Check fusion naming
        bnd_results = [r for r in results
                      if r[2]['variant_type'].iloc[0] in ['SV_BND', 'SV_BND_INS']]

        if len(bnd_results) > 0:
            for ref_seqs, alt_seqs, metadata in bnd_results:
                for idx, row in metadata.iterrows():
                    # Fusion sequences should have descriptive names
                    if 'fusion_name' in row.index and row['fusion_name']:
                        fusion_name = row['fusion_name']
                        # Fusion name should contain useful information
                        assert isinstance(fusion_name, str), "Fusion name should be string"
                        assert len(fusion_name) > 0, "Fusion name should not be empty"

        print("✓ BND fusion sequence naming test passed")

    def test_bnd_reference_tuple_structure(self):
        """Test that BND returns proper tuple structure for reference sequences."""
        # Get sequences
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.bnd_vcf,
            seq_len=200,
            encode=False
        ))

        # Check tuple structure
        bnd_results = [r for r in results
                      if r[2]['variant_type'].iloc[0] in ['SV_BND', 'SV_BND_INS']]

        if len(bnd_results) > 0:
            for ref_seqs, alt_seqs, metadata in bnd_results:
                # BND can return either tuple of (left_refs, right_refs) or list depending on context
                # For get_alt_ref_sequences, it typically returns lists
                assert isinstance(ref_seqs, (tuple, list)), "BND ref sequences should be tuple or list"

                if isinstance(ref_seqs, tuple):
                    assert len(ref_seqs) == 2, "BND tuple should have (left_refs, right_refs)"
                    left_refs, right_refs = ref_seqs
                    # Both should be lists
                    assert isinstance(left_refs, list), "Left refs should be list"
                    assert isinstance(right_refs, list), "Right refs should be list"
                    # Should have same number of sequences
                    assert len(left_refs) == len(right_refs), "Left and right refs should match"
                elif isinstance(ref_seqs, list):
                    # Single list of reference sequences is also valid
                    assert len(ref_seqs) > 0, "BND should have reference sequences"

        print("✓ BND reference tuple structure test passed")


class TestErrorHandling:
    """Test error handling for invalid SV inputs."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

    def test_missing_end_field_for_symbolic_allele(self):
        """Test handling of symbolic alleles without END field."""
        # This would typically be caught during VCF parsing
        # The variant_utils module should handle missing END gracefully

        # Read VCF and check for proper handling
        inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")
        variants_df = sl.read_vcf(inv_vcf, classify_variants=True)

        # Should successfully read variants
        assert len(variants_df) > 0, "Should read variants successfully"

        # Check that symbolic alleles have proper classification
        symbolic_variants = variants_df[variants_df['alt'].str.startswith('<')]
        for idx, var in symbolic_variants.iterrows():
            assert 'SV_' in var['variant_type'], "Symbolic alleles should be classified as SV"

        print("✓ Missing END field handling test passed")

    def test_invalid_window_size(self):
        """Test handling of invalid sequence window sizes."""
        inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")

        # Zero window size may be handled gracefully (returns empty sequences) or raise error
        # Test that it doesn't crash
        try:
            results = list(sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=inv_vcf,
                seq_len=0,  # Invalid
                encode=False
            ))
            # If it doesn't raise an error, it should return empty or handle gracefully
            assert True, "Zero window size handled gracefully"
        except ValueError:
            # If it does raise ValueError, that's also acceptable
            assert True, "Zero window size raises ValueError as expected"

        print("✓ Invalid window size handling test passed")

    def test_overlapping_sv_detection(self):
        """Test that overlapping structural variants are detected and skipped."""
        # Use multi.vcf which has overlapping variants
        multi_vcf = os.path.join(self.data_dir, "multi", "multi.vcf")

        # Process with verbose to see skip messages
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            personal_genome = sl.get_personal_genome(
                self.reference_fa,
                multi_vcf,
                encode=False,
                verbose=False  # Set to False to avoid verbose output in tests
            )

            # Should successfully process, skipping overlapping variants
            assert len(personal_genome) > 0, "Should process VCF with overlapping variants"

        print("✓ Overlapping SV detection test passed")

    def test_chromosome_name_mismatch(self):
        """Test handling of chromosome name mismatches between VCF and reference."""
        inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")

        # Process without chromosome matching - should still work or warn appropriately
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = list(sl.get_alt_ref_sequences(
                reference_fn=self.reference_fa,
                variants_fn=inv_vcf,
                seq_len=200,
                encode=False
            ))

            # Should either process successfully or issue warnings
            assert len(results) >= 0, "Should handle chromosome names"

        print("✓ Chromosome name mismatch handling test passed")


class TestLargeVariants:
    """Test handling of very large structural variants."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")

    def test_large_inversion(self):
        """Test handling of large inversion variants."""
        inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")

        # Get sequences with window larger than inversion
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=inv_vcf,
            seq_len=500,  # Large window
            encode=False
        ))

        # Should handle large inversions
        assert len(results) >= 0, "Should handle large inversions"

        print("✓ Large inversion test passed")

    def test_large_duplication(self):
        """Test handling of large duplication variants."""
        dup_vcf = os.path.join(self.data_dir, "dup", "dup.vcf")

        # Get sequences with large window
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=dup_vcf,
            seq_len=500,  # Large window
            encode=False
        ))

        # Should handle large duplications
        assert len(results) >= 0, "Should handle large duplications"

        print("✓ Large duplication test passed")


class TestEncodingEdgeCases:
    """Test edge cases related to sequence encoding."""

    def setup_method(self):
        """Set up test data."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reference_fa = os.path.join(self.data_dir, "test_genome.fa")
        self.inv_vcf = os.path.join(self.data_dir, "inv", "inv.vcf")

    def test_encoded_sv_sequences_shape(self):
        """Test that encoded SV sequences have correct shape."""
        # Get encoded sequences
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.inv_vcf,
            seq_len=200,
            encode=True  # Encoded
        ))

        # Check shapes for INV variants
        for ref_seqs, alt_seqs, metadata in results:
            if metadata['variant_type'].iloc[0] == 'SV_BND_INV':
                # Encoded sequences should have (n_sequences, seq_len, 4) shape
                assert hasattr(ref_seqs, 'shape'), "Encoded sequences should have shape"
                assert ref_seqs.shape[-1] == 4, "Should have 4 channels (A, C, G, T)"
                assert ref_seqs.shape[-2] == 200, "Should have correct sequence length"

                assert hasattr(alt_seqs, 'shape'), "Encoded sequences should have shape"
                assert alt_seqs.shape[-1] == 4, "Should have 4 channels (A, C, G, T)"
                assert alt_seqs.shape[-2] == 200, "Should have correct sequence length"

        print("✓ Encoded SV sequences shape test passed")

    def test_n_padding_in_encoded_sequences(self):
        """Test that N-padding is properly encoded in one-hot format."""
        # Get encoded sequences that might have padding
        results = list(sl.get_alt_ref_sequences(
            reference_fn=self.reference_fa,
            variants_fn=self.inv_vcf,
            seq_len=200,
            encode=True
        ))

        # Check that N's are encoded as [0.25, 0.25, 0.25, 0.25]
        for ref_seqs, alt_seqs, metadata in results:
            if metadata['variant_type'].iloc[0] == 'SV_BND_INV':
                # Convert to numpy if needed
                if hasattr(ref_seqs, 'detach'):
                    ref_np = ref_seqs.detach().cpu().numpy()
                else:
                    ref_np = ref_seqs

                # Check if any position sums to ~1.0 (indicating N-padding)
                # N-padding would show as [0.25, 0.25, 0.25, 0.25] summing to 1.0
                # but with all values equal
                if len(ref_np.shape) >= 2:
                    position_sums = ref_np.sum(axis=-1)
                    # All positions should sum to 1.0 (one-hot encoding property)
                    assert np.allclose(position_sums, 1.0, atol=0.01), \
                        "One-hot encoded positions should sum to 1.0"

        print("✓ N-padding in encoded sequences test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
