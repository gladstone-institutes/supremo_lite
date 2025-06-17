"""
Personalized sequence generation for supremo_lite.

This module provides functions for creating personalized genomes by applying
variants to a reference genome and generating sequence windows around variants.
"""

import warnings
from .variant_utils import read_vcf
from pyfaidx import Fasta

def get_personal_genome(reference_fn, variants_fn):
    """
    Create a personalized genome by applying variants to a reference genome.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame

    Returns:
        A dictionary mapping chromosome names to personalized sequences
    """
    # Read variants if filename provided
    if isinstance(variants_fn, str):
        variants = read_vcf(variants_fn)
    else:
        variants = variants_fn

    # Sort variants by chromosome and position
    variants = variants.sort_values(["chrom", "pos"])

    # Load the reference genome once
    if isinstance(reference_fn, str):  # File path
        reference = Fasta(reference_fn)
    else:  # Dictionary-like object
        reference = reference_fn

    # Initialize personalized genome with all chromosomes from reference
    personal_genome = {}
    
    # First, add all chromosomes from reference (unchanged initially)
    for chrom in reference.keys():  # Use .keys() to get chromosome names
        # Get the sequence for this chromosome
        chrom_seq = reference[chrom]
        
        # Convert to string - pyfaidx FastaRecord objects convert automatically
        personal_genome[chrom] = str(chrom_seq)

    # Group variants by chromosome and apply them
    if not variants.empty:
        for chrom, chrom_vars in variants.groupby("chrom"):
            # Skip chromosomes not in reference
            if chrom not in personal_genome:
                continue
                
            # Get reference sequence for this chromosome
            ref_seq = personal_genome[chrom]  # We already have it from above

            # Apply variants sequentially
            personal_seq = ref_seq
            offset = 0  # Track position shifts due to indels

            for _, var in chrom_vars.iterrows():
                pos = var["pos"] + offset - 1  # Convert to 0-based and apply offset
                ref = var["ref"]
                alt = var["alt"]

                # Skip if variant is outside sequence bounds
                if pos >= len(personal_seq):
                    continue

                # Verify reference allele matches
                if personal_seq[pos : pos + len(ref)] != ref:
                    warnings.warn(
                        f"Reference allele mismatch at {chrom}:{var['pos']}. Expected {ref}, found {personal_seq[pos:pos+len(ref)]}."
                    )
                    continue

                # Apply the variant
                personal_seq = personal_seq[:pos] + alt + personal_seq[pos + len(ref) :]

                # Update offset for indels
                offset += len(alt) - len(ref)

            # Update the chromosome sequence
            personal_genome[chrom] = personal_seq

    return personal_genome
def get_personal_sequences(reference_fn, variants_fn, seq_len):
    """
    Create sequence windows centered on each variant position.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        seq_len: Length of the sequence window

    Returns:
        A list of tuples containing (chrom, start, end, sequence)
    """
    # Read variants if filename provided
    if isinstance(variants_fn, str):
        variants = read_vcf(variants_fn)
    else:
        variants = variants_fn

    sequences = []

    # For each variant, extract a sequence window
    for _, var in variants.iterrows():
        chrom = var["chrom"]
        pos = var["pos"]  # 1-based position

        # Calculate window boundaries
        window_start = max(0, pos - seq_len // 2)
        window_end = window_start + seq_len

        # Get reference sequence for this chromosome
        if hasattr(reference_fn, "__getitem__"):  # Dictionary-like
            try:
                ref_seq = reference_fn[chrom][window_start:window_end]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
            except IndexError:
                # Handle case where window extends beyond chromosome
                ref_seq = reference_fn[chrom][window_start:]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
                ref_seq += "N" * (seq_len - len(ref_seq))  # Pad with Ns
        else:  # Assume it's a file path
            # This is a simplified implementation
            ref_seq = "N" * seq_len  # Placeholder
            warnings.warn(
                "File-based sequence extraction not implemented yet. Returning N's."
            )

        sequences.append((chrom, window_start, window_end, ref_seq))

    return sequences


def get_pam_disrupting_personal_sequences(
    reference_fn, variants_fn, seq_len, max_pam_distance, pam_sequence="NGG"
):
    """
    Generate sequences for variants that disrupt PAM sites.

    Args:
        reference_fn: Path to reference genome file or dictionary-like object
        variants_fn: Path to variants file or DataFrame
        seq_len: Length of sequence windows
        max_pam_distance: Maximum distance from variant to PAM site
        pam_sequence: PAM sequence pattern (default: 'NGG' for SpCas9)

    Returns:
        A dictionary of sequences with PAM sites, and variants that disrupt them
    """
    # Read variants if filename provided
    if isinstance(variants_fn, str):
        variants = read_vcf(variants_fn)
    else:
        variants = variants_fn

    # Filter variants that are near PAM sites
    pam_disrupting_variants = []
    pam_intact_sequences = []
    pam_disrupted_sequences = []

    # For each variant, check if it disrupts a PAM site
    for _, var in variants.iterrows():
        chrom = var["chrom"]
        pos = var["pos"]  # 1-based position
        ref = var["ref"]
        alt = var["alt"]

        # Calculate window boundaries
        window_start = max(0, pos - seq_len // 2)
        window_end = window_start + seq_len

        # Get reference sequence for this region
        if hasattr(reference_fn, "__getitem__"):  # Dictionary-like
            try:
                ref_seq = reference_fn[chrom][window_start:window_end]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
            except IndexError:
                # Handle case where window extends beyond chromosome
                ref_seq = reference_fn[chrom][window_start:]
                if hasattr(ref_seq, "seq"):  # Handle pyfaidx-like objects
                    ref_seq = ref_seq.seq
                ref_seq += "N" * (seq_len - len(ref_seq))  # Pad with Ns
        else:  # Assume it's a file path
            # This is a simplified implementation
            ref_seq = "N" * seq_len  # Placeholder
            warnings.warn(
                "File-based sequence extraction not implemented yet. Returning N's."
            )

        # Find PAM sites in the reference sequence
        pam_sites = []
        for i in range(len(ref_seq) - len(pam_sequence) + 1):
            potential_pam = ref_seq[i : i + len(pam_sequence)]
            # Check if the potential PAM matches the pattern
            if all(
                a == b or b == "N"
                for a, b in zip(potential_pam.upper(), pam_sequence.upper())
            ):
                pam_sites.append(i)

        # Filter PAM sites that are within max_pam_distance of the variant
        variant_pos_in_window = pos - window_start
        nearby_pam_sites = [
            p for p in pam_sites if abs(p - variant_pos_in_window) <= max_pam_distance
        ]

        if nearby_pam_sites:
            pam_disrupting_variants.append(var)

            # Create sequence with the variant but PAM intact
            variant_seq = (
                ref_seq[: variant_pos_in_window - 1]
                + alt
                + ref_seq[variant_pos_in_window - 1 + len(ref) :]
            )
            pam_intact_sequences.append((chrom, window_start, window_end, variant_seq))

            # Create sequences with PAM disrupted
            for pam_site in nearby_pam_sites:
                # Make a version where the PAM is disrupted
                disrupted_seq = (
                    variant_seq[:pam_site] + "NNN" + variant_seq[pam_site + 3 :]
                )  # Disrupt with NNN
                pam_disrupted_sequences.append(
                    (chrom, window_start, window_end, disrupted_seq)
                )

    return {
        "variants": pam_disrupting_variants,
        "pam_intact": pam_intact_sequences,
        "pam_disrupted": pam_disrupted_sequences,
    }
