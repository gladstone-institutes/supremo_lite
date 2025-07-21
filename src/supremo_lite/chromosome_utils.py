"""
Chromosome name matching utilities for supremo_lite.

This module provides functions for handling mismatches in chromosome naming
between FASTA references and VCF files using intelligent heuristics.
"""

import re
import warnings
from typing import Dict, Set, Optional, List, Tuple


def normalize_chromosome_name(chrom_name: str) -> str:
    """
    Normalize chromosome name to a standard format.

    Args:
        chrom_name: Raw chromosome name from VCF or FASTA

    Returns:
        Normalized chromosome name (without 'chr' prefix, uppercase)

    Examples:
        'chr1' -> '1'
        'CHR1' -> '1'
        'chrX' -> 'X'
        'chrMT' -> 'MT'
        'M' -> 'MT'  # Mitochondrial normalization
    """
    # Convert to string and strip whitespace
    normalized = str(chrom_name).strip()

    # Remove 'chr' prefix (case insensitive)
    normalized = re.sub(r"^chr", "", normalized, flags=re.IGNORECASE)

    # Handle mitochondrial chromosome variants
    if normalized.upper() in ["M", "MITO", "MITOCHONDRION"]:
        normalized = "MT"

    # Convert to uppercase for consistency
    normalized = normalized.upper()

    return normalized


def create_chromosome_mapping(
    reference_chroms: Set[str], vcf_chroms: Set[str]
) -> Dict[str, str]:
    """
    Create a mapping from VCF chromosome names to reference chromosome names.

    This function uses heuristics to match chromosome names between VCF and FASTA:
    1. Exact match (case sensitive)
    2. Exact match (case insensitive)
    3. Normalized match (with/without 'chr' prefix)
    4. Special cases for mitochondrial chromosomes

    Args:
        reference_chroms: Set of chromosome names from reference FASTA
        vcf_chroms: Set of chromosome names from VCF file

    Returns:
        Tuple of (mapping dict, unmatched set)

    Example:
        reference_chroms = {'1', '2', 'X', 'Y', 'MT'}
        vcf_chroms = {'chr1', 'chr2', 'chrX', 'chrY', 'chrM'}
        Returns: {'chr1': '1', 'chr2': '2', 'chrX': 'X', 'chrY': 'Y', 'chrM': 'MT'}
    """
    mapping = {}
    unmatched_vcf = set()

    # Try to match each VCF chromosome
    for vcf_chrom in vcf_chroms:
        matched_ref = None

        # 1. Try exact match (case sensitive)
        if vcf_chrom in reference_chroms:
            matched_ref = vcf_chrom

        # 2. Try exact match (case insensitive)
        if matched_ref is None:
            for ref_chrom in reference_chroms:
                if vcf_chrom.lower() == ref_chrom.lower():
                    matched_ref = ref_chrom
                    break

        # 3. Try removing/adding chr prefix
        if matched_ref is None:
            # If VCF has 'chr' prefix, try without it
            if vcf_chrom.lower().startswith("chr"):
                no_chr = vcf_chrom[3:]
                if no_chr in reference_chroms:
                    matched_ref = no_chr
                else:
                    # Try case insensitive match without chr
                    for ref_chrom in reference_chroms:
                        if no_chr.lower() == ref_chrom.lower():
                            matched_ref = ref_chrom
                            break

            # If VCF doesn't have 'chr' prefix, try with it
            else:
                with_chr = f"chr{vcf_chrom}"
                if with_chr in reference_chroms:
                    matched_ref = with_chr
                else:
                    # Try case insensitive match with chr
                    for ref_chrom in reference_chroms:
                        if with_chr.lower() == ref_chrom.lower():
                            matched_ref = ref_chrom
                            break

        # 4. Try normalized matching (handles mitochondrial variants)
        if matched_ref is None:
            vcf_normalized = normalize_chromosome_name(vcf_chrom)
            for ref_chrom in reference_chroms:
                ref_normalized = normalize_chromosome_name(ref_chrom)
                if vcf_normalized == ref_normalized:
                    matched_ref = ref_chrom
                    break

        # Record result
        if matched_ref is not None:
            mapping[vcf_chrom] = matched_ref
        else:
            unmatched_vcf.add(vcf_chrom)

    return mapping, unmatched_vcf


def apply_chromosome_mapping(variants_df, mapping: Dict[str, str]):
    """
    Apply chromosome name mapping to a variants DataFrame.

    Args:
        variants_df: Pandas DataFrame with 'chrom' column
        mapping: Dictionary mapping original to new chromosome names

    Returns:
        Modified DataFrame with updated chromosome names
    """
    variants_df = variants_df.copy()

    # Apply mapping to chromosome column
    variants_df["chrom"] = variants_df["chrom"].map(lambda x: mapping.get(x, x))

    return variants_df


def get_chromosome_match_report(
    reference_chroms: Set[str],
    vcf_chroms: Set[str],
    mapping: Dict[str, str],
    unmatched: Set[str],
) -> str:
    """
    Generate a human-readable report of chromosome matching results.

    Args:
        reference_chroms: Set of reference chromosome names
        vcf_chroms: Set of VCF chromosome names
        mapping: Successful mappings
        unmatched: Unmatched VCF chromosomes

    Returns:
        Formatted report string
    """
    report_lines = []

    report_lines.append("Chromosome Matching Report")
    report_lines.append("=" * 40)

    report_lines.append(
        f"Reference chromosomes ({len(reference_chroms)}): {sorted(reference_chroms)}"
    )
    report_lines.append(f"VCF chromosomes ({len(vcf_chroms)}): {sorted(vcf_chroms)}")
    report_lines.append("")

    if mapping:
        report_lines.append(f"Successfully matched ({len(mapping)}):")
        for vcf_chrom, ref_chrom in sorted(mapping.items()):
            if vcf_chrom != ref_chrom:
                report_lines.append(f"  '{vcf_chrom}' -> '{ref_chrom}'")
            else:
                report_lines.append(f"  '{vcf_chrom}' (exact match)")

    if unmatched:
        report_lines.append("")
        report_lines.append(f"Unmatched VCF chromosomes ({len(unmatched)}):")
        for chrom in sorted(unmatched):
            report_lines.append(f"  '{chrom}' (no suitable reference match found)")

    report_lines.append("")
    coverage = len(mapping) / len(vcf_chroms) * 100 if vcf_chroms else 100
    report_lines.append(
        f"Matching coverage: {coverage:.1f}% ({len(mapping)}/{len(vcf_chroms)})"
    )

    return "\n".join(report_lines)


def match_chromosomes_with_report(
    reference_chroms: Set[str], vcf_chroms: Set[str], verbose: bool = True
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Match chromosomes and optionally print a detailed report.

    Args:
        reference_chroms: Set of reference chromosome names
        vcf_chroms: Set of VCF chromosome names
        verbose: Whether to print matching report

    Returns:
        Tuple of (mapping dict, unmatched set)
    """
    mapping, unmatched = create_chromosome_mapping(reference_chroms, vcf_chroms)

    if verbose and (
        len(mapping) < len(vcf_chroms) or any(k != v for k, v in mapping.items())
    ):
        report = get_chromosome_match_report(
            reference_chroms, vcf_chroms, mapping, unmatched
        )
        print(report)

    if unmatched:
        warnings.warn(
            f"Could not match {len(unmatched)} VCF chromosomes to reference: "
            f"{sorted(unmatched)}. These variants will be skipped."
        )

    return mapping, unmatched
