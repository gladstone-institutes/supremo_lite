"""
Variant reading and handling utilities for supremo_lite.

This module provides functions for reading variants from VCF files
and other related operations.
"""

import io
import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, Optional


def read_vcf(path, include_info=True, classify_variants=True):
    """
    Read VCF file into pandas DataFrame with enhanced variant classification.

    Args:
        path: Path to VCF file
        include_info: Whether to include INFO field (default: True)
        classify_variants: Whether to classify variant types (default: True)

    Returns:
        DataFrame with columns: chrom, pos1, id, ref, alt, [info], [variant_type]
        
    Notes:
        - INFO field parsing enables structural variant classification
        - variant_type column uses VCF 4.2 compliant classification
        - Compatible with existing code expecting basic 5-column format
    """
    with open(path, "r") as f:
        lines = [l for l in f if not l.startswith("##")]

    # Determine columns to read based on parameters
    if include_info:
        usecols = [0, 1, 2, 3, 4, 7]  # Include INFO field
        base_columns = ["chrom", "pos1", "id", "ref", "alt", "info"]
    else:
        usecols = [0, 1, 2, 3, 4]  # Original columns only
        base_columns = ["chrom", "pos1", "id", "ref", "alt"]

    df = pd.read_csv(
        io.StringIO("".join(lines)),
        sep="\t",
        usecols=usecols,
    )

    # Set column names
    df.columns = base_columns

    # Validate that pos1 column is numeric
    if not pd.api.types.is_numeric_dtype(df["pos1"]):
        raise ValueError(
            f"Position column (second column) must be numeric, got {df['pos1'].dtype}"
        )
    
    # Filter out multiallelic variants (ALT alleles containing commas)
    df = _filter_multiallelic_variants(df)

    # Add variant classification if requested
    if classify_variants:
        df['variant_type'] = df.apply(
            lambda row: classify_variant_type(
                row['ref'], 
                row['alt'], 
                parse_vcf_info(row.get('info', '')) if include_info else None
            ), 
            axis=1
        )

    return df


def read_vcf_chunked(path, n_chunks=1, include_info=True, classify_variants=True):
    """
    Read VCF file in chunks using generator with enhanced variant classification.

    Args:
        path: Path to VCF file
        n_chunks: Number of chunks to split variants into (default: 1)
        include_info: Whether to include INFO field (default: True)
        classify_variants: Whether to classify variant types (default: True)

    Yields:
        DataFrame chunks with columns: chrom, pos1, id, ref, alt, [info], [variant_type]
    """
    with open(path, "r") as f:
        # Skip header lines
        lines = [l for l in f if not l.startswith("##")]

    # Determine columns to read based on parameters
    if include_info:
        usecols = [0, 1, 2, 3, 4, 7]  # Include INFO field
        base_columns = ["chrom", "pos1", "id", "ref", "alt", "info"]
    else:
        usecols = [0, 1, 2, 3, 4]  # Original columns only
        base_columns = ["chrom", "pos1", "id", "ref", "alt"]

    # Read full dataframe first
    full_df = pd.read_csv(
        io.StringIO("".join(lines)), sep="\t", usecols=usecols
    )

    # Handle empty DataFrame
    if len(full_df) == 0:
        return

    # Set column names
    full_df.columns = base_columns

    # Validate that pos1 column is numeric
    if not pd.api.types.is_numeric_dtype(full_df["pos1"]):
        raise ValueError(
            f"Position column (second column) must be numeric, got {full_df['pos1'].dtype}"
        )
    
    # Filter out multiallelic variants (ALT alleles containing commas)
    full_df = _filter_multiallelic_variants(full_df)

    # Add variant classification if requested
    if classify_variants:
        full_df['variant_type'] = full_df.apply(
            lambda row: classify_variant_type(
                row['ref'], 
                row['alt'], 
                parse_vcf_info(row.get('info', '')) if include_info else None
            ), 
            axis=1
        )

    # Split into chunks using numpy array_split
    # Use numpy array_split to create n_chunks approximately equal chunks
    indices = np.array_split(np.arange(len(full_df)), n_chunks)

    for chunk_indices in indices:
        if len(chunk_indices) > 0:
            yield full_df.iloc[chunk_indices].reset_index(drop=True)


def get_vcf_chromosomes(path):
    """
    Get list of chromosomes in VCF file without loading all variants.

    Args:
        path: Path to VCF file

    Returns:
        Set of chromosome names found in the VCF file
    """
    chromosomes = set()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue
            # Parse first column (chromosome)
            chrom = line.split("\t")[0]
            chromosomes.add(chrom)
    return chromosomes


def read_vcf_chromosome(path, target_chromosome, include_info=True, classify_variants=True):
    """
    Read VCF file for a specific chromosome only with enhanced variant classification.

    Args:
        path: Path to VCF file
        target_chromosome: Chromosome name to filter for
        include_info: Whether to include INFO field (default: True)
        classify_variants: Whether to classify variant types (default: True)

    Returns:
        DataFrame with variants only from specified chromosome 
        (columns: chrom, pos1, id, ref, alt, [info], [variant_type])
    """
    chromosome_lines = []
    header_line = None

    with open(path, "r") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header_line = line
                continue

            # Check if this line is for our target chromosome
            chrom = line.split("\t")[0]
            if chrom == target_chromosome:
                chromosome_lines.append(line)

    # Determine columns to read based on parameters
    if include_info:
        usecols = [0, 1, 2, 3, 4, 7]  # Include INFO field
        base_columns = ["chrom", "pos1", "id", "ref", "alt", "info"]
    else:
        usecols = [0, 1, 2, 3, 4]  # Original columns only
        base_columns = ["chrom", "pos1", "id", "ref", "alt"]

    if not chromosome_lines:
        # Return empty DataFrame with correct columns if no variants found
        empty_columns = base_columns.copy()
        if classify_variants:
            empty_columns.append("variant_type")
        return pd.DataFrame(columns=empty_columns)

    # Combine header and chromosome-specific lines
    vcf_data = header_line + "".join(chromosome_lines)

    # Parse into DataFrame
    df = pd.read_csv(io.StringIO(vcf_data), sep="\t", usecols=usecols)

    # Set column names
    df.columns = base_columns

    # Validate that pos1 column is numeric
    if len(df) > 0 and not pd.api.types.is_numeric_dtype(df["pos1"]):
        raise ValueError(
            f"Position column (second column) must be numeric, got {df['pos1'].dtype}"
        )
    
    # Filter out multiallelic variants (ALT alleles containing commas)
    if len(df) > 0:
        df = _filter_multiallelic_variants(df)

    # Add variant classification if requested
    if classify_variants and len(df) > 0:
        df['variant_type'] = df.apply(
            lambda row: classify_variant_type(
                row['ref'], 
                row['alt'], 
                parse_vcf_info(row.get('info', '')) if include_info else None
            ), 
            axis=1
        )

    return df


def read_vcf_chromosomes_chunked(path, target_chromosomes, n_chunks=1, include_info=True, classify_variants=True):
    """
    Read VCF file for specific chromosomes in chunks with enhanced variant classification.

    Args:
        path: Path to VCF file
        target_chromosomes: List/set of chromosome names to include
        n_chunks: Number of chunks per chromosome (default: 1)
        include_info: Whether to include INFO field (default: True)
        classify_variants: Whether to classify variant types (default: True)

    Yields:
        Tuples of (chromosome, variants_dataframe) for each chunk
        DataFrame columns: chrom, pos1, id, ref, alt, [info], [variant_type]
    """
    target_chromosomes = set(target_chromosomes)

    for chrom in target_chromosomes:
        chrom_variants = read_vcf_chromosome(path, chrom, include_info, classify_variants)

        if len(chrom_variants) == 0:
            continue

        if n_chunks == 1:
            # Single chunk - yield all variants for this chromosome
            yield chrom, chrom_variants
        else:
            # Multiple chunks - split chromosome variants into n_chunks
            indices = np.array_split(np.arange(len(chrom_variants)), n_chunks)

            for i, chunk_indices in enumerate(indices):
                if len(chunk_indices) > 0:
                    chunk_df = chrom_variants.iloc[chunk_indices].reset_index(drop=True)
                    yield f"{chrom}_chunk_{i+1}", chunk_df


def parse_vcf_info(info_string: str) -> Dict:
    """
    Parse VCF INFO field to extract variant information according to VCF 4.2 specification.
    
    Args:
        info_string: VCF INFO field string (e.g., "SVTYPE=INV;END=1234;SVLEN=100")
        
    Returns:
        dict: Parsed INFO field values with appropriate type conversion
        
    VCF 4.2 INFO field specification:
        - Key=Value pairs separated by semicolons
        - Boolean flags have no value (key presence = True)  
        - Numeric values auto-converted to int/float
        - Reserved keys: AA, AC, AF, AN, BQ, CIGAR, DB, DP, END, H2, H3, MQ, MQ0, NS, SB, etc.
        
    Examples:
        parse_vcf_info("SVTYPE=INV;END=1234;SVLEN=100") 
        → {'SVTYPE': 'INV', 'END': 1234, 'SVLEN': 100}
        
        parse_vcf_info("DB;H2;AF=0.5") 
        → {'DB': True, 'H2': True, 'AF': 0.5}
    """
    info_dict = {}
    if not info_string or info_string == '.':
        return info_dict
        
    for field in info_string.split(';'):
        field = field.strip()
        if not field:
            continue
            
        if '=' in field:
            key, value = field.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle comma-separated lists (like AC=1,2,3)
            if ',' in value:
                value_list = [v.strip() for v in value.split(',')]
                # Try to convert list elements to numbers
                converted_list = []
                for v in value_list:
                    try:
                        if '.' in v:
                            converted_list.append(float(v))
                        else:
                            converted_list.append(int(v))
                    except ValueError:
                        converted_list.append(v)
                info_dict[key] = converted_list
            else:
                # Single value - try numeric conversion
                try:
                    if '.' in value:
                        info_dict[key] = float(value)
                    else:
                        info_dict[key] = int(value)
                except ValueError:
                    info_dict[key] = value
        else:
            # Boolean flag (presence = True)
            info_dict[field.strip()] = True
            
    return info_dict


def classify_variant_type(ref_allele: str, alt_allele: str, info_dict: Optional[Dict] = None) -> str:
    """
    Classify variant type according to VCF 4.2 specification using comprehensive heuristics.
    
    This function implements the complete VCF 4.2 variant classification rules with proper
    handling of structural variants, standard sequence variants, and edge cases.
    
    Args:
        ref_allele: Reference allele sequence (REF field)
        alt_allele: Alternate allele sequence (ALT field) 
        info_dict: Parsed INFO field dictionary (optional, for structural variants)
        
    Returns:
        str: Variant type classification
        
    VCF 4.2 Variant Types (in classification priority order):
        - 'complex': Complex/multiallelic variants (ALT contains comma)
        - 'missing': Missing/upstream deletion allele (ALT = '*')
        - 'SV_INV': Inversion structural variant
        - 'SV_DUP': Duplication structural variant  
        - 'SV_DEL': Deletion structural variant
        - 'SV_INS': Insertion structural variant
        - 'SV_CNV': Copy number variant
        - 'SV_BND': Breakend/translocation
        - 'SNV': Single nucleotide variant
        - 'MNV': Milti-nucleotide variant (alt len = ref len but no prefix)
        - 'INS': Sequence insertion
        - 'DEL': Sequence deletion
        - 'complex': Complex/multi-nucleotide variant (same length substitution)
        - 'unknown': Unclassifiable variant
    Note: MNV is not part of the official VCF 4.2 spec, they are treated the same as SNVs
    for all functions in supremo_lite
    Examples:
        # Multiallelic variants
        classify_variant_type('A', 'G,T') → 'complex'
        classify_variant_type('T', 'TGGG,C') → 'complex'
        
        # Standard variants
        classify_variant_type('A', 'G') → 'SNV'
        classify_variant_type('AGG', 'TCG') → 'MNV'
        classify_variant_type('T', 'TGGG') → 'INS'  
        classify_variant_type('CGAGAA', 'C') → 'DEL'
        
        # Structural variants
        classify_variant_type('N', '<INV>') → 'SV_INV'
        classify_variant_type('G', 'G]17:198982]') → 'SV_BND'
        
        # Special cases
        classify_variant_type('T', '*') → 'missing'
        
    VCF 4.2 Reference: https://samtools.github.io/hts-specs/VCFv4.2.pdf
    """
    if not ref_allele or not alt_allele:
        return 'unknown'
    
    # Normalize alleles (VCF allows mixed case)
    ref = ref_allele.upper().strip()
    alt = alt_allele.upper().strip()
    
    # PRIORITY 0: Multiallelic variants (comma-separated ALT alleles)
    # Multiple alternative alleles in single ALT field indicate complex variant
    if ',' in alt:
        return 'complex'
    
    # PRIORITY 1: Handle missing/upstream deletion alleles
    # The '*' allele indicates missing due to upstream deletion (VCF 4.2 spec)
    if alt == '*':
        return 'missing'
    
    # PRIORITY 2: Structural variants with symbolic alleles
    # Format: <ID> where ID indicates structural variant type
    if alt.startswith('<') and alt.endswith('>'):
        sv_type = alt[1:-1].upper()  # Extract type from <INV>, <DUP>, etc.
        
        # Map symbolic alleles to standard classifications
        if sv_type in ['INV']:
            return 'SV_INV'
        elif sv_type in ['DUP', 'DUP:TANDEM']:
            return 'SV_DUP'
        elif sv_type in ['DEL', 'DEL:ME', 'DEL:ME:ALU', 'DEL:ME:L1']:
            return 'SV_DEL'
        elif sv_type in ['INS', 'INS:ME', 'INS:ME:ALU', 'INS:ME:L1']:
            return 'SV_INS'
        elif sv_type in ['CNV']:
            return 'SV_CNV'
        elif sv_type in ['BND', 'TRA']:
            return 'SV_BND'
        else:
            # Unknown symbolic allele - return unknown rather than creating arbitrary SV types
            return 'unknown'


    # PRIORITY 3: Breakend notation (complex rearrangements)
    # Format examples: A[chr2:1000[, ]chr1:100]T, etc.
    breakend_pattern = r'[\[\]]'
    if re.search(breakend_pattern, alt):
        return 'SV_BND'
    
    # PRIORITY 4: Check SVTYPE in INFO field for additional SV classification
    if info_dict and 'SVTYPE' in info_dict:
        svtype = str(info_dict['SVTYPE']).upper()
        if svtype in ['INV']:
            return 'SV_INV'
        elif svtype in ['DUP', 'TANDEM']:
            return 'SV_DUP'
        elif svtype in ['DEL']:
            return 'SV_DEL'
        elif svtype in ['INS']:
            return 'SV_INS'
        elif svtype in ['CNV']:
            return 'SV_CNV'
        elif svtype in ['BND', 'TRA', 'TRANSLOCATION']:
            return 'SV_BND'
    
    # PRIORITY 5: Standard sequence variants based on length comparison
    ref_len = len(ref)
    alt_len = len(alt)
    
    if ref_len == 1 and alt_len == 1:
        # Single base substitution
        if ref != alt:
            return 'SNV'
        else:
            # Identical alleles - should not occur in valid VCF
            return 'unknown'
            
    elif ref_len == 1 and alt_len > 1:
        # Potential insertion: check if REF is prefix of ALT
        if alt.startswith(ref):
            return 'INS'
        else:
            # REF not a prefix - complex variant
            return 'complex'
            
    elif ref_len > 1 and alt_len == 1:
        # Potential deletion: check if ALT is prefix of REF  
        if ref.startswith(alt):
            return 'DEL'
        else:
            # ALT not a prefix - complex variant
            return 'complex'
            
    elif ref_len > 1 and alt_len > 1:
        # Multi-base variant - determine if complex substitution or indel
        # Check for shared prefix/suffix to identify indel vs substitution
        
        # Find longest common prefix
        prefix_len = 0
        min_len = min(ref_len, alt_len)
        while prefix_len < min_len and ref[prefix_len] == alt[prefix_len]:
            prefix_len += 1
        
        # Find longest common suffix
        suffix_len = 0
        while (suffix_len < min_len - prefix_len and 
               ref[ref_len - 1 - suffix_len] == alt[alt_len - 1 - suffix_len]):
            suffix_len += 1
        
        # Analyze the variant structure
        if prefix_len + suffix_len >= min_len:
            # Significant overlap - likely indel
            if ref_len > alt_len:
                return 'DEL'
            elif alt_len > ref_len:
                return 'INS'
            else:
                # Same length with shared prefix/suffix - substitution
                return 'complex'
        else:
            # Limited overlap - substitution
            return 'MNV'
    
    else:
        # Empty allele - should not occur in valid VCF
        return 'unknown'


def _filter_multiallelic_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out variants with multiallelic ALT fields (containing commas).
    
    Args:
        df: DataFrame with variant data including 'alt' column
        
    Returns:
        DataFrame with multiallelic variants removed
        
    Notes:
        Issues a warning when multiallelic variants are found and removed.
        Multiallelic variants have ALT fields like "G,T" indicating multiple
        alternative alleles at the same position.
    """
    if 'alt' not in df.columns or len(df) == 0:
        return df
        
    # Identify multiallelic variants (ALT field contains comma)
    multiallelic_mask = df['alt'].str.contains(',', na=False)
    n_multiallelic = multiallelic_mask.sum()
    
    if n_multiallelic > 0:
        warnings.warn(
            f"Found {n_multiallelic} multiallelic variants with comma-separated ALT alleles. "
            f"These variants have been removed from the dataset. "
            f"Consider preprocessing your VCF file to split multiallelic sites if needed.",
            UserWarning
        )
        
        # Filter out multiallelic variants
        df = df[~multiallelic_mask].reset_index(drop=True)
    
    return df

