"""
Variant reading and handling utilities for supremo_lite.

This module provides functions for reading variants from VCF files
and other related operations.
"""
import io
import pandas as pd
import numpy as np

def read_vcf(path):
    """
    Read VCF file into pandas DataFrame.
    
    Args:
        path: Path to VCF file
        
    Returns:
        DataFrame with columns: chrom, pos, id, ref, alt
    """
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        sep='\t',
        usecols=[0, 1, 2, 3, 4]  # Select only first 5 columns
    ).rename(columns={
        '#CHROM': 'chrom',
        'POS': 'pos',
        'ID': 'id',
        'REF': 'ref',
        'ALT': 'alt'
    })

def read_vcf_chunked(path, chunk_size=1000):
    """
    Read VCF file in chunks using generator.
    
    Args:
        path: Path to VCF file
        chunk_size: Number of variants per chunk (default: 1000)
        
    Yields:
        DataFrame chunks with columns: chrom, pos, id, ref, alt
    """
    with open(path, 'r') as f:
        # Skip header lines
        lines = [l for l in f if not l.startswith('##')]
    
    # Read full dataframe first
    full_df = pd.read_csv(
        io.StringIO(''.join(lines)),
        sep='\t',
        usecols=[0, 1, 2, 3, 4]
    ).rename(columns={
        '#CHROM': 'chrom',
        'POS': 'pos',
        'ID': 'id',
        'REF': 'ref',
        'ALT': 'alt'
    })
    
    # Split into chunks using numpy array_split
    n_chunks = max(1, (len(full_df) + chunk_size - 1) // chunk_size)
    
    # Use numpy array_split to create approximately equal chunks
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
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                continue
            # Parse first column (chromosome)
            chrom = line.split('\t')[0]
            chromosomes.add(chrom)
    return chromosomes


def read_vcf_chromosome(path, target_chromosome):
    """
    Read VCF file for a specific chromosome only.
    
    Args:
        path: Path to VCF file
        target_chromosome: Chromosome name to filter for
        
    Returns:
        DataFrame with variants only from specified chromosome
    """
    chromosome_lines = []
    header_line = None
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                header_line = line
                continue
            
            # Check if this line is for our target chromosome
            chrom = line.split('\t')[0]
            if chrom == target_chromosome:
                chromosome_lines.append(line)
    
    if not chromosome_lines:
        # Return empty DataFrame with correct columns if no variants found
        return pd.DataFrame(columns=['chrom', 'pos', 'id', 'ref', 'alt'])
    
    # Combine header and chromosome-specific lines
    vcf_data = header_line + ''.join(chromosome_lines)
    
    # Parse into DataFrame
    df = pd.read_csv(
        io.StringIO(vcf_data),
        sep='\t',
        usecols=[0, 1, 2, 3, 4]
    ).rename(columns={
        '#CHROM': 'chrom',
        'POS': 'pos', 
        'ID': 'id',
        'REF': 'ref',
        'ALT': 'alt'
    })
    
    return df


def read_vcf_chromosomes_chunked(path, target_chromosomes, chunk_size=50000):
    """
    Read VCF file for specific chromosomes in chunks.
    
    Args:
        path: Path to VCF file
        target_chromosomes: List/set of chromosome names to include
        chunk_size: Variants per chunk for large chromosomes
        
    Yields:
        Tuples of (chromosome, variants_dataframe) for each chunk
    """
    target_chromosomes = set(target_chromosomes)
    
    for chrom in target_chromosomes:
        chrom_variants = read_vcf_chromosome(path, chrom)
        
        if len(chrom_variants) == 0:
            continue
            
        if len(chrom_variants) <= chunk_size:
            # Small chromosome - yield all at once
            yield chrom, chrom_variants
        else:
            # Large chromosome - chunk within chromosome  
            n_chunks = max(1, (len(chrom_variants) + chunk_size - 1) // chunk_size)
            indices = np.array_split(np.arange(len(chrom_variants)), n_chunks)
            
            for i, chunk_indices in enumerate(indices):
                if len(chunk_indices) > 0:
                    chunk_df = chrom_variants.iloc[chunk_indices].reset_index(drop=True)
                    yield f"{chrom}_chunk_{i+1}", chunk_df