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