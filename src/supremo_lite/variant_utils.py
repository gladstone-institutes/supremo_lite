"""
Variant reading and handling utilities for supremo_lite.

This module provides functions for reading variants from VCF files 
and other related operations.
"""

import pandas as pd

def read_vcf(fn):
    """
    Read a VCF file into a pandas DataFrame.
    
    Args:
        fn: Path to the VCF file
    
    Returns:
        A DataFrame with columns: chrom, pos, id, ref, alt
    """
    return pd.read_table(
        fn, 
        comment='#', 
        usecols=list(range(5)), 
        names=['chrom', 'pos', 'id', 'ref', 'alt']
    )