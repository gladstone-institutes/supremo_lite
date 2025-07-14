"""
Variant reading and handling utilities for supremo_lite.

This module provides functions for reading variants from VCF files
and other related operations.
"""
import io
import pandas as pd

def read_vcf(path):
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