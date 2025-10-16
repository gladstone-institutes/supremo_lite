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
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class BreakendVariant:
    """
    Represents a single breakend variant from a VCF file.

    This class stores all information needed to process a BND variant,
    including mate relationships and inserted sequences.
    """
    id: str                    # VCF ID field (e.g., "bnd_W")
    chrom: str                 # Chromosome name
    pos: int                   # 1-based position
    ref: str                   # Reference allele
    alt: str                   # Complete ALT field (e.g., "G]17:198982]")
    mate_id: str               # MATEID from INFO field
    mate_chrom: str            # Mate chromosome (parsed from ALT)
    mate_pos: int              # Mate position (parsed from ALT)
    orientation: str           # Breakend orientation (e.g., "ref_then_mate")
    inserted_seq: str          # Novel sequence at junction
    info: str                  # Complete INFO field
    variant_type: str = "SV_BND"  # Always BND for breakend variants

    def __post_init__(self):
        """Validate breakend data after initialization."""
        if not self.id:
            raise ValueError("Breakend ID cannot be empty")
        if self.pos <= 0:
            raise ValueError("Breakend position must be positive")
        if self.mate_pos <= 0:
            raise ValueError("Breakend mate position must be positive")


@dataclass
class BreakendPair:
    """
    Represents a pair of mated breakend variants that create a novel adjacency.

    This class coordinates the application of both breakends to create
    complex rearrangements like translocations, inversions, etc.
    """
    breakend1: BreakendVariant
    breakend2: BreakendVariant
    is_valid: bool = True
    validation_errors: List[str] = None
    validation_warnings: List[str] = None

    def __post_init__(self):
        """Validate that the two breakends form a consistent pair."""
        if self.validation_errors is None:
            self.validation_errors = []
        if self.validation_warnings is None:
            self.validation_warnings = []

        # Validate mate relationships
        if self.breakend1.mate_id != self.breakend2.id:
            self.validation_errors.append(
                f"Breakend {self.breakend1.id} MATEID {self.breakend1.mate_id} "
                f"does not match mate ID {self.breakend2.id}"
            )
            self.is_valid = False

        if self.breakend2.mate_id != self.breakend1.id:
            self.validation_errors.append(
                f"Breakend {self.breakend2.id} MATEID {self.breakend2.mate_id} "
                f"does not match mate ID {self.breakend1.id}"
            )
            self.is_valid = False

        # Validate coordinate consistency
        if self.breakend1.mate_chrom != self.breakend2.chrom:
            self.validation_errors.append(
                f"Breakend {self.breakend1.id} mate chromosome {self.breakend1.mate_chrom} "
                f"does not match actual chromosome {self.breakend2.chrom}"
            )
            self.is_valid = False

        if self.breakend1.mate_pos != self.breakend2.pos:
            self.validation_errors.append(
                f"Breakend {self.breakend1.id} mate position {self.breakend1.mate_pos} "
                f"does not match actual position {self.breakend2.pos}"
            )
            self.is_valid = False

    @property
    def rearrangement_type(self) -> str:
        """
        Determine the type of rearrangement represented by this breakend pair.

        Returns:
            str: Rearrangement type ('translocation', 'inversion', 'duplication', 'complex')
        """
        if not self.is_valid:
            return 'invalid'

        # Check if breakends are on same chromosome
        if self.breakend1.chrom == self.breakend2.chrom:
            # Same chromosome - could be inversion, duplication, or deletion
            pos1, pos2 = self.breakend1.pos, self.breakend2.pos
            orient1, orient2 = self.breakend1.orientation, self.breakend2.orientation

            # Simple heuristics for now - detailed implementation would need more logic
            if abs(pos1 - pos2) < 1000:  # Close positions might be duplication
                return 'duplication'
            elif orient1 != orient2:  # Different orientations suggest inversion
                return 'inversion'
            else:
                return 'complex'
        else:
            # Different chromosomes - translocation
            return 'translocation'

    def get_affected_regions(self) -> List[Tuple[str, int, int]]:
        """
        Get genomic regions affected by this breakend pair.

        Returns:
            List of tuples (chrom, start, end) for affected regions
        """
        regions = []

        # Add region around first breakend
        regions.append((
            self.breakend1.chrom,
            max(1, self.breakend1.pos - 1),  # Include position before breakend
            self.breakend1.pos + len(self.breakend1.ref)
        ))

        # Add region around second breakend
        regions.append((
            self.breakend2.chrom,
            max(1, self.breakend2.pos - 1),  # Include position before breakend
            self.breakend2.pos + len(self.breakend2.ref)
        ))

        return regions


@dataclass
class Breakend:
    """Enhanced breakend with classification information."""
    id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    mate_chrom: str
    mate_pos: int
    orientation: str
    inserted_seq: str
    classification: str  # 'paired', 'missing_mate', 'singleton_insertion'
    mate_breakend: Optional['Breakend'] = None

    @classmethod
    def from_breakend_variant(cls, variant: BreakendVariant, classification: str) -> 'Breakend':
        """Create from BreakendVariant."""
        return cls(
            id=variant.id,
            chrom=variant.chrom,
            pos=variant.pos,
            ref=variant.ref,
            alt=variant.alt,
            mate_chrom=variant.mate_chrom,
            mate_pos=variant.mate_pos,
            orientation=variant.orientation,
            inserted_seq=variant.inserted_seq,
            classification=classification,
            mate_breakend=None
        )


class BNDClassifier:
    """
    BND classifier that doesn't depend on MATEID fields.

    Classifies BNDs into categories:
    1. Paired breakends - have matching mates by coordinates
    2. Missing mates - reference coordinates not present in VCF (can be inferred)
    3. Insertions with mates - insertions where mate is present
    4. Insertions without mates - insertions where mate is missing (inferred)
    """

    def __init__(self):
        self.all_breakends = []
        self.coordinate_index = {}  # Map (chrom, pos) -> breakend

    def classify_all_breakends(self, vcf_path: str) -> Dict[str, List[Breakend]]:
        """
        Classify all BND variants from a VCF file.

        Returns:
            Dict with keys 'paired', 'missing_mate', 'singleton_insertion'
        """
        # TODO: This function needs to respect the verbose arguement

        # Load VCF with variant classification
        variants_df = read_vcf(vcf_path, include_info=True, classify_variants=True)
        bnd_variants = variants_df[variants_df['variant_type'].isin(['SV_BND', 'SV_BND_INS'])]

        print(f"Found {len(bnd_variants)} BND variants")

        # Parse all breakends and build coordinate index
        self._parse_and_index_breakends(bnd_variants)

        # Classify breakends
        classified = {
            'paired': [],
            'missing_mate': [],
            'insertion_with_mate': [],
            'insertion_missing_mate': []
        }

        processed_ids = set()

        for breakend in self.all_breakends:
            if breakend.id in processed_ids:
                continue

            # Try to find mate by coordinates
            mate_key = (breakend.mate_chrom, breakend.mate_pos)

            if mate_key in self.coordinate_index:
                # Found mate - this is a paired breakend
                mate_breakend = self.coordinate_index[mate_key]

                # Create enhanced breakends for the pair
                enhanced1 = Breakend.from_breakend_variant(breakend, 'paired')
                enhanced2 = Breakend.from_breakend_variant(mate_breakend, 'paired')

                enhanced1.mate_breakend = enhanced2
                enhanced2.mate_breakend = enhanced1

                classified['paired'].extend([enhanced1, enhanced2])
                processed_ids.add(breakend.id)
                processed_ids.add(mate_breakend.id)

            else:
                # No mate found - ALWAYS infer the missing mate and create a fusion
                # Create an inferred mate breakend
                inferred_mate = BreakendVariant(
                    id=f"{breakend.id}_inferred_mate",
                    chrom=breakend.mate_chrom,
                    pos=breakend.mate_pos,
                    ref="N",  # Unknown reference at mate position
                    alt="<INFERRED>",
                    mate_id=breakend.id,
                    mate_chrom=breakend.chrom,
                    mate_pos=breakend.pos,
                    orientation=self._infer_mate_orientation(breakend.orientation),
                    inserted_seq="",  # Novel sequence stays on the original breakend
                    info=f"INFERRED_FROM={breakend.id}",
                    variant_type='SV_BND'
                )

                # Create enhanced breakends for the inferred pair
                enhanced1 = Breakend.from_breakend_variant(breakend, 'paired')
                enhanced2 = Breakend.from_breakend_variant(inferred_mate, 'paired')

                enhanced1.mate_breakend = enhanced2
                enhanced2.mate_breakend = enhanced1

                classified['paired'].extend([enhanced1, enhanced2])

                if breakend.inserted_seq:
                    print(f"INFO: Inferred missing mate for {breakend.id} with novel sequence '{breakend.inserted_seq}' "
                          f"-> created fusion with inferred mate at {breakend.mate_chrom}:{breakend.mate_pos}")
                else:
                    print(f"INFO: Inferred missing mate for {breakend.id} "
                          f"-> created fusion with inferred mate at {breakend.mate_chrom}:{breakend.mate_pos}")

                processed_ids.add(breakend.id)

        # Apply semantic classification to detect DUP and INV patterns
        # First detect duplications
        dup_breakends = self._detect_duplication_pattern(classified['paired'])

        # Remove duplication breakends from paired list before inversion detection
        dup_ids = {b.id for b in dup_breakends}
        remaining_paired = [b for b in classified['paired'] if b.id not in dup_ids]

        # Then detect inversions from remaining breakends
        inv_breakends = self._detect_inversion_pattern(remaining_paired)

        # Remove reclassified breakends from 'paired' category and add to semantic categories
        reclassified_ids = {b.id for b in dup_breakends + inv_breakends}
        classified['paired'] = [b for b in classified['paired'] if b.id not in reclassified_ids]

        # Add semantic classifications
        classified['dup_breakends'] = dup_breakends
        classified['inv_breakends'] = inv_breakends

        # Print classification summary
        print(f"\nBND Classification Summary:")
        print(f"  Paired breakends (true translocations): {len(classified['paired'])}")
        print(f"  Duplication breakends (SV_BND_DUP): {len(dup_breakends)}")
        print(f"  Inversion breakends (SV_BND_INV): {len(inv_breakends)}")
        total_inferred = len([bnd for bnd in classified['paired'] + dup_breakends + inv_breakends if 'inferred' in bnd.id])
        print(f"  Inferred mates created: {total_inferred}")

        return classified

    def _detect_duplication_pattern(self, paired_breakends: List[Breakend]) -> List[Breakend]:
        """
        Detect duplication patterns from paired breakends.

        A duplication pattern consists of 2 breakends:
        - BND1: position A pointing to position B
        - BND2: position B pointing to position A
        - Same chromosome, A < B (tandem duplication)

        Returns:
            List of breakends reclassified as SV_BND_DUP
        """
        dup_breakends = []
        processed_ids = set()

        for breakend in paired_breakends:
            if breakend.id in processed_ids or not breakend.mate_breakend:
                continue

            mate = breakend.mate_breakend

            # Check if this forms a duplication pattern
            if (breakend.chrom == mate.chrom and  # Same chromosome
                breakend.chrom == breakend.mate_chrom and  # Mate points to same chromosome
                mate.chrom == mate.mate_chrom and  # Mate's mate points to same chromosome
                breakend.pos != mate.pos):  # Different positions

                # Check mutual pointing (A->B, B->A)
                points_to_mate = (breakend.mate_chrom == mate.chrom and
                                breakend.mate_pos == mate.pos)
                mate_points_back = (mate.mate_chrom == breakend.chrom and
                                  mate.mate_pos == breakend.pos)

                if points_to_mate and mate_points_back:
                    # Check orientations to determine if this is truly a duplication or inversion
                    # Duplication: orientations should be compatible with copy-paste behavior
                    # Inversion: orientations should indicate sequence reversal

                    orientation1 = breakend.orientation
                    orientation2 = mate.orientation

                    # Simple heuristic: if we have more than 2 breakends on same chromosome pointing to each other,
                    # it's likely an inversion pattern, not duplication (which typically involves 2 breakends)
                    # Count breakends on this chromosome
                    same_chrom_breakends = [b for b in paired_breakends if b.chrom == breakend.chrom and b.mate_chrom == breakend.chrom]

                    if len(same_chrom_breakends) > 2:
                        # Likely inversion pattern with multiple breakends - skip duplication classification
                        continue

                    # This is a duplication pattern - reclassify both breakends
                    dup_breakend1 = Breakend.from_breakend_variant(
                        BreakendVariant(
                            id=breakend.id,
                            chrom=breakend.chrom,
                            pos=breakend.pos,
                            ref=breakend.ref,
                            alt=breakend.alt,
                            mate_id=getattr(breakend, 'mate_id', ''),
                            mate_chrom=breakend.mate_chrom,
                            mate_pos=breakend.mate_pos,
                            orientation=breakend.orientation,
                            inserted_seq=breakend.inserted_seq,
                            info='',
                            variant_type='SV_BND_DUP'
                        ), 'SV_BND_DUP'
                    )
                    dup_breakend2 = Breakend.from_breakend_variant(
                        BreakendVariant(
                            id=mate.id,
                            chrom=mate.chrom,
                            pos=mate.pos,
                            ref=mate.ref,
                            alt=mate.alt,
                            mate_id=getattr(mate, 'mate_id', ''),
                            mate_chrom=mate.mate_chrom,
                            mate_pos=mate.mate_pos,
                            orientation=mate.orientation,
                            inserted_seq=mate.inserted_seq,
                            info='',
                            variant_type='SV_BND_DUP'
                        ), 'SV_BND_DUP'
                    )

                    # Maintain mate relationships
                    dup_breakend1.mate_breakend = dup_breakend2
                    dup_breakend2.mate_breakend = dup_breakend1

                    dup_breakends.extend([dup_breakend1, dup_breakend2])
                    processed_ids.add(breakend.id)
                    processed_ids.add(mate.id)

        return dup_breakends

    def _detect_inversion_pattern(self, paired_breakends: List[Breakend]) -> List[Breakend]:
        """
        Detect inversion patterns from paired breakends.

        An inversion pattern consists of 4 breakends forming 2 pairs:
        - Pair 1: Outer breakpoints (A, B) with inverted orientations
        - Pair 2: Inner breakpoints (C, D) with inverted orientations
        - Same chromosome, positions in order A < C < D < B

        Returns:
            List of breakends reclassified as SV_BND_INV
        """
        inv_breakends = []
        processed_ids = set()

        # Group breakends by chromosome for efficiency
        chrom_groups = {}
        for breakend in paired_breakends:
            if breakend.id in processed_ids:
                continue
            chrom = breakend.chrom
            if chrom not in chrom_groups:
                chrom_groups[chrom] = []
            chrom_groups[chrom].append(breakend)

        # Look for inversion patterns within each chromosome
        for chrom, breakends in chrom_groups.items():
            if len(breakends) < 4:  # Need at least 4 breakends for inversion
                continue

            # Sort by position
            breakends_sorted = sorted(breakends, key=lambda x: x.pos)

            # Check for inversion patterns - simplified heuristic
            # Look for breakends that point inward (toward each other)
            for i in range(len(breakends_sorted) - 1):
                breakend1 = breakends_sorted[i]
                breakend2 = breakends_sorted[i + 1]

                if (breakend1.id in processed_ids or breakend2.id in processed_ids or
                    not breakend1.mate_breakend or not breakend2.mate_breakend):
                    continue

                # Check if this is part of an inversion pattern
                # For inversion: we expect 4 breakends on same chromosome with crossed connections
                if (breakend1.chrom == breakend2.chrom == chrom and
                    breakend1.mate_chrom == breakend2.mate_chrom == chrom and
                    breakend1.mate_breakend and breakend2.mate_breakend):

                    # Check if the 4 breakends form inversion pattern
                    # Simple heuristic: 4 breakends all pointing to each other on same chromosome
                    same_chrom_count = len([b for b in breakends_sorted if b.chrom == chrom and b.mate_chrom == chrom])

                    if same_chrom_count == 4:
                        # Reclassify all 4 breakends as inversion
                        for breakend_to_classify in breakends_sorted:
                            if breakend_to_classify.id in processed_ids:
                                continue

                            # Reclassify as inversion
                            inv_breakend = Breakend.from_breakend_variant(
                                BreakendVariant(
                                    id=breakend_to_classify.id,
                                    chrom=breakend_to_classify.chrom,
                                    pos=breakend_to_classify.pos,
                                    ref=breakend_to_classify.ref,
                                    alt=breakend_to_classify.alt,
                                    mate_id=getattr(breakend_to_classify, 'mate_id', ''),
                                    mate_chrom=breakend_to_classify.mate_chrom,
                                    mate_pos=breakend_to_classify.mate_pos,
                                    orientation=breakend_to_classify.orientation,
                                    inserted_seq=breakend_to_classify.inserted_seq,
                                    info='',
                                    variant_type='SV_BND_INV'
                                ), 'SV_BND_INV'
                            )
                            inv_breakends.append(inv_breakend)
                            processed_ids.add(breakend_to_classify.id)

                        # Exit the loop since we processed all breakends for this chromosome
                        break

        return inv_breakends

    def _infer_mate_orientation(self, original_orientation: str) -> str:
        """
        Infer the orientation of a missing mate breakend based on the original breakend's orientation.

        BND orientation pairs (original -> inferred mate):
        - t[p[ -> ]p]t
        - t]p] -> [p[t
        - ]p]t -> t[p[
        - [p[t -> t]p]
        """
        orientation_pairs = {
            't[p[': ']p]t',
            't]p]': '[p[t',
            ']p]t': 't[p[',
            '[p[t': 't]p]'
        }
        return orientation_pairs.get(original_orientation, ']p]t')

    def _parse_and_index_breakends(self, bnd_variants: pd.DataFrame):
        """Parse breakends and build coordinate index."""
        for _, variant in bnd_variants.iterrows():
            try:
                # Parse ALT field
                alt_info = parse_breakend_alt(variant['alt'])
                if not alt_info['is_valid']:
                    warnings.warn(f"Could not parse ALT field for {variant['id']}: {variant['alt']}")
                    continue

                # Parse INFO field for optional MATEID
                info_dict = parse_vcf_info(variant.get('info', ''))
                mate_id = info_dict.get('MATEID', None)

                # Create BreakendVariant
                breakend_var = BreakendVariant(
                    id=variant['id'],
                    chrom=variant['chrom'],
                    pos=variant['pos1'],
                    ref=variant['ref'],
                    alt=variant['alt'],
                    mate_id=mate_id,
                    mate_chrom=alt_info['mate_chrom'],
                    mate_pos=alt_info['mate_pos'],
                    orientation=alt_info['orientation'],
                    inserted_seq=alt_info['inserted_seq'],
                    info=variant.get('info', ''),
                    variant_type='SV_BND'
                )

                self.all_breakends.append(breakend_var)

                # Index by coordinates
                coord_key = (breakend_var.chrom, breakend_var.pos)
                self.coordinate_index[coord_key] = breakend_var

            except Exception as e:
                warnings.warn(f"Error processing breakend {variant['id']}: {e}")


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
    
    # TODO: use the read_tsv built in functionality for this instead
    with open(path, "r") as f:
        lines = [l for l in f if not l.startswith("##")]

    # Determine columns to read based on parameters
    if include_info:
        usecols = [0, 1, 2, 3, 4, 7]  # Include INFO field
        base_columns = ["chrom", "pos1", "id", "ref", "alt", "info"]
    else:
        usecols = [0, 1, 2, 3, 4]  # Include ID field by default
        base_columns = ["chrom", "pos1", "id", "ref", "alt"]

    # TODO: use read_tsv instead
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
    # Note: INV and DUP variants represented by multiple BNDs are handled by
    # BNDClassifier and group_variants_by_semantic_type() functions
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
        usecols = [0, 1, 2, 3, 4]  # Include ID field by default
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
        usecols = [0, 1, 2, 3, 4]  # Include ID field by default
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


def group_variants_by_semantic_type(variants_df: pd.DataFrame, vcf_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Group variants by semantic type for unified processing.

    This function groups variants so that DUP and SV_BND_DUP are processed together,
    INV and SV_BND_INV are processed together, etc.

    Args:
        variants_df: DataFrame with variants including variant_type column
        vcf_path: Optional VCF path for BND semantic classification

    Returns:
        Dict with keys: 'standard', 'dup_variants', 'inv_variants', 'bnd_variants'
    """
    grouped = {
        'standard': pd.DataFrame(),
        'dup_variants': pd.DataFrame(),
        'inv_variants': pd.DataFrame(),
        'bnd_variants': pd.DataFrame()
    }

    # Standard variants (SNV, INS, DEL, MNV)
    standard_types = ['SNV', 'MNV', 'INS', 'DEL', 'complex']
    grouped['standard'] = variants_df[variants_df['variant_type'].isin(standard_types)].copy()

    # Symbolic DUP variants
    dup_types = ['SV_DUP']
    grouped['dup_variants'] = variants_df[variants_df['variant_type'].isin(dup_types)].copy()

    # Symbolic INV variants
    inv_types = ['SV_INV']
    grouped['inv_variants'] = variants_df[variants_df['variant_type'].isin(inv_types)].copy()

    # Handle BND variants with semantic classification
    bnd_types = ['SV_BND', 'SV_BND_INS']
    bnd_variants = variants_df[variants_df['variant_type'].isin(bnd_types)]

    if len(bnd_variants) > 0 and vcf_path:
        # Use BNDClassifier to get semantic classifications
        classifier = BNDClassifier()
        classified_breakends = classifier.classify_all_breakends(vcf_path)

        # Extract variant IDs for each semantic type
        dup_bnd_ids = {b.id for b in classified_breakends.get('dup_breakends', [])}
        inv_bnd_ids = {b.id for b in classified_breakends.get('inv_breakends', [])}
        true_bnd_ids = {b.id for b in classified_breakends.get('paired', [])}

        # Group BND variants by semantic type
        dup_bnd_variants = bnd_variants[bnd_variants['id'].isin(dup_bnd_ids)].copy()
        inv_bnd_variants = bnd_variants[bnd_variants['id'].isin(inv_bnd_ids)].copy()
        true_bnd_variants = bnd_variants[bnd_variants['id'].isin(true_bnd_ids)].copy()

        # Update variant_type for semantic consistency
        dup_bnd_variants['variant_type'] = 'SV_BND_DUP'
        inv_bnd_variants['variant_type'] = 'SV_BND_INV'

        # Combine with symbolic variants
        grouped['dup_variants'] = pd.concat([grouped['dup_variants'], dup_bnd_variants], ignore_index=True)
        grouped['inv_variants'] = pd.concat([grouped['inv_variants'], inv_bnd_variants], ignore_index=True)
        grouped['bnd_variants'] = true_bnd_variants.copy()
    else:
        # No BND semantic classification possible
        grouped['bnd_variants'] = bnd_variants.copy()

    return grouped


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
    Note: This function only correctly classifies variants that are represented in a single
    VCF record, this means that an additional classification step is needed for BNDs that 
    actually represent INV or DUP variants as those can be represented as 4 or 2 VCF records
    respectively.


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
        - 'SV_BND_INS': Breakend/translocation with inserted sequence
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
        classify_variant_type('T', ']chr2:20]ATCGT') → 'SV_BND_INS'
        
        # Special cases
        classify_variant_type('T', '*') → 'missing'
        
    VCF 4.2 Reference: https://samtools.github.io/hts-specs/VCFv4.2.pdf
    """
    if not ref_allele or not alt_allele:
        return 'missing_ref_or_alt'
    
    # Normalize alleles (VCF allows mixed case)
    ref = ref_allele.upper().strip()
    alt = alt_allele.upper().strip()
    
    # PRIORITY 0: Multiallelic variants (comma-separated ALT alleles)
    # Multiple alternative alleles in single ALT field indicate complex variant
    if ',' in alt:
        return 'multiallelic'
    
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
        elif sv_type in ['DUP','DUP:TANDEM']:
            return 'SV_DUP'
        elif sv_type in ['DEL']:
            return 'SV_DEL'
        elif sv_type in ['INS']:
            return 'SV_INS'
        elif sv_type in ['CNV']:
            return 'SV_CNV'
        elif sv_type in ['BND', 'TRA']:
            return 'SV_BND'
        else:
            # Fallback to returning the ALT
            return alt


    # PRIORITY 3: Breakend notation (complex rearrangements)
    # Format examples: A[chr2:1000[, ]chr1:100]T, etc.
    breakend_pattern = r'[\[\]]'
    if re.search(breakend_pattern, alt):
        # Check if BND has inserted sequence by parsing the ALT field
        try:
            breakend_info = parse_breakend_alt(alt)
            if breakend_info['is_valid'] and breakend_info['inserted_seq']:
                return 'SV_BND_INS'  # BND with insertion
            else:
                return 'SV_BND'      # Standard BND
        except:
            # If parsing fails, fallback to returning the ALT
            return alt
    
    # PRIORITY 4: Check SVTYPE in INFO field for additional SV classification
    # Note: Symbolic ALT fields (<INV>, <DUP>) are handled by priority 3, so this mainly
    # serves as fallback for non-standard VCF files
    if info_dict and 'SVTYPE' in info_dict:
        svtype = str(info_dict['SVTYPE']).upper()
        if svtype in ['INV']:
            return 'SV_INV'
        elif svtype in ['DUP']:
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
            return alt
            
    elif ref_len == 1 and alt_len > 1:
        # Potential insertion: check if REF is prefix of ALT
        if alt.startswith(ref):
            return 'INS'
        else:
            # REF not a prefix - complex variant
            return alt
            
    elif ref_len > 1 and alt_len == 1:
        # Potential deletion: check if ALT is prefix of REF  
        if ref.startswith(alt):
            return 'DEL'
        else:
            # ALT not a prefix - complex variant
            return alt
            
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
                return alt
        else:
            # Limited overlap - substitution
            return 'MNV'
    
    else:
        # Not parsed - should not occur in valid VCF
        return alt


def parse_breakend_alt(alt_allele: str) -> Dict:
    """
    Parse breakend ALT field to extract mate information and inserted sequence.

    Args:
        alt_allele: ALT field from BND variant (e.g., "G]17:198982]", "]13:123456]AGTNNNNNCAT")

    Returns:
        dict: Parsed breakend information with keys:
            - 'mate_chrom': Chromosome of mate breakend
            - 'mate_pos': Position of mate breakend (1-based)
            - 'orientation': Breakend orientation ('t[p[', 't]p]', ']p]t', '[p[t')
            - 'inserted_seq': Novel sequence inserted at junction (empty string if none)
            - 'is_valid': Boolean indicating if ALT field was successfully parsed

    Breakend ALT format examples (VCF 4.2):
        - t[p[: piece extending to the right of p is joined after t
        - t]p]: reverse comp piece extending left of p is joined after t
        - ]p]t: piece extending to the left of p is joined before t
        - [p[t: reverse comp piece extending right of p is joined before t

    Examples:
        parse_breakend_alt("G]17:198982]")
        → {'mate_chrom': '17', 'mate_pos': 198982, 'orientation': 't]p]',
           'inserted_seq': '', 'is_valid': True}

        parse_breakend_alt("]13:123456]AGTNNNNNCAT")
        → {'mate_chrom': '13', 'mate_pos': 123456, 'orientation': ']p]t',
           'inserted_seq': 'AGTNNNNNCAT', 'is_valid': True}
    """
    import re

    result = {
        'mate_chrom': None,
        'mate_pos': None,
        'orientation': None,
        'inserted_seq': '',
        'is_valid': False
    }

    if not alt_allele or not isinstance(alt_allele, str):
        return result

    # Patterns for the four breakend orientations
    # t[p[ format: sequence + [ + position + [
    pattern1 = r'^(.+?)\[([^:]+):(\d+)\[$'
    # t]p] format: sequence + ] + position + ]
    pattern2 = r'^(.+?)\]([^:]+):(\d+)\]$'
    # ]p]t format: ] + position + ] + sequence
    pattern3 = r'^\]([^:]+):(\d+)\](.+?)$'
    # [p[t format: [ + position + [ + sequence
    pattern4 = r'^\[([^:]+):(\d+)\[(.+?)$'

    # Try each pattern
    match = re.match(pattern1, alt_allele)
    if match:
        prefix_seq, mate_chrom, mate_pos = match.groups()
        result['mate_chrom'] = mate_chrom
        result['mate_pos'] = int(mate_pos)
        result['orientation'] = 't[p['  # t[p[
        result['inserted_seq'] = prefix_seq[1:] if len(prefix_seq) > 1 else ''  # Remove reference base
        result['is_valid'] = True
        return result

    match = re.match(pattern2, alt_allele)
    if match:
        prefix_seq, mate_chrom, mate_pos = match.groups()
        result['mate_chrom'] = mate_chrom
        result['mate_pos'] = int(mate_pos)
        result['orientation'] = 't]p]'  # t]p]
        result['inserted_seq'] = prefix_seq[1:] if len(prefix_seq) > 1 else ''  # Remove reference base
        result['is_valid'] = True
        return result

    match = re.match(pattern3, alt_allele)
    if match:
        mate_chrom, mate_pos, suffix_seq = match.groups()
        result['mate_chrom'] = mate_chrom
        result['mate_pos'] = int(mate_pos)
        result['orientation'] = ']p]t'  # ]p]t
        result['inserted_seq'] = suffix_seq[:-1] if len(suffix_seq) > 1 else ''  # Remove reference base
        result['is_valid'] = True
        return result

    match = re.match(pattern4, alt_allele)
    if match:
        mate_chrom, mate_pos, suffix_seq = match.groups()
        result['mate_chrom'] = mate_chrom
        result['mate_pos'] = int(mate_pos)
        result['orientation'] = '[p[t'  # [p[t
        result['inserted_seq'] = suffix_seq[:-1] if len(suffix_seq) > 1 else ''  # Remove reference base
        result['is_valid'] = True
        return result

    return result


def validate_breakend_pair(bnd1: Dict, bnd2: Dict) -> Dict:
    """
    Validate that two breakend variants form a consistent mate pair.

    Args:
        bnd1: First breakend variant (dict with id, mate_id, chrom, pos, etc.)
        bnd2: Second breakend variant (dict with id, mate_id, chrom, pos, etc.)

    Returns:
        dict: Validation result with keys:
            - 'is_valid': Boolean indicating if pair is valid
            - 'errors': List of validation error messages
            - 'warnings': List of validation warning messages
    """
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    # Check that they reference each other as mates
    if bnd1.get('mate_id') != bnd2.get('id'):
        result['errors'].append(f"BND {bnd1.get('id')} MATEID {bnd1.get('mate_id')} does not match mate ID {bnd2.get('id')}")
        result['is_valid'] = False

    if bnd2.get('mate_id') != bnd1.get('id'):
        result['errors'].append(f"BND {bnd2.get('id')} MATEID {bnd2.get('mate_id')} does not match mate ID {bnd1.get('id')}")
        result['is_valid'] = False

    # Check that mate positions are consistent with actual positions
    if bnd1.get('mate_chrom') != bnd2.get('chrom'):
        result['errors'].append(f"BND {bnd1.get('id')} mate chromosome {bnd1.get('mate_chrom')} does not match actual chromosome {bnd2.get('chrom')}")
        result['is_valid'] = False

    if bnd1.get('mate_pos') != bnd2.get('pos'):
        result['errors'].append(f"BND {bnd1.get('id')} mate position {bnd1.get('mate_pos')} does not match actual position {bnd2.get('pos')}")
        result['is_valid'] = False

    # Check orientation consistency (complex logic depending on rearrangement type)
    orientation1 = bnd1.get('orientation')
    orientation2 = bnd2.get('orientation')

    # For now, just warn about complex orientation validation - this would need detailed implementation
    if orientation1 and orientation2:
        result['warnings'].append(f"Orientation validation not fully implemented: {orientation1} vs {orientation2}")

    return result


def create_breakend_pairs(variants_df: pd.DataFrame) -> List[BreakendPair]:
    """
    Create BreakendPair objects from BND variants in a DataFrame.

    This function pairs breakend variants based on coordinate matching rather than MATEID,
    making it more robust and not dependent on optional INFO fields.

    Args:
        variants_df: DataFrame containing BND variants with variant_type='SV_BND'

    Returns:
        List of BreakendPair objects representing valid breakend pairs

    Notes:
        - Pairs breakends by matching coordinates from ALT field parsing
        - Does not require MATEID field to be present
        - Issues warnings for unpaired or invalid breakends
    """
    # Filter for BND variants only (including BND with insertions)
    bnd_variants = variants_df[variants_df['variant_type'].isin(['SV_BND', 'SV_BND_INS'])].copy()

    if len(bnd_variants) == 0:
        return []

    # Parse all breakend variants
    breakend_variants = []
    for _, variant in bnd_variants.iterrows():
        try:
            # Parse ALT field to get mate information
            breakend_info = parse_breakend_alt(variant['alt'])

            if not breakend_info['is_valid']:
                warnings.warn(f"Could not parse breakend ALT field for variant {variant['id']}: {variant['alt']}")
                continue

            # Parse INFO field for optional MATEID (but don't require it)
            info_dict = parse_vcf_info(variant.get('info', ''))
            mate_id = info_dict.get('MATEID', None)

            # Create BreakendVariant object
            breakend = BreakendVariant(
                id=variant['id'],
                chrom=variant['chrom'],
                pos=variant['pos1'],
                ref=variant['ref'],
                alt=variant['alt'],
                mate_id=mate_id,  # May be None
                mate_chrom=breakend_info['mate_chrom'],
                mate_pos=breakend_info['mate_pos'],
                orientation=breakend_info['orientation'],
                inserted_seq=breakend_info['inserted_seq'],
                info=variant.get('info', ''),
                variant_type='SV_BND'
            )
            breakend_variants.append(breakend)

        except Exception as e:
            warnings.warn(f"Error processing breakend variant {variant['id']}: {e}")
            continue

    # Create pairs by coordinate matching
    pairs = []
    used_breakends = set()

    for i, bnd1 in enumerate(breakend_variants):
        if bnd1.id in used_breakends:
            continue

        # Find mate by coordinate matching
        mate_found = False
        for j, bnd2 in enumerate(breakend_variants):
            if i == j or bnd2.id in used_breakends:
                continue

            # Check if these breakends are mates based on coordinates
            if (bnd1.mate_chrom == bnd2.chrom and bnd1.mate_pos == bnd2.pos and
                bnd2.mate_chrom == bnd1.chrom and bnd2.mate_pos == bnd1.pos):

                try:
                    # Create pair (validation happens in BreakendPair.__post_init__)
                    pair = BreakendPair(bnd1, bnd2)
                    pairs.append(pair)
                    used_breakends.add(bnd1.id)
                    used_breakends.add(bnd2.id)
                    mate_found = True
                    break

                except Exception as e:
                    warnings.warn(f"Invalid breakend pair {bnd1.id}-{bnd2.id}: {e}")
                    continue

        if not mate_found:
            warnings.warn(f"No mate found for breakend {bnd1.id} at {bnd1.chrom}:{bnd1.pos}")

    return pairs


def load_breakend_variants(variants_fn: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[Tuple]]:
    """
    Load variants and separate BND variants into pairs using enhanced classifier.

    Args:
        variants_fn: Path to VCF file or DataFrame with variant data

    Returns:
        Tuple of (standard_variants_df, breakend_pairs_list)
        - standard_variants_df: DataFrame with non-BND variants
        - breakend_pairs_list: List of tuples (bnd1, bnd2) for BND pairs
    """
    # Import here to avoid circular imports
    from .personalize import _load_variants

    # Load all variants with proper normalization and classification
    if isinstance(variants_fn, str):
        all_variants = read_vcf(variants_fn, include_info=True, classify_variants=True)
        vcf_path = variants_fn
    else:
        # Use the existing _load_variants function which properly handles
        # DataFrame normalization (pos->pos1, variant_type, etc.)
        all_variants = _load_variants(variants_fn)
        vcf_path = None

    # Separate BND and standard variants (including all BND types)
    bnd_variants = all_variants[all_variants['variant_type'].isin(['SV_BND', 'SV_BND_INS', 'SV_BND_DUP', 'SV_BND_INV'])]
    standard_variants = all_variants[~all_variants['variant_type'].isin(['SV_BND', 'SV_BND_INS', 'SV_BND_DUP', 'SV_BND_INV'])]

    # Create breakend pairs using enhanced classifier
    breakend_pairs = []
    if len(bnd_variants) > 0 and vcf_path:
        classifier = BNDClassifier()
        classified_breakends = classifier.classify_all_breakends(vcf_path)

        # Convert classified breakends to pairs
        paired_breakends = classified_breakends['paired']
        processed_ids = set()

        for breakend in paired_breakends:
            if breakend.id in processed_ids:
                continue

            if breakend.mate_breakend:
                pair = (breakend, breakend.mate_breakend)
                breakend_pairs.append(pair)
                processed_ids.add(breakend.id)
                processed_ids.add(breakend.mate_breakend.id)

    return standard_variants, breakend_pairs


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

