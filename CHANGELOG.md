# Changelog


## v0.5.0 (07/21/2025)

- **Chromosome Name Matching**: Added intelligent heuristics to handle chromosome name mismatches between FASTA and VCF files (e.g., `chr1` ↔ `1`, `chrM` ↔ `MT`)
- **Chunked VCF Processing**: Implemented memory-efficient chunked processing for large VCF files with `chunk_size` parameter across all VCF-processing functions
- **Enhanced API**: All functions now support `chunk_size=1` parameter for backward compatibility and memory efficiency
- **Comprehensive PAM Disruption Testing**: Added extensive test suite for PAM disruption functionality covering multiple scenarios and edge cases
- **Enhanced Error Handling**: Improved chromosome matching with detailed reporting and warnings for unmatched chromosomes
- **Updated Documentation**: Comprehensive README with GitHub installation instructions, new features, and performance tips
- **Test Coverage**: Added 67 tests with 87% code coverage including chunked processing and chromosome matching scenarios

## v0.4.0 (06/14/2025)

- Ovelapping indels are handled according to the same strategy as bcftools consensus. After an indel is applied that position is frozen and no other variants can be applied to the same region.

- Test cases for variant application



## v0.1.0 (01/05/2025)

- First release of `supremo_lite`!