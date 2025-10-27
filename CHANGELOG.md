# Changelog

## v0.5.4 (10/24/2025)

### Critical Bug Fixes
- **`get_alt_sequences()` Variant Isolation**: Fixed critical bug where ALL variants were being applied together instead of individually
  - **Impact**: Previously, each window contained effects from ALL variants in the chunk/chromosome, not just the single variant it was centered on

### Major Features
- **PAM Disruption INDEL Detection**: Enhanced `get_pam_disrupting_alt_sequences` (renamed from `get_pam_disrupting_personal_sequences`) to correctly detect when INDELs create new PAM sites
  - **Key Enhancement**: Variants that create new PAMs or shift existing PAMs are now correctly identified as NOT disrupting (PAM remains functional)
  - **Use Case**: Critical for CRISPR resistance analysis where INDELs might form new PAM sites
  - **Implementation**: Compares PAM sites in reference vs. alternate sequences, accounting for positional shifts from INDELs

### Breaking Changes
- **Function Renamed**: `get_pam_disrupting_personal_sequences` → `get_pam_disrupting_alt_sequences` for consistency with naming conventions (`get_alt_sequences`, `get_alt_ref_sequences`)

### Documentation
- **Variant Classification Flowchart**: Added comprehensive variant classification flowchart (SVG and PNG) showing the decision tree for automatic variant type detection
- **Updated Notebooks**: Updated all tutorial notebook
- **Documentation Cleanup**: Removed deprecated notebooks
- **New User Guide**: Added comprehensive [PAM Disruption Analysis](docs/user_guide/pam_disruption.md) guide with INDEL detection examples
- **Updated References**: Updated all documentation and examples to use new function name

### Code Quality
- **Test Improvements**: Enhanced test coverage for contact map alignment and prediction alignment edge cases
- **Single-Variant Isolation Tests**: Added comprehensive tests to verify each variant window contains only its specific variant
- **INDEL PAM Formation Tests**: Added new test suite (`test_pam_indel_formation.py`) with 8 tests covering deletion/insertion PAM creation scenarios
- **Mock Model Refinements**: Improved TestModel2D implementation for better matrix handling
- **Bug Fixes**: Fixed minor issues in prediction alignment for edge cases

## v0.5.3 (10/17/2025)

### Major Features
- **BND Variant Support**: Complete support for breakend (BND) translocations and complex structural variants
  - Multi-phase variant processing with coordinate tracking
  - Automatic BND classification to detect duplications and inversions
  - Proper handling of inter-chromosomal and intra-chromosomal breakends
  - Chimeric reference sequence creation for translocations
- **Prediction Alignment System**: New `align_predictions_by_coordinate()` function
  - Aligns reference and alternate predictions accounting for coordinate changes from variants
  - Supports 1D and 2D predictions
  - Handles all variant types: SNV, INS, DEL, DUP, INV, BND
  - Cross-pattern masking for 2D inversions
- **Mock Models for Testing**: Added TestModel and TestModel2D
  - PyTorch-based mock genomic models for testing workflows
  - Configurable binning and cropping
  - Complete documentation and examples in notebooks
- **Brisket Integration**: Optional fast one-hot encoding
  - 10x faster sequence encoding when brisket is installed
  - Automatic fallback to numpy implementation
  - Install with: `pip install supremo_lite[fast]`
- **Custom Encoder Support**: All sequence generation functions now accept custom encoder functions
  - Allows integration with specialized encoding schemes
  - Backward compatible with default one-hot encoding

### Improvements
- **Enhanced Variant Classification**: Automatic structural variant type detection from VCF INFO fields
- **Chromosome Ordering**: Output sequences now maintain reference genome chromosome order
- **Metadata Tracking**: Enhanced metadata for BND variants including mate positions and orientations
- **SVLEN Extraction**: Automatic extraction of structural variant length from VCF INFO field


## v0.5.2 (08/19/2025)

- **Minimum dependency versions established**: Set minimum supported versions for all dependencies
  - Python: `^3.8`
  - torch: `>=1.13.0`
  - pandas: `>=1.5.0`
  - pyfaidx: `>=0.7.0`

## v0.5.1 (08/19/2025)


- **Fixed VCF position column handling**: All VCF reading functions now consistently treat the second column as `pos1` regardless of header name 
- **Added numeric validation**: Position columns are now validated to be numeric, throwing clear error messages for invalid data types
- **Simplified DataFrame input**: Removed complex column name handling logic in favor of consistent `pos1` column naming
- **Updated tests**: All test cases now use standardized `pos1` column names for consistency
- Addressed issues #5, #6, #7, #8

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