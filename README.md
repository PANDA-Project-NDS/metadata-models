# Journal Metadata Models

This repository defines a standalone domain specification for journal metadata, derived from the [Journal Metadata Exchange Format (JMEF) v1.0](https://journalmetadata.org/version-1.0/doc.html). It extends the JMEF specification to support the high-precision requirements of APC research and evidence-based extraction.

## Key Concepts

### Evidence-Sourced Values
To support a "Two-Judge" evaluation system, the models support two modes of operation controlled by the `WITH_EVIDENCE` environment variable:

- **Evidence Mode (`WITH_EVIDENCE=1`)**: Uses `EvidenceSourcedValue`. Every extracted value is paired with an `Evidence` object containing the verbatim quote from the source text and a source identifier.
- **Clean Mode (`WITH_EVIDENCE=0`)**: Uses `CleanSourcedValue`. Only the final extracted value is stored.

This allows the pipeline to extract data with full traceability and then strip the evidence to produce a clean JSON output for downstream use.

### Modular Schema
The canonical `JournalMetadata` model is a composite of four modular sub-schemas. This design enables targeted extraction passes, reducing the token context required for each LLM call:

1. **Basic Info**: Title, publisher, ISSN, scope, facts, and metrics.
2. **Policies**: Publication frequency, submission guidelines, review policies, and open access criteria.
3. **Fees**: Article Processing Charges (APCs), discounts, and membership models.
4. **Editorial**: Editorial board members, their roles, and affiliations.

### Extended Pricing Model (Beyond JMEF)
While JMEF primarily treats pricing as a binary "no-fees" criterion for Diamond OA qualification, this specification implements a comprehensive pricing engine:
- **Granular APCs**: Maps specific fees to article types (e.g., "Review" vs "Research") or custom categories.
- **Monetary Precision**: Enforces ISO 4217 currency standards to ensure consistent and accurate financial tracking.
- **Complex Discounts**: Tracks fixed, percent, and waiver types, capturing the verbatim eligibility criteria.
- **Membership Dependency**: Tracks whether pricing and discounts are contingent upon society or institutional membership.

## Schema Components

### Core Models
- `JournalMetadata`: The root schema composing all extraction passes.
- `ISSN`: Handles print, online, and linking ISSNs with built-in format validation (`NNNN-NNNN`).
- `APC`: Defines pricing per article type or category.
- `Editor`: Captures editorial board members with institutional affiliations.
- `DiamondOpenAccess`: Implements classification based on DIAMAS and CRAFT-OA project standards.

### Vocabularies (`vocab.py`)
To ensure data canonicalization, the package defines several `Literal` types for categorical values:
- `ArticleTypeValue`: Based on COARS resource type leaf nodes.
- `IndexingService`: List of supported indexing services (e.g., Scopus, Web of Science, DOAJ).
- `Frequency`: Canonical publication frequencies (e.g., Monthly, Quarterly).
- `ReviewType`: Peer review workflows (e.g., single-blind, double-blind, open-review).

## Validation
The models employ Pydantic validators to maintain high data quality:
- **Format Validation**: Enforces strict regex for ISSNs.
- **Type Coercion**: Ensures monetary values are rounded to integers and affiliations are stored as sets.
- **Logic Validation**: Ensures that 'fixed' discounts have an associated amount and 'percent' discounts have a percentage.

## Scripts
The repository includes utility scripts for schema distribution:
- `scripts/dump_clean_schema.py`: Generates a self-contained JSON schema with all internal references inlined.
- `scripts/metadata_schema_html.py`: Generates a human-readable HTML version of the schema documentation.

## Usage

To use the models in another module:

```python
from models.journal import JournalMetadata

# The model will automatically adjust based on WITH_EVIDENCE env var
metadata = JournalMetadata(
    title="Example Journal",
    # ... other fields
)
```
