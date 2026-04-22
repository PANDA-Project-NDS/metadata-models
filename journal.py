import re
from typing import List, Optional, TypeVar, Generic, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator

T = TypeVar("T")

# Supported currency literals
SupportedCurrency = Literal["USD", "EUR"]


class Evidence(BaseModel):
    """Container for evidence supporting an extracted value."""
    quote: str = Field(description="Verbatim sentence or fragment from the source text.")
    source: str = Field(description="Source identifier (file name, URL, or URI).")


class Sourced(BaseModel, Generic[T]):
    """A value paired with its supporting evidence."""
    value: T
    evidence: Optional[Evidence] = Field(default=None, description="Evidence supporting this specific value.")


class PublicationFrequency(BaseModel):
    """Structured information about how often the journal is published."""
    frequency: Optional[Sourced[str]] = Field(default=None,
                                              description="Human readable frequency label (e.g., 'Monthly', 'Quarterly').")
    issues_per_year: Optional[Sourced[int]] = Field(default=None,
                                                    description="Exact number of issues per year when available.")


class ReviewProcess(BaseModel):
    """Details regarding the peer review workflow and type."""
    type: Optional[Sourced[Optional[Literal["single-blind", "double-blind", "open", "other"]]]] = Field(
        default=None,
        description="Canonical review type."
    )
    description: Optional[Sourced[str]] = Field(default=None,
                                                description="Freeform description of the review workflow.")


class Membership(BaseModel):
    """Information about society or institutional membership models."""
    type: Optional[Sourced[str]] = Field(default=None,
                                         description="Membership type (e.g., 'society', 'institutional').")
    details: Optional[Sourced[str]] = Field(default=None, description="Further membership details.")


class ISSNRaw(BaseModel):
    """Original raw ISSN values if canonical parsing fails."""
    print: Optional[Sourced[str]] = Field(default=None, description="Original raw print ISSN.")
    online: Optional[Sourced[str]] = Field(default=None, description="Original raw online ISSN.")


class ISSN(BaseModel):
    """International Standard Serial Number identifiers."""
    print: Optional[Sourced[str]] = Field(default=None, description="Print ISSN in NNNN-NNNN form.")
    online: Optional[Sourced[str]] = Field(default=None, description="Online ISSN in NNNN-NNNN form.")
    raw: Optional[ISSNRaw] = Field(default=None, description="Raw values for reference.")

    @field_validator("print", "online")
    @classmethod
    def validate_issn_format(cls, v: Optional[Sourced[str]]) -> Optional[Sourced[str]]:
        if v is not None and v.value is not None:
            if not re.match(r"^\d{4}-\d{3}[\dX]$", v.value):
                raise ValueError("ISSN must follow the NNNN-NNNN format")
        return v


class Fee(BaseModel):
    """Numeric fee amount and its associated currency."""
    amount: Optional[Sourced[float]] = Field(default=None, description="Numeric fee value.")
    currency: Optional[Sourced[SupportedCurrency]] = Field(default=None,
                                                           description="ISO 4217 currency code (USD or EUR).")


class APC(BaseModel):
    """Article Processing Charge details for a specific category or article type."""
    article_type: Optional[Sourced[str]] = Field(default=None, description="Article type this fee applies to.")
    category: Optional[Sourced[str]] = Field(default=None,
                                             description="Optional category label (e.g., Frontiers-style categories).")
    fee: Optional[Fee] = Field(default=None, description="Parsed numeric fee.")
    raw: Optional[Sourced[str]] = Field(default=None, description="Original textual fee value as found in the source.")
    description: Optional[Sourced[str]] = Field(default=None, description="Human-readable notes about the fee.")
    regular_fee: Optional[Sourced[str]] = Field(default=None, description="Table-specific field for regular pricing.")
    footnote: Optional[Sourced[str]] = Field(default=None, description="Footnote text associated with the fee entry.")


class Discount(BaseModel):
    """Information about waivers or discounts available for publication fees."""
    type: Optional[Sourced[str]] = Field(default=None, description="Discount type label.")
    amount: Optional[Fee] = Field(default=None, description="Numeric discount amount.")
    eligibility: Optional[Sourced[str]] = Field(default=None, description="Criteria for discount eligibility.")
    raw: Optional[Sourced[str]] = Field(default=None, description="Original textual discount description.")


class ArticleType(BaseModel):
    """Definition of an article type supported by the journal."""
    type: Sourced[str] = Field(..., description="The name of the article type.")
    notes: Optional[Sourced[str]] = Field(default=None, description="Optional supplementary notes.")


class Editor(BaseModel):
    """Member of the journal's editorial board."""
    name: Sourced[str] = Field(..., description="Full name of the editor.")
    role: Optional[Sourced[str]] = Field(default=None, description="Role or title (e.g., 'Editor-in-Chief').")
    affiliations: Optional[List[Sourced[str]]] = Field(default_factory=list,
                                                       description="List of institutional affiliations.")


class AdditionalInformation(BaseModel):
    """Publisher-specific metadata and policy statements."""
    open_access_statement: Optional[Sourced[str]] = Field(default=None, description="Open access policy statement.")
    copyright_statement: Optional[Sourced[str]] = Field(default=None, description="Copyright and licensing statement.")
    quality_assurance: Optional[Sourced[str]] = Field(default=None,
                                                      description="Notes on quality assurance or peer review standards.")
    other: Optional[List[Sourced[str]]] = Field(default_factory=list,
                                                description="Other minor publisher-specific fields.")


class ImpactMetrics(BaseModel):
    """Quantifiable journal metrics."""
    cite_score: Optional[Sourced[float]] = Field(default=None, description="CiteScore metric.")
    impact_factor: Optional[Sourced[float]] = Field(default=None, description="Impact Factor metric.")
    other: Optional[List[Sourced[float]]] = Field(default_factory=list, description="Other numeric impact metrics.")


class Facts(BaseModel):
    """Brief metadata summary, often found in 'Journal Facts' sidebars."""
    short_name: Optional[Sourced[str]] = Field(default=None, description="Shortened journal name.")
    abbreviation: Optional[Sourced[str]] = Field(default=None, description="Journal abbreviation.")
    indexed_in: Optional[Sourced[str]] = Field(default=None, description="List of indexing services.")
    impact: Optional[ImpactMetrics] = Field(default=None, description="Impact metrics associated with the facts block.")


# --- Modular Domain Blocks ---

class JournalIdentity(BaseModel):
    """Core identification for the journal."""
    title: Optional[Sourced[str]] = Field(default=None, description="Canonical journal title.")
    publisher: Optional[Sourced[str]] = Field(default=None, description="Publisher name.")
    issn: Optional[ISSN] = Field(default=None, description="ISSN identifiers.")


class JournalScope(BaseModel):
    """Information about the journal's focus and sections."""
    description: Optional[Sourced[str]] = Field(default=None, description="Prose summary of focus and scope.")
    mission_scope: Optional[Sourced[str]] = Field(default=None, description="Concise mission and scope summary.")
    journal_sections: Optional[List[Sourced[str]]] = Field(default_factory=list,
                                                           description="Sections within the journal.")


class SubmissionInfo(BaseModel):
    """Details regarding article submissions."""
    submission_guidelines: Optional[Sourced[str]] = Field(default=None, description="Full submission guidelines text.")
    article_types: Optional[List[ArticleType]] = Field(default_factory=list,
                                                       description="List of supported article types.")


class ReviewAndPolicy(BaseModel):
    """Peer review workflow and additional publisher policies."""
    review_process: Optional[ReviewProcess] = Field(default=None, description="Review workflow details.")
    additional_information: Optional[AdditionalInformation] = Field(default=None,
                                                                    description="Publisher-specific policy details.")


class Pricing(BaseModel):
    """Article processing charges and discounts."""
    article_processing_charges: Optional[List[APC]] = Field(default_factory=list,
                                                            description="Article Processing Charges.")
    discounts: Optional[List[Discount]] = Field(default_factory=list, description="Waivers and discounts.")


class Editorial(BaseModel):
    """Editorial board and staff."""
    editors: Optional[List[Editor]] = Field(default_factory=list, description="Editorial board members.")


# --- Agent Extraction Targets ---

class BasicInfoExtraction(BaseModel):
    """Pass 1: Extracts basic journal information, scope, and identifiers."""
    identity: Optional[JournalIdentity] = Field(default=None)
    scope: Optional[JournalScope] = Field(default=None)
    facts: Optional[Facts] = Field(default=None)


class PoliciesExtraction(BaseModel):
    """Pass 2: Extracts publication frequency, submission guidelines, and review policies."""
    publication_frequency: Optional[PublicationFrequency] = Field(default=None)
    submissions: Optional[SubmissionInfo] = Field(default=None)
    policies: Optional[ReviewAndPolicy] = Field(default=None)


class FeesExtraction(BaseModel):
    """Pass 3: Extracts fees, APCs, discounts, and membership information."""
    pricing: Optional[Pricing] = Field(default=None)
    membership: Optional[Membership] = Field(default=None)


class PeopleMetricsExtraction(BaseModel):
    """Pass 4: Extracts editorial board members and impact metrics."""
    editorial: Optional[Editorial] = Field(default=None)
    metrics: Optional[ImpactMetrics] = Field(default=None)


# --- Final Schema ---

class JournalMetadata(BaseModel):
    """
    Canonical journal metadata schema.
    Composed of multiple modular sub-schemas for targeted extraction passes.
    """
    model_config = ConfigDict(title="JournalMetadata")

    basic_info: BasicInfoExtraction
    policies: PoliciesExtraction
    fees: FeesExtraction
    people_metrics: PeopleMetricsExtraction
