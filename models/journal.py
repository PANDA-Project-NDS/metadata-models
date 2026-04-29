import re
from typing import List, Optional, TypeVar, Generic, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator
from .vocab import Frequency, ReviewType, ArticleTypeValue, SupportedCurrency, IndexingService

T = TypeVar("T")


class Evidence(BaseModel):
    """Container for evidence supporting an extracted value."""

    model_config = ConfigDict(extra="forbid")
    quote: str = Field(
        description="Verbatim sentence or fragment from the source text."
    )
    source: str = Field(description="Source identifier (file name, URL, or URI).")


class SourcedModel(BaseModel):
    """Provide combined evidence for multiple values of a sub-object."""
    model_config = ConfigDict(extra="forbid")
    evidence: Optional[Evidence] = Field(
        default=None, description="Evidence supporting this specific value."
    )


class SourcedValue(SourcedModel, Generic[T]):
    """A value paired with its supporting evidence."""

    model_config = ConfigDict(extra="forbid")
    value: T


class PublicationFrequency(SourcedModel):
    """Structured information about how often the journal is published."""

    model_config = ConfigDict(extra="forbid")
    frequency: Optional[Frequency] = Field(
        default=None,
        description="Human readable frequency label (e.g., 'Monthly', 'Quarterly').",
    )
    issues_per_year: Optional[int] = Field(
        default=None, description="Exact number of issues per year when available."
    )


class ReviewProcess(BaseModel):
    """Details regarding the peer review workflow and type."""

    model_config = ConfigDict(extra="forbid")
    type: Optional[
        SourcedValue[ReviewType]
    ] = Field(
        default=None,
        description="Canonical review type. Null if not applicable.",
    )
    description: Optional[SourcedValue[str]] = Field(
        default=None,
        description="Summary of the review workflow.",
    )


class Membership(BaseModel):
    """Information about society or institutional membership models."""

    model_config = ConfigDict(extra="forbid")
    type: Optional[
        SourcedValue[Literal["society", "institutional", "individual", "corporate"]]
    ] = Field(
        default=None, description="Membership type (e.g., 'society', 'institutional')."
    )
    details: Optional[SourcedValue[str]] = Field(
        default=None, description="Further membership details."
    )


class ISSN(BaseModel):
    """International Standard Serial Number identifiers."""

    model_config = ConfigDict(extra="forbid")
    print: Optional[SourcedValue[str]] = Field(
        default=None, description="Print ISSN in NNNN-NNNN form."
    )
    online: Optional[SourcedValue[str]] = Field(
        default=None, description="Online ISSN in NNNN-NNNN form."
    )

    @field_validator("print", "online")
    @classmethod
    def validate_issn_format(
        cls, v: Optional[SourcedValue[str]]
    ) -> Optional[SourcedValue[str]]:
        if v is not None and v.value is not None:
            if not re.match(r"^\d{4}-\d{3}[\dX]$", v.value):
                raise ValueError("ISSN must follow the NNNN-NNNN format")
        return v


class MonetaryAmount(SourcedModel):
    """Numeric Monetary value with associated currency."""

    model_config = ConfigDict(extra="forbid")
    value: int = Field(..., description="Numeric money value (rounded).")
    currency: SupportedCurrency = Field(..., description="ISO 4217 currency code. USD and EUR only.")


class APC(SourcedModel):
    """Article Processing Charge details for a specific category or article type."""

    model_config = ConfigDict(extra="forbid")
    article_type: Optional[ArticleTypeValue] = Field(
        default=None, description="Article type this fee applies to."
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category label",
    )
    fee: MonetaryAmount = Field(..., description="Price of APC")


class Discount(SourcedModel):
    """Information about waivers or discounts available for publication fees."""

    model_config = ConfigDict(extra="forbid")
    type: Optional[Literal["waiver", "fixed", "percent"]] = Field(
        default=None, description="Discount type label."
    )
    amount: Optional[MonetaryAmount] = Field(default=None, description="Fixed monetary discount amount if applicable.")
    eligibility: Optional[str] = Field(
        default=None, description="Criteria for discount eligibility as stated."
    )


class ArticleType(SourcedModel):
    """Definition of an article type supported by the journal."""

    model_config = ConfigDict(extra="forbid")
    type: ArticleTypeValue = Field(..., description="The name of the article type.")
    notes: Optional[str] = Field(
        default=None, description="Optional supplementary notes."
    )


class Editor(SourcedModel):
    """Member of the journal's editorial board."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Full name of the editor.")
    role: Optional[str] = Field(
        default=None, description="Role or title as stated (e.g., 'Editor-in-Chief')."
    )
    affiliations: Optional[List[str]] = Field(
        default_factory=list, description="List of institutional affiliations. Institute names, not locations."
    )


class PublisherPolicies(BaseModel):
    """Publisher-specific metadata and policy statements."""

    model_config = ConfigDict(extra="forbid")

    open_access_statement: Optional[SourcedValue[str]] = Field(
        default=None, description="Open access policy statement."
    )
    copyright_statement: Optional[SourcedValue[str]] = Field(
        default=None, description="Copyright and licensing statement."
    )
    quality_assurance: Optional[SourcedValue[str]] = Field(
        default=None, description="Notes on quality assurance or peer review standards."
    )


class Metrics(BaseModel):
    """Quantifiable journal metrics."""

    model_config = ConfigDict(extra="forbid")

    cite_score: Optional[SourcedValue[float]] = Field(
        default=None, description="CiteScore metric."
    )
    impact_factor: Optional[SourcedValue[float]] = Field(
        default=None, description="Impact Factor metric."
    )
    acceptance_rate: Optional[SourcedValue[float]] = Field(default=None, description="Acceptance rate in percent (e.g., 23.5)")


class Facts(BaseModel):
    """Brief metadata summary, often found in 'Journal Facts' sidebars."""

    model_config = ConfigDict(extra="forbid")

    short_name: Optional[SourcedValue[str]] = Field(
        default=None, description="Shortened journal name."
    )
    abbreviation: Optional[SourcedValue[str]] = Field(
        default=None, description="Journal abbreviation."
    )
    indexed_in: Optional[List[IndexingService]] = Field(
        default=None, description="Indexing services."
    )


# --- Modular Domain Blocks ---


class SubmissionInfo(BaseModel):
    """Details regarding article submissions."""

    model_config = ConfigDict(extra="forbid")

    submission_guidelines: Optional[SourcedValue[str]] = Field(
        default=None, description="Full submission guidelines text."
    )
    article_types: Optional[List[ArticleType]] = Field(
        default_factory=list, description="List of supported article types."
    )


class ReviewAndPolicy(BaseModel):
    """Peer review workflow and additional publisher policies."""

    model_config = ConfigDict(extra="forbid")

    review_process: Optional[ReviewProcess] = Field(
        default=None, description="Review workflow details."
    )
    additional_information: Optional[PublisherPolicies] = Field(
        default=None, description="Publisher-specific policy details."
    )


class Pricing(BaseModel):
    """Article processing charges and discounts."""

    model_config = ConfigDict(extra="forbid")

    article_processing_charges: Optional[List[APC]] = Field(
        default_factory=list, description="Article Processing Charges."
    )
    discounts: Optional[List[Discount]] = Field(
        default_factory=list, description="Waivers and discounts."
    )


class Editorial(BaseModel):
    """Editorial board and staff."""

    model_config = ConfigDict(extra="forbid")

    editors: Optional[List[Editor]] = Field(
        default_factory=list, description="Editorial board members."
    )


# --- Agent Extraction Targets ---


class BasicInfoExtraction(BaseModel):
    """Pass 1: Extracts basic journal information, scope, and identifiers."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[SourcedValue[str]] = Field(
        default=None, description="Canonical journal title."
    )
    publisher: Optional[SourcedValue[str]] = Field(
        default=None, description="Publisher name."
    )
    issn: Optional[ISSN] = Field(default=None, description="ISSN identifiers.")
    scope: Optional[SourcedValue[str]] = Field(
        default=None, description="Prose summary of focus and scope."
    )
    facts: Optional[Facts] = Field(default=None)
    metrics: Optional[Metrics] = Field(default=None, description="Journal metrics")


class PoliciesExtraction(BaseModel):
    """Pass 2: Extracts publication frequency, submission guidelines, and review policies."""

    model_config = ConfigDict(extra="forbid")

    publication_frequency: Optional[PublicationFrequency] = Field(default=None)
    submissions: Optional[SubmissionInfo] = Field(default=None)
    policies: Optional[ReviewAndPolicy] = Field(default=None)


class FeesExtraction(BaseModel):
    """Pass 3: Extracts fees, APCs, discounts, and membership information."""

    model_config = ConfigDict(extra="forbid")

    pricing: Optional[Pricing] = Field(default=None)
    membership: Optional[Membership] = Field(default=None)


class PeopleExtraction(BaseModel):
    """Pass 4: Extracts editorial board members."""

    model_config = ConfigDict(extra="forbid")

    editorial: Optional[Editorial] = Field(default=None)


# --- Final Schema ---


class JournalMetadata(
    BasicInfoExtraction, PoliciesExtraction, FeesExtraction, PeopleExtraction
):
    """
    Canonical journal metadata schema.
    Composed of multiple modular sub-schemas for targeted extraction passes.
    """

    model_config = ConfigDict(title="JournalMetadata")
