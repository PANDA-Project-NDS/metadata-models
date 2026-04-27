import re
from typing import List, Optional, TypeVar, Generic

from pydantic import BaseModel, Field, ConfigDict, field_validator
from .vocab import *

T = TypeVar("T")


class Evidence(BaseModel):
    """Container for evidence supporting an extracted value."""

    model_config = ConfigDict(extra="forbid")
    quote: str = Field(
        description="Verbatim sentence or fragment from the source text."
    )
    source: str = Field(description="Source identifier (file name, URL, or URI).")


class SourcedModel(BaseModel):
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
    frequency: Optional[FrequencyLiteral] = Field(
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
        SourcedValue[ReviewTypeLiteral]
    ] = Field(
        default=None,
        description="Canonical review type. Empty if usual types not applicable.",
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


class Fee(SourcedModel):
    """Numeric fee amount and its associated currency."""

    model_config = ConfigDict(extra="forbid")
    amount: Optional[int] = Field(default=None, description="Numeric fee value.")
    currency: Optional[SupportedCurrencyLiteral] = Field(
        default=None, description="ISO 4217 currency code. USD and EUR only."
    )


class APC(SourcedModel):
    """Article Processing Charge details for a specific category or article type."""

    model_config = ConfigDict(extra="forbid")
    article_type: Optional[ArticleTypeLiteral] = Field(
        default=None, description="Article type this fee applies to."
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category label (e.g., Frontiers-style categories).",
    )
    fee: Optional[Fee] = Field(default=None, description="Parsed numeric fee.")


class Discount(SourcedModel):
    """Information about waivers or discounts available for publication fees."""

    model_config = ConfigDict(extra="forbid")
    type: Optional[Literal["waiver", "fixed", "percent"]] = Field(
        default=None, description="Discount type label."
    )
    amount: Optional[Fee] = Field(default=None, description="Numeric discount amount.")
    eligibility: Optional[str] = Field(
        default=None, description="Criteria for discount eligibility as stated."
    )


class ArticleType(SourcedModel):
    """Definition of an article type supported by the journal."""

    model_config = ConfigDict(extra="forbid")
    type: ArticleTypeLiteral = Field(..., description="The name of the article type.")
    notes: Optional[str] = Field(
        default=None, description="Optional supplementary notes."
    )


class Editor(SourcedModel):
    """Member of the journal's editorial board."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Full name of the editor.")
    role: Optional[EditorRoleLiteral] = Field(
        default=None, description="Role or title (e.g., 'Editor-in-Chief')."
    )
    affiliations: Optional[List[str]] = Field(
        default_factory=list, description="List of institutional affiliations."
    )


class AdditionalInformation(BaseModel):
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


class ImpactMetrics(BaseModel):
    """Quantifiable journal metrics."""

    model_config = ConfigDict(extra="forbid")

    cite_score: Optional[SourcedValue[float]] = Field(
        default=None, description="CiteScore metric."
    )
    impact_factor: Optional[SourcedValue[float]] = Field(
        default=None, description="Impact Factor metric."
    )


class Facts(BaseModel):
    """Brief metadata summary, often found in 'Journal Facts' sidebars."""

    model_config = ConfigDict(extra="forbid")

    short_name: Optional[SourcedValue[str]] = Field(
        default=None, description="Shortened journal name."
    )
    abbreviation: Optional[SourcedValue[str]] = Field(
        default=None, description="Journal abbreviation."
    )
    indexed_in: Optional[SourcedValue[str]] = Field(
        default=None, description="Indexing service."
    )


# --- Modular Domain Blocks ---


class JournalIdentity(BaseModel):
    """Core identification for the journal."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[SourcedValue[str]] = Field(
        default=None, description="Canonical journal title."
    )
    publisher: Optional[SourcedValue[str]] = Field(
        default=None, description="Publisher name."
    )
    issn: Optional[ISSN] = Field(default=None, description="ISSN identifiers.")


class JournalScope(BaseModel):
    """Information about the journal's focus and sections."""

    model_config = ConfigDict(extra="forbid")

    description: Optional[SourcedValue[str]] = Field(
        default=None, description="Prose summary of focus and scope."
    )
    journal_sections: Optional[List[SourcedValue[str]]] = Field(
        default_factory=list, description="Sections within the journal."
    )


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
    additional_information: Optional[AdditionalInformation] = Field(
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

    identity: Optional[JournalIdentity] = Field(default=None)
    scope: Optional[JournalScope] = Field(default=None)
    facts: Optional[Facts] = Field(default=None)
    metrics: Optional[ImpactMetrics] = Field(default=None, description="Impact metrics")


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
