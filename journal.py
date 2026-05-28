import os
import re
from typing import List, Optional, TypeVar, Generic, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

try:
    from .vocab import (
        Frequency,
        ReviewType,
        ArticleTypeValue,
        SupportedCurrency,
        IndexingService,
    )
except ImportError:
    from vocab import (
        Frequency,
        ReviewType,
        ArticleTypeValue,
        SupportedCurrency,
        IndexingService,
    )

T = TypeVar("T")

if TYPE_CHECKING:
    INCLUDE_EVIDENCE = True
else:
    INCLUDE_EVIDENCE = os.environ.get("WITH_EVIDENCE", "").lower() in (
        "1",
        "true",
        "yes",
    )


class Evidence(BaseModel):
    """Container for evidence supporting an extracted value."""

    model_config = ConfigDict(extra="forbid")
    quote: str = Field(
        description="Verbatim sentence or fragment from the source text."
    )
    source: str = Field(description="Source identifier (file name, URL, or URI).")


# --- Clean variants (no evidence) ---


class CleanSourcedModel(BaseModel):
    """Provide combined evidence for multiple values of a sub-object."""

    model_config = ConfigDict(extra="forbid")


class CleanSourcedValue(CleanSourcedModel, Generic[T]):
    """A value paired with its supporting evidence."""

    model_config = ConfigDict(extra="forbid")
    value: T


# --- Evidence-enabled variants ---


class EvidenceSourcedModel(CleanSourcedModel):
    """Provide combined evidence for multiple values of a sub-object."""

    evidence: Optional[Evidence] = Field(
        default=None, description="Evidence supporting this specific value."
    )


class EvidenceSourcedValue(CleanSourcedValue, Generic[T]):
    """A value paired with its supporting evidence."""

    evidence: Optional[Evidence] = Field(
        default=None, description="Evidence supporting this specific value."
    )


# --- Module-level alias ---
SourcedModel = EvidenceSourcedModel if INCLUDE_EVIDENCE else CleanSourcedModel
SourcedValue = EvidenceSourcedValue if INCLUDE_EVIDENCE else CleanSourcedValue


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
    type: Optional[SourcedValue[ReviewType]] = Field(
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


class DiamondOpenAccess(SourcedModel):
    """Journal diamond open access classification as defined within the DIAMAS and CRAFT‑OA projects."""

    model_config = ConfigDict(extra="forbid")

    scholarly_journal: Optional[bool] = Field(
        default=None,
        description=(
            "Meets scholarly journal standards. "
            "The journal should be a scholarly journal that selects papers via an explicitly "
            "described evaluation process before and/or after publication, "
            "in line with accepted practices in the relevant discipline."
        ),
    )
    community_owned: Optional[bool] = Field(
        default=None,
        description=(
            "Owned and governed by academic community. "
            "The journal title must be owned by public or not-for-profit organisations "
            "(or parts thereof) whose mission includes performing or promoting research and scholarship. "
            "These include but are not limited to research performing organisations (RPOs), "
            "research funding organisations (RFOs), organisations connected to RPOs "
            "(university libraries, university presses, faculties, and departments), "
            "research institutes, and scholarly societies. "
            "The journal should explain its ownership status on its webpage."
        ),
    )
    open_access_with_open_licenses: Optional[bool] = Field(
        default=None,
        description=(
            "Content is openly accessible under open licenses. "
            "All outputs of the journal should be Open Access and carry an open license "
            "that is included in the article-level metadata."
        ),
    )
    no_fees: Optional[bool] = Field(
        default=None,
        description=(
            "No fees for authors or readers. "
            "Publication in the journal is not contingent on the payment of fees of any kind "
            "(e.g. article processing charges or membership dues). "
            "The journal should state this as such on its webpage. "
            "Voluntary author contributions and donations are allowed, if this is not a condition for publication."
        ),
    )
    open_to_all_authors: Optional[bool] = Field(
        default=None,
        description=(
            "Accepts submissions from all eligible authors without restriction. "
            "Authorship in the journal should not be limited to any type of affiliation. "
            "Any author can submit an article that is in line with the aims and scope of the journal."
        ),
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
    linking: Optional[SourcedValue[str]] = Field(
        default=None, description="Linking ISSN (ISSN-L) in NNNN-NNNN form."
    )

    @field_validator("print", "online", "linking")
    @classmethod
    def validate_issn_format(
        cls, v: Optional[SourcedValue[str]]
    ) -> Optional[SourcedValue[str]]:
        if v is not None and v.value is not None:
            if not re.match(r"^\d{4}-\d{3}[\dX]$", v.value):
                raise ValueError("ISSN must follow the NNNN-NNNN format")
        return v


class MonetaryAmount(BaseModel):
    """Numeric Monetary value with associated currency."""

    model_config = ConfigDict(extra="forbid")
    value: int = Field(..., description="Numeric money value (rounded).")
    currency: SupportedCurrency = Field(
        ..., description="ISO 4217 currency code. USD and EUR only."
    )


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
    type: Literal["waiver", "fixed", "percent"] = Field(
        ..., description="Discount type label."
    )
    amount: Optional[MonetaryAmount] = Field(
        default=None, description="Fixed monetary discount amount if applicable."
    )
    percentage: Optional[float] = Field(
        default=None, description="Percentage discount amount if applicable."
    )
    eligibility: Optional[str] = Field(
        default=None, description="Criteria for discount eligibility as stated (quote)."
    )

    @model_validator(mode="after")
    def validate_discount_fields(self) -> "Discount":
        if self.type == "fixed" and self.amount is None:
            raise ValueError("'fixed' discount requires 'amount'")
        if self.type == "percent" and self.percentage is None:
            raise ValueError("'percent' discount requires 'percentage'")
        if self.type == "waiver" and self.amount is not None and self.percentage is not None:
            raise ValueError("'waiver' discount should not have 'amount' or 'percentage'")
        return self


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
    affiliations: List[str] = Field(
        default_factory=list,
        description="List of institutional affiliations. Institute names, not locations.",
    )

    @field_validator("affiliations", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return v


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
    acceptance_rate: Optional[SourcedValue[float]] = Field(
        default=None, description="Acceptance rate in percent (e.g., 23.5)"
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
    indexed_in: List[IndexingService] = Field(
        default_factory=list, description="Indexing services."
    )

    @field_validator("indexed_in", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return v


# --- Modular Domain Blocks ---


class SubmissionInfo(BaseModel):
    """Details regarding article submissions."""

    model_config = ConfigDict(extra="forbid")

    submission_guidelines: Optional[SourcedValue[str]] = Field(
        default=None, description="Full submission guidelines text."
    )
    article_types: List[ArticleType] = Field(
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

    article_processing_charges: List[APC] = Field(
        default_factory=list, description="Article Processing Charges."
    )
    discounts: List[Discount] = Field(
        default_factory=list, description="Waivers and discounts."
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
    """Pass 2: Extracts publication frequency, submission guidelines, and review policies, accepted languages and
    open access criteria."""

    model_config = ConfigDict(extra="forbid")

    publication_frequency: Optional[PublicationFrequency] = Field(default=None)
    submissions: Optional[SubmissionInfo] = Field(default=None)
    policies: Optional[ReviewAndPolicy] = Field(default=None)
    languages: Optional[SourcedValue[List[str]]] = Field(
        default=None,
        description="ISO 639-2/T language codes (e.g., 'eng', 'fra', 'deu').",
    )
    diamond_open_access: Optional[DiamondOpenAccess] = Field(default=None)


class FeesExtraction(BaseModel):
    """Pass 3: Extracts fees, APCs, discounts, and membership information."""

    model_config = ConfigDict(extra="forbid")

    pricing: Optional[Pricing] = Field(default=None)
    membership: Optional[Membership] = Field(default=None)


class EditorialExtraction(BaseModel):
    """Pass 4: Extracts editorial board members and staff."""

    model_config = ConfigDict(extra="forbid")

    editors: List[Editor] = Field(
        default_factory=list, description="Editorial board members."
    )


# --- Final Schema ---


class JournalMetadata(
    BasicInfoExtraction, PoliciesExtraction, FeesExtraction, EditorialExtraction
):
    """
    Canonical journal metadata schema.
    Composed of multiple modular sub-schemas for targeted extraction passes.
    """

    model_config = ConfigDict(title="JournalMetadata")
    journal_id: Optional[str] = Field(default=None, title="Journal ID")
    uri: Optional[str] = Field(
        default=None, description="Canonical journal homepage URI."
    )
