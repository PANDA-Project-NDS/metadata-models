import json
import os
import re
from datetime import date
from typing import TYPE_CHECKING, Any, Generic, List, Optional, TypeAlias, TypeVar

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

try:
    from .vocab import (
        ArticleTypeValue,
        DiscountType,
        EligibilityMechanism,
        Frequency,
        IndexingService,
        LicenseType,
        MembershipType,
        ReviewType,
        SupportedCurrency,
        publisher_examples,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from vocab import (
        ArticleTypeValue,
        DiscountType,
        EligibilityMechanism,
        Frequency,
        IndexingService,
        LicenseType,
        MembershipType,
        ReviewType,
        SupportedCurrency,
        publisher_examples,
    )

T = TypeVar("T")

load_dotenv()

if TYPE_CHECKING:
    INCLUDE_EVIDENCE = True
else:
    INCLUDE_EVIDENCE = os.environ.get("WITH_EVIDENCE", "").lower() in (
        "1",
        "true",
        "yes",
    )


class _JsonStringParser:
    """Mixin: auto-parses JSON-string input into dict before Pydantic validation.

    Defends against LLM double-encoding field values as JSON strings instead of
    objects (e.g. ``"title": "{\\"value\\": \\"X\\"}"`` instead of
    ``"title": {"value": "X"}``).
    """

    @model_validator(mode="before")
    @classmethod
    def _auto_parse_json_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return data


class JsonHandlingBaseModel(_JsonStringParser, BaseModel):
    pass


class Evidence(BaseModel):
    """Container for evidence supporting an extracted value."""

    model_config = ConfigDict(extra="forbid")
    quote: str = Field(
        description="Verbatim sentence or fragment from the source text."
    )
    source: str = Field(description="Source identifier (file name, URL, or URI).")


# --- Clean variants (no evidence) ---


class CleanSourcedModel(JsonHandlingBaseModel):
    """Provide combined evidence for multiple values of a sub-object."""

    model_config = ConfigDict(extra="forbid")


class CleanSourcedValue(CleanSourcedModel, Generic[T]):
    """A value paired with its supporting evidence."""

    model_config = ConfigDict(extra="forbid")
    value: T


# TODO: maybe wrap/flatten .value on de-/serialization
#    @model_validator(mode="before")
#    @classmethod
#    def _wrap_plain_value(cls, data):
#        if not isinstance(data, dict):
#            return {"value": data}
#        return data
#    @model_serializer(mode="wrap")
#    def _serialize(self, handler, info):
#        if len(self.model_fields) == 1 and "value" in self.model_fields:
#            return self.value
#        return handler(self)


# --- Evidence-enabled variants ---


class EvidenceSourcedModel(CleanSourcedModel):
    """Provide combined evidence for multiple values of a sub-object."""

    evidence: Evidence = Field(
        ..., description="Evidence supporting this specific value."
    )


class EvidenceSourcedValue(CleanSourcedValue[T]):
    """A value paired with its supporting evidence."""

    evidence: Evidence = Field(
        ..., description="Evidence supporting this specific value."
    )


# --- Module-level alias ---
if INCLUDE_EVIDENCE:
    SourcedModel: type[CleanSourcedModel] = EvidenceSourcedModel
    SourcedValue: TypeAlias = EvidenceSourcedValue[T]
else:
    SourcedModel: type[CleanSourcedModel] = CleanSourcedModel
    SourcedValue: TypeAlias = CleanSourcedValue[T]


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


class ReviewProcess(JsonHandlingBaseModel):
    """Reviewing process used by the journal."""

    model_config = ConfigDict(extra="forbid")

    type: Optional[SourcedValue[ReviewType]] = Field(
        default=None,
        description="Type of the review process.",
    )
    description: Optional[SourcedValue[str]] = Field(
        default=None,
        description="Summary of the review workflow.",
    )


class Membership(JsonHandlingBaseModel):
    """Information about society or institutional membership models."""

    model_config = ConfigDict(extra="forbid")

    type: Optional[SourcedValue[MembershipType]] = Field(
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


class ISSN(JsonHandlingBaseModel):
    """International Standard Serial Number identifiers in NNNN-NNNN form."""

    model_config = ConfigDict(extra="forbid")

    print: Optional[SourcedValue[str]] = Field(
        default=None, description="Print ISSN (ISSN-P)"
    )
    online: Optional[SourcedValue[str]] = Field(
        default=None, description="Online ISSN (ISSN-E / eISSN)"
    )
    linking: Optional[SourcedValue[str]] = Field(
        default=None, description="Linking ISSN (ISSN-L) if stated explicitly"
    )

    @field_validator("print", "online", "linking", mode="before")
    @classmethod
    def fix_issn_dash(cls, v):
        """Add dash to ISSN if given without — handles dict (JSON) and CleanSourcedValue (Python) input."""
        if v is None:
            return v
        raw = None
        if isinstance(v, dict):
            raw = v.get("value")
        elif isinstance(v, CleanSourcedValue):
            raw = v.value
        if isinstance(raw, str):
            cleaned = raw.replace(" ", "")
            if len(cleaned) == 8:
                fixed = f"{cleaned[:4]}-{cleaned[4:]}"
                if isinstance(v, dict):
                    v["value"] = fixed
                else:
                    v.value = fixed
        return v

    @field_validator("print", "online", "linking")
    @classmethod
    def validate_issn_format(
        cls, v: Optional[SourcedValue[str]]
    ) -> Optional[SourcedValue[str]]:
        if v is not None and v.value is not None:
            if not re.match(r"^\d{4}-\d{3}[\dX]$", v.value):
                raise ValueError("ISSN must follow the NNNN-NNNN format")
        return v

    @model_validator(mode="after")
    def check_issn_consistency(self) -> "ISSN":
        p = self.print.value if self.print else None
        o = self.online.value if self.online else None
        l = self.linking.value if self.linking else None

        if p is not None and o is not None and p == o:
            raise ValueError("print and online ISSN must not be identical")
        if l is not None and l != p and l != o:
            raise ValueError(f"linking ISSN {l} must match print ({p}) or online ({o})")
        return self


class MonetaryAmount(JsonHandlingBaseModel):
    """Numeric Monetary value with associated currency."""

    model_config = ConfigDict(extra="forbid")
    value: int = Field(..., description="Numeric money value (rounded).")
    currency: SupportedCurrency = Field(..., description="ISO 4217 currency code.")

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency_case(cls, v):
        """Normalize currency code to uppercase (e.g., 'usd' -> 'USD')."""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("value", mode="before")
    @classmethod
    def coerce_to_int(cls, v):
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            match = re.search(r"\d+", v.replace(",", ""))
            if match:
                return int(match.group())
        return v

    def __hash__(self):
        return hash((self.value, self.currency))


class APC(SourcedModel):
    """Article Processing Charge details for a specific category or article type."""

    model_config = ConfigDict(extra="forbid")
    article_type: Optional[ArticleTypeValue] = Field(
        default=None, description="Article type this fee applies to."
    )
    category: Optional[str] = Field(
        default=None,
        description="Category label instead of or additional to `article_type`.",
    )
    per_page: bool = Field(default=False, description="APC charged per page.")
    per_figure: bool = Field(default=False, description="APC charged per figure.")
    license_name: Optional[str] = Field(
        default=None, description="Stated license name (only if license-related)."
    )
    license_type: Optional[LicenseType] = Field(
        default=None, description="Machine-readable license (only if license-related)."
    )
    license_related: bool = Field(
        default=False, description="True if APC is tied to a specific license."
    )
    fee: List[MonetaryAmount] = Field(
        default_factory=list, description="Price of APC. One per currency."
    )

    @field_validator("fee", mode="before")
    @classmethod
    def coerce_fee_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # Single MonetaryAmount object or dict — wrap in list.
        if isinstance(v, (MonetaryAmount, dict)):
            return [v]
        return v

    def __hash__(self):
        return hash(
            (
                self.article_type,
                self.category,
                self.per_page,
                self.per_figure,
                self.license_related,
            )
        )

    def __eq__(self, other):
        if isinstance(other, APC):
            return (
                self.article_type,
                self.category,
                self.per_page,
                self.per_figure,
                self.license_related,
            ) == (
                other.article_type,
                other.category,
                other.per_page,
                other.per_figure,
                self.license_related,
            )
        return super().__eq__(other)

    @model_validator(mode="after")
    def dedup_sort_fees(self) -> "APC":
        self.fee = sorted(dict.fromkeys(self.fee), key=lambda f: f.currency)
        return self


class Discount(JsonHandlingBaseModel):
    """Information about waivers or discounts available for publication fees."""

    model_config = ConfigDict(extra="forbid")
    type: DiscountType = Field(..., description="Discount type label.")
    amount: Optional[MonetaryAmount] = Field(
        default=None, description="Fixed monetary discount amount if applicable."
    )
    percentage: Optional[float] = Field(
        default=None,
        gt=0,
        le=100,
        description="Percentage discount amount if applicable.",
    )
    eligibility: str = Field(
        ...,
        min_length=1,
        description="Criteria for discount eligibility as stated (quote).",
    )
    eligibility_mechanism: EligibilityMechanism = Field(
        default="unconditional",
        description="Primary structural eligibility gate for this discount, independent "
        "of time-limits or article-type restrictions (captured in dedicated fields). "
        "'institutional_agreement' when the author's institution has a transformative "
        "or partnership agreement with the publisher (e.g., DEAL, CRKN, WOAA). "
        "'individual_membership' when the author's personal society or organization "
        "membership is required. "
        "'country_based' when eligibility depends on the country of the author's "
        "institution (e.g., Research4Life, World Bank income classifications). "
        "'unconditional' when the discount has no institutional/membership/country gate. "
        "'other' when a gate exists but doesn't fit the categories above (e.g., hardship "
        "application, editorial invitation).",
    )
    eligible_article_types: list[str] = Field(
        default_factory=list,
        description=(
            "Article types eligible for this discount. Empty means the "
            "discount applies to all article types with no restriction; a "
            "non-empty list means only these specific types qualify."
        ),
        examples=[["Editorial", "Review", "Commentary"], ["Opinion"], []],
    )
    time_limited: bool = Field(
        default=False,
        description="Discount is available only for a limited time period.",
    )
    expires_after: Optional[date] = Field(
        default=None,
        description="Expiry date after which the discount is no longer available. "
        "None when the end date is unspecified (e.g., 'until further notice').",
    )

    @field_validator("eligible_article_types", mode="before")
    @classmethod
    def dedup_article_types(cls, v):
        """Deduplicate article types case-insensitively, preserving
        the casing of the first occurrence."""
        if isinstance(v, list):
            seen: list[str] = []
            seen_lower: set[str] = set()
            for item in v:
                key = item.strip().lower() if isinstance(item, str) else item
                if key not in seen_lower:
                    seen_lower.add(key)
                    seen.append(item.strip() if isinstance(item, str) else item)
            return seen
        return v

    @model_validator(mode="after")
    def validate_discount_fields(self) -> "Discount":
        """Enforce discount type invariants.

        - 'fixed' requires amount, forbids percentage.
        - 'percent' requires percentage, forbids amount.
        - 'waiver' forbids both amount and percentage.
        - Converts 100% percent to waiver.
        """

        if self.type == "fixed" and self.amount is None:
            raise ValueError("'fixed' discount requires 'amount'")
        if self.type == "fixed" and self.percentage is not None:
            raise ValueError("'fixed' discount should not have 'percentage'")
        if self.type == "percent" and self.percentage is None:
            raise ValueError("'percent' discount requires 'percentage'")
        if self.type == "percent" and self.amount is not None:
            raise ValueError("'percent' discount should not have 'amount'")
        if self.type == "waiver" and (
            self.amount is not None or self.percentage is not None
        ):
            raise ValueError(
                "'waiver' discount should not have 'amount' or 'percentage'"
            )

        # convert 100 percent to waiver
        if self.type == "percent" and self.percentage == 100.0:
            self.percentage = None
            self.type = "waiver"

        return self

    @model_validator(mode="after")
    def check_expiry_consistency(self) -> "Discount":
        """Ensure time_limited and expires_after are consistent.

        A permanent discount (time_limited=False) must not carry
        an expiry date.
        """
        if not self.time_limited and self.expires_after is not None:
            raise ValueError(
                "expires_after must be None when time_limited is False "
                "(a permanent discount cannot have an expiry date)."
            )
        return self


class ArticleType(SourcedModel):
    """Definition of an article type supported by the journal."""

    model_config = ConfigDict(extra="forbid")
    type: ArticleTypeValue = Field(..., description="The name of the article type.")
    notes: Optional[str] = Field(
        default=None, description="Optional supplementary notes."
    )

    def __hash__(self):
        return hash(self.type)

    def __eq__(self, other):
        if isinstance(other, ArticleType):
            return self.type == other.type
        return super().__eq__(other)


class Editor(JsonHandlingBaseModel):
    """Member of the journal's editorial board."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Full name of the editor.")
    role: Optional[str] = Field(
        default=None, description="Role or title as stated (e.g., 'Editor-in-Chief')."
    )
    affiliations: set[str] = Field(
        default_factory=set,
        description="Set of institutional affiliations. Institute names, not locations.",
    )

    @field_validator("affiliations", mode="before")
    @classmethod
    def coerce_to_set(cls, v):
        if v is None or v == "":
            return set()
        if isinstance(v, str):
            return {v}
        return v


class PublisherPolicies(JsonHandlingBaseModel):
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


class Metrics(JsonHandlingBaseModel):
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


class Facts(JsonHandlingBaseModel):
    """Brief metadata summary, often found in 'Journal Facts' sidebars."""

    model_config = ConfigDict(extra="forbid")

    short_name: Optional[SourcedValue[str]] = Field(
        default=None, description="Shortened journal name."
    )
    abbreviation: Optional[SourcedValue[str]] = Field(
        default=None, description="Journal abbreviation."
    )
    indexed_in: Optional[SourcedValue[set[IndexingService]]] = Field(
        default=None, description="Indexing services."
    )


# --- Modular Domain Blocks ---


class SubmissionInfo(JsonHandlingBaseModel):
    """Details regarding article submissions."""

    model_config = ConfigDict(extra="forbid")

    submission_guidelines: Optional[str] = Field(
        default=None, description="Full submission guidelines text."
    )
    article_types: set[ArticleType] = Field(
        default_factory=set, description="Set of supported article types."
    )
    languages: Optional[SourcedValue[set[str]]] = Field(
        default=None,
        description="Accepted languages for article submissions. ISO 639-2/T language codes (e.g., 'eng', 'fra', 'deu').",
    )


class PublicationPolicy(JsonHandlingBaseModel):
    """Policies related to the publication process in the journal."""

    model_config = ConfigDict(extra="forbid")

    review_process: Optional[ReviewProcess] = Field(
        default=None, description="Review process details."
    )
    publisher_policies: Optional[PublisherPolicies] = Field(
        default=None, description="Publisher-specific policy details."
    )
    # TODO
    # licences: Set[str] = Field(default=set(), description="licenses under which the journal publishes its contents.")


class Pricing(JsonHandlingBaseModel):
    """Article processing charges and discounts."""

    model_config = ConfigDict(extra="forbid")

    article_processing_charges: List[APC] = Field(
        default_factory=list, description="Article Processing Charges."
    )
    discounts: List[Discount] = Field(
        default_factory=list, description="Waivers and discounts."
    )

    @model_validator(mode="after")
    def dedup_apcs(self) -> "Pricing":
        """Deduplicate APCs by identity, merging fees for matches."""
        seen: dict[APC, None] = {}  # dict to preserve insertion order
        for apc in self.article_processing_charges:
            if apc in seen:
                existing = next(e for e in seen if e == apc)
                existing_key_vals = {(f.value, f.currency) for f in existing.fee}
                for f in apc.fee:
                    if (f.value, f.currency) not in existing_key_vals:
                        existing.fee.append(f)
                existing.fee = sorted(existing.fee, key=lambda f: f.currency)
            else:
                seen[apc] = None
        self.article_processing_charges = list(seen)
        return self


# --- Agent Extraction Targets ---


class BasicInfoExtraction(JsonHandlingBaseModel):
    """Pass 1: Extracts basic journal information, scope, and identifiers."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[SourcedValue[str]] = Field(
        default=None, description="Canonical journal title."
    )
    publisher: Optional[SourcedValue[str]] = Field(
        default=None,
        examples=publisher_examples,
        description="Publisher name.",
    )
    issn: Optional[ISSN] = Field(default=None, description="ISSN identifiers.")
    scope: Optional[SourcedValue[str]] = Field(
        default=None, description="Prose summary of focus and scope."
    )
    facts: Optional[Facts] = Field(default=None)
    metrics: Optional[Metrics] = Field(default=None, description="Journal metrics")


class PoliciesExtraction(JsonHandlingBaseModel):
    """Pass 2: Extracts publication frequency, submission guidelines, and publication policies, accepted languages and
    open access criteria."""

    model_config = ConfigDict(extra="forbid")

    publication_frequency: Optional[PublicationFrequency] = Field(default=None)
    submissions: Optional[SubmissionInfo] = Field(default=None)
    publication_policy: Optional[PublicationPolicy] = Field(default=None)
    diamond_open_access: Optional[DiamondOpenAccess] = Field(default=None)


class FeesExtraction(JsonHandlingBaseModel):
    """Pass 3: Extracts fees, APCs, discounts, and membership information."""

    model_config = ConfigDict(extra="forbid")

    pricing: Optional[Pricing] = Field(default=None)
    membership: Optional[Membership] = Field(default=None)


class EditorialExtraction(JsonHandlingBaseModel):
    """Pass 4: Extracts editorial board members and staff."""

    model_config = ConfigDict(extra="forbid")

    editors: List[Editor] = Field(
        default_factory=list, description="Editorial board members."
    )


type JournalPass = (
    BasicInfoExtraction | PoliciesExtraction | FeesExtraction | EditorialExtraction
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
    journal_id: Optional[str] = Field(default=None, title="Internal Journal ID")
    publisher_id: Optional[str] = Field(default=None, title="Internal Publisher ID")
    uri: Optional[str] = Field(
        default=None, description="Canonical journal homepage URI."
    )
