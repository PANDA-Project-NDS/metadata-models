from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class JournalIdentity:
    """Unique identifier for a journal sample."""
    publisher: str
    journal: str

class CollectedEvidence(BaseModel):
    """Single piece of evidence collected from source text."""
    field_hint: str = Field(
        description="Which schema field this evidence relates to, e.g. 'issn.print', 'pricing.article_processing_charges', 'editors'"
    )
    quote: str = Field(
        description="Exact, verbatim quote from the source text"
    )
    source: str = Field(
        description="Filename of the source file"
    )
    note: str = Field(
        description="Brief explanation of relevance, e.g. 'print ISSN on masthead'"
    )


class MapResult(BaseModel):
    """Map phase wrapper of evidence list."""
    evidence: list[CollectedEvidence]


class FieldError(BaseModel):
    """A field that failed verification."""
    field_path: str = Field(
        description="Dotted field path, e.g. 'pricing.article_processing_charges[0].fee.value'"
    )
    issue: str = Field(
        description="Description of the problem, e.g. 'value 6080 not found in evidence'"
    )
    evidence: str | None = Field(
        default=None,
        description="The evidence quote that failed verification, if available"
    )


class VerificationResult(BaseModel):
    """Result of verification agent."""
    is_correct: bool = Field(
        description="True if all fields with evidence are correct"
    )
    errors: list[FieldError] = Field(
        default_factory=list,
        description="Fields that failed verification. Empty if is_correct is True."
    )
