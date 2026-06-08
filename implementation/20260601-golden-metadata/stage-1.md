# Stage 1 — Evidence Types

**Goal**: Add the evidence collection types for the golden pipeline.

**Depends on**: Stage 0

## Files to Create/Modify

| File | Action |
|---|---|
| `golden/__init__.py` | **Create** — empty |
| `golden/models.py` | **Create** — `MapResult`, `CollectedEvidence`, `FieldError`, `VerificationResult` |
| `golden/parser.py` | **Create** — move from `scripts/editor_parser.py`, remove `sys.path.insert` hack |

## `golden/__init__.py`

Empty file to make `golden` a package.

## `golden/parser.py`

Copy of `scripts/editor_parser.py` with two changes:
1. Remove `import sys` and `sys.path.insert(0, ...)` (line 12-15) — no longer needed in workspace package
2. Keep `from models.journal import Editor` — resolves via workspace dependency

## `golden/models.py`

```python
from pydantic import BaseModel, Field

class MapResult(BaseModel):
    """Pydantic-ai requires a single BaseModel output_type, not list[T]."""
    evidence: list[CollectedEvidence]


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
```

## Acceptance Criteria

- `from golden.models import MapResult, CollectedEvidence, FieldError, VerificationResult` works
- `JournalSourcesDeps(index=..., journal_id="test")` requires index (already the case)
- All existing tests still pass
