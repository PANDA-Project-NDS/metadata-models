# Stage 4 — Phase 2: Two-Judge Evaluation

**Goal**: Replace the single `judge_input_output` Phase 2 with two separate judges — a **Coverage Judge** that compares extraction completeness against `journal-samples/coverage.md`, and an **Evidence Judge** that validates extracted values against source evidence quotes.

**Depends on**: Stage 1.5

## Rationale

| Concern | Old approach | New approach |
|---|---|---|
| Completeness | Judge inferred expectations from rubric | Coverage Judge knows publisher-specific status (PRESENT/PARTIAL/MISSING/N/A) |
| Correctness | Single pass/fail verdict applied to all fields | Evidence Judge produces per-field verdicts |
| Missing fields with evidence | Not detected | `missing_with_evidence` flag catches assembly gaps |
| Coverage data | Unused | `coverage.md` fed directly as judge context — no parsing needed |
| `pydantic-evals` dependency | Required | Removed — replaced with custom `pydantic-ai` agents |

## Data Flow

```
Phase 1 output: JournalMetadata (with evidence)
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                  ▼
     coverage.md        evidence triples     extracted JSON
            │                 │                  │
            ▼                 ▼                  ▼
   ┌────────────┐    ┌───────────────┐
   │ Coverage   │    │ Evidence      │
   │ Judge      │    │ Judge         │
   │            │    │               │
   │ complete?  │    │ correct?      │
   │ per schema │    │ missing_with  │
   │ field      │    │ _evidence?    │
   └──────┬─────┘    └───────┬───────┘
          │                  │
          └──────┬───────────┘
                 ▼
          merge verdicts
                 │
                 ▼
       write golden JSON + grading sidecar
```

## Files to Create

| File | Action |
|---|---|
| `golden/agents/judge.py` | **Create** — CoverageJudge + EvidenceJudge agents, verdict models, merge function |

## Verdict Output Types

```python
from pydantic import BaseModel, Field


class CoverageFieldVerdict(BaseModel):
    """Per-field verdict from the Coverage Judge."""
    field_path: str = Field(
        description="Dotted field path, e.g. 'issn.print' or 'pricing.article_processing_charges'"
    )
    coverage_pass: bool = Field(
        description="True if null/non-null state matches coverage.md expectation for this publisher"
    )
    coverage_issue: str | None = Field(
        default=None,
        description="Human-readable explanation if coverage_pass is False, e.g. "
                    "'ISSN is null but coverage says PRESENT for ACS'"
    )


class CoverageJudgeResult(BaseModel):
    """Output of the Coverage Judge — one verdict per top-level schema field."""
    verdicts: list[CoverageFieldVerdict]
    summary: str = Field(
        description="One-line summary, e.g. '37/40 fields match coverage expectations'"
    )


class EvidenceVerdict(BaseModel):
    """Per-field verdict from the Evidence Judge."""
    field_path: str
    evidence_pass: bool = Field(
        description="True if the extracted value is supported by the evidence quote"
    )
    missing_with_evidence: bool = Field(
        default=False,
        description="True if the field is null/empty but evidence triples with matching "
                    "field_hint exist — indicates an assembly gap in Phase 1"
    )
    reason: str | None = Field(
        default=None,
        description="Free-text explanation, e.g. 'value 6080 not found in evidence quote' "
                    "or 'evidence says CHF 1800 but field is null'"
    )


class EvidenceJudgeResult(BaseModel):
    """Output of the Evidence Judge — one verdict per evidence triple."""
    verdicts: list[EvidenceVerdict]
```

## Coverage Judge Agent

### Input

The Coverage Judge receives two pieces of data in a single prompt:

1. **Publisher coverage section** from `coverage.md` — raw markdown for the relevant publisher (e.g. the ACS table, notes, and score)
2. **Extracted metadata JSON** — the full `JournalMetadata.model_dump(mode="json")` (with evidence included)

No evidence triples. No schema descriptions. The judge reads the coverage table directly.

### System Prompt

```
COVERAGE_JUDGE_PROMPT = """You verify that extracted metadata matches what is
expected for this publisher, based on the coverage report.

The coverage report marks each metadata type per publisher:

  🟢 PRESENT          → should have a non-null value in the extracted output
  🟡 PARTIAL          → may have partial extraction; null is acceptable but
                        flag it if the notes say all journals have it
  🔴 MISSING          → genuinely absent from publisher pages; null is correct
  🟣 NOT APPLICABLE   → category does not apply; null is correct
  🟤 EXTRACTION FAILURE → metadata exists on the page but the scraper couldn't
                        capture it; null is expected, no penalty

RULES:
- Evaluate every top-level field present in the extracted metadata, including null values
- A field passes coverage if its null/non-null state matches the coverage status
  and the notes in the coverage report
- Use the notes column to understand nuances
  (e.g. "pISSN genuinely absent — online-only journal" means null is fine)
- Do NOT check correctness of values — only whether values exist where expected
- Output one CoverageFieldVerdict per top-level schema field

Output via final_result with the complete evaluation.
"""
```

### Agent Factory

```python
def make_coverage_judge() -> Agent[None, CoverageJudgeResult]:
    return Agent(
        name="Coverage Judge",
        model=judge_model,
        output_type=CoverageJudgeResult,
        system_prompt=COVERAGE_JUDGE_PROMPT,
    )
```

## Evidence Judge Agent

### Input

The Evidence Judge receives:

1. **Evidence triples** — `list[{"field_path": str, "value": any, "evidence": str | None}]` from `flatten_metadata()`. Every leaf field in the schema produces one triple. Evidence is `None` for fields without quotes (e.g. parsed editors).
2. **Extracted metadata JSON** — the full `JournalMetadata.model_dump(mode="json")` (with evidence)

### System Prompt

```
EVIDENCE_JUDGE_PROMPT = """You verify that extracted values are supported by
their evidence quotes, and that no field with clear evidence was left null.

You receive:
1. Evidence triples: (field_path, value, evidence_quote) for every leaf field
2. The full extracted metadata JSON

RULES:
- For each triple where evidence is not None:
  • Is the extracted value correctly derived from the evidence quote?
  • Be strict: the value must be explicitly stated or clearly derivable
  • Vague, ambiguous, or contradictory evidence = fail
- For triples where value IS null but evidence IS not None:
  • Set missing_with_evidence = True
  • This catches cases where the collector found a quote but the
    assembler/completeness/correction phase failed to populate the field
- For triples where evidence is None: skip — no evidence to judge against
- Only output EvidenceVerdict for triples that have non-null evidence. Omit all others.

Output one EvidenceVerdict per evidence triple that has non-null evidence.
"""
```

### Agent Factory

```python
def make_evidence_judge() -> Agent[None, EvidenceJudgeResult]:
    return Agent(
        name="Evidence Judge",
        model=judge_model,
        output_type=EvidenceJudgeResult,
        system_prompt=EVIDENCE_JUDGE_PROMPT,
    )
```

## Model Config

Both judges use the same model, controlled by `JUDGE_MODEL` env var with fallback to `OPENAI_MODEL`:

```python
from agents.base import get_model

judge_model = get_model("judge")  # JUDGE_MODEL env var
```

## Grading Merge

```python
def merge_verdicts(
    coverage_result: CoverageJudgeResult,
    evidence_result: EvidenceJudgeResult,
    metadata,
    journal_id: str,
) -> dict:
    """Merge both judges into a single grading sidecar dict.

    Coverage verdicts seed the grading (one per top-level schema field).
    Evidence verdicts are aggregated to top-level fields: any leaf fail
    makes the parent fail.
    """
    grading = {}

    # Seed with coverage verdicts (top-level fields)
    for v in coverage_result.verdicts:
        grading[v.field_path] = {
            "coverage_pass": v.coverage_pass,
            "coverage_issue": v.coverage_issue,
            "evidence_pass": None,
            "missing_with_evidence": False,
            "reason": None,
        }

    # Aggregate evidence verdicts to top-level fields
    for v in evidence_result.verdicts:
        top = v.field_path.split(".")[0].split("[")[0]
        if top not in grading:
            grading[top] = {
                "coverage_pass": None,
                "coverage_issue": None,
                "evidence_pass": None,
                "missing_with_evidence": False,
                "reason": None,
            }
        entry = grading[top]
        # Worst-case wins: any fail makes the top-level fail
        if entry["evidence_pass"] is False:
            continue  # already failed
        if entry["evidence_pass"] is None or v.evidence_pass:
            entry["evidence_pass"] = v.evidence_pass
        if v.missing_with_evidence:
            entry["missing_with_evidence"] = True
            entry["reason"] = v.reason

    # Aggregate counts
    cov_pass = sum(1 for g in grading.values() if g.get("coverage_pass") is True)
    cov_fail = sum(1 for g in grading.values() if g.get("coverage_pass") is False)
    ev_pass = sum(1 for g in grading.values() if g.get("evidence_pass") is True)
    ev_fail = sum(1 for g in grading.values() if g.get("evidence_pass") is False)
    ev_skip = sum(1 for g in grading.values() if g.get("evidence_pass") is None)
    missing = sum(1 for g in grading.values() if g.get("missing_with_evidence"))

    return {
        "journal_id": journal_id,
        "total_fields": len(grading),
        "coverage_pass": cov_pass,
        "coverage_fail": cov_fail,
        "evidence_pass": ev_pass,
        "evidence_fail": ev_fail,
        "evidence_skipped": ev_skip,
        "missing_with_evidence": missing,
        "fields": grading,
    }
```

## Orchestration (called from `main()` in stage 3)

```python
async def judge_journal(
    metadata: JournalMetadata,
    publisher: str,
    coverage_text: str,
) -> dict:
    """Run both judges and return merged grading sidecar."""

    # Phase 2a: Coverage Judge
    coverage_judge = make_coverage_judge()
    coverage_input = (
        f"## Publisher Coverage ({publisher})\n\n"
        f"{coverage_text}\n\n"
        f"## Extracted Metadata\n"
        f"{json.dumps(metadata.model_dump(mode='json'), indent=2)}"
    )
    coverage_result = (await coverage_judge.run(coverage_input)).output

    # Phase 2b: Evidence Judge
    fields = flatten_metadata(metadata)
    evidence_triples = [
        {"field_path": p, "value": v, "evidence": e}
        for p, v, e in fields
    ]
    evidence_judge = make_evidence_judge()
    evidence_input = (
        f"## Evidence Triples\n"
        f"{json.dumps(evidence_triples, indent=2, default=str)}\n\n"
        f"## Extracted Metadata\n"
        f"{json.dumps(metadata.model_dump(mode='json'), indent=2)}"
    )
    evidence_result = (await evidence_judge.run(evidence_input)).output

    return merge_verdicts(
        coverage_result, evidence_result, metadata, metadata.journal_id
    )
```

## Coverage Text Loading

The coverage document is loaded once at startup and cached as a dict keyed by publisher:

```python
import re
from pathlib import Path

COVERAGE_PATH = Path("journal-samples/coverage.md")

# Map coverage.md heading names → directory names used by discover_journals
_COVERAGE_KEY_MAP = {
    "springer link": "springer_link",
    "springer nature": "springer_nature",
    "taylor & francis": "tandf",
}


def load_coverage_sections() -> dict[str, str]:
    """Split coverage.md by publisher heading into {dir_name: raw_markdown}."""
    text = COVERAGE_PATH.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_pub = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        m = re.match(r"^### (.+?) \(\d+ journals: .+\)", line)
        if m:
            if current_pub and current_lines:
                sections[current_pub] = "\n".join(current_lines)
            current_pub = m.group(1).lower()
            current_lines = [line]
        elif current_pub:
            # Stop at next top-level heading or end
            if line.startswith("## ") and not line.startswith("### "):
                sections[current_pub] = "\n".join(current_lines)
                current_pub = None
                current_lines = []
            else:
                current_lines.append(line)

    if current_pub and current_lines:
        sections[current_pub] = "\n".join(current_lines)

    # Remap heading keys → directory names so discover_journals lookups succeed
    remapped: dict[str, str] = {}
    for key, text in sections.items():
        remapped[_COVERAGE_KEY_MAP.get(key, key)] = text

    return remapped
```

## Cost

| Judge | Calls per journal | Total (38 journals) |
|---|---|---|
| Coverage Judge | 1 | 38 |
| Evidence Judge | 1 | 38 |
| **Total** | **2** | **76** |

Each call is a short context (~2K tokens for coverage + ~5K for metadata = ~7K per call). No `pydantic-evals` dependency.

## Acceptance Criteria

- `from golden.agents.judge import make_coverage_judge, make_evidence_judge, merge_verdicts` works
- Coverage Judge correctly passes ACS journals with non-null ISSN, fails them with null ISSN
- Coverage Judge correctly passes Copernicus with null metrics (N/A), flags Wiley AEM with null publication frequency (PRESENT)
- Evidence Judge correctly flags null fields that have evidence triples (`missing_with_evidence=True`)
- Evidence Judge skips fields without evidence (e.g. parsed editors)
- `load_coverage_sections()` returns keys matching directory names: `acs`, `copernicus`, `elsevier`, `emerald`, `frontiers`, `ieee`, `mdpi`, `sage`, `springer_link`, `springer_nature`, `tandf`, `wiley`
- Merged grading sidecar has `coverage_pass`, `evidence_pass`, `missing_with_evidence` per field
- All existing tests still pass
