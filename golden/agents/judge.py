import json
import re
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.base import get_model
from golden.lib.flatten import flatten_metadata


judge_model = get_model("judge")


# --- Verdict Output Types ---

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
        description="Human-readable explanation if coverage_pass is False"
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
        description="True if the field is null/empty but evidence triples with matching field_hint exist"
    )
    reason: str | None = Field(
        default=None,
        description="Free-text explanation"
    )


class EvidenceJudgeResult(BaseModel):
    """Output of the Evidence Judge — one verdict per evidence triple."""
    verdicts: list[EvidenceVerdict]


# --- Prompts ---

COVERAGE_JUDGE_PROMPT = """You verify that extracted metadata matches what is
expected for this publisher, based on the coverage report.

The coverage report marks each metadata type per publisher:

  PRESENT          → should have a non-null value in the extracted output
  PARTIAL          → may have partial extraction; null is acceptable but
                     flag it if the notes say all journals have it
  MISSING          → genuinely absent from publisher pages; null is correct
  NOT APPLICABLE   → category does not apply; null is correct
  EXTRACTION FAILURE → metadata exists on the page but the scraper couldn't
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


# --- Agent Factories ---

def make_coverage_judge() -> Agent[None, CoverageJudgeResult]:
    return Agent(
        name="Coverage Judge",
        model=judge_model,
        output_type=CoverageJudgeResult,
        system_prompt=COVERAGE_JUDGE_PROMPT,
        instrument=True,
    )


def make_evidence_judge() -> Agent[None, EvidenceJudgeResult]:
    return Agent(
        name="Evidence Judge",
        model=judge_model,
        output_type=EvidenceJudgeResult,
        system_prompt=EVIDENCE_JUDGE_PROMPT,
        instrument=True,
    )


# --- Grading Merge ---

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


# --- Orchestration ---

async def judge_journal(
    metadata,
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
    fields = list(flatten_metadata(metadata))
    evidence_triples = [
        {"field_path": f.path, "value": f.value, "evidence": f.evidence}
        for f in fields
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


# --- Coverage Text Loading ---

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