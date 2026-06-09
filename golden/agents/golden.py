import json

from pydantic_ai import Agent

from agents.base import get_model
from agents.config import PassConfig
from agents.factory import context_instructions, journal_search_tool
from golden.lib.flatten import get_slimmer_schema
from golden.models import (
    MapResult,
    VerificationResult,
)
from search import JournalSourcesDeps


# Per-agent model config
map_model = get_model("map")
reduce_model = get_model("reduce")
completeness_model = get_model("completeness")
verification_model = get_model("verification")
correction_model = get_model("correction")


# --- Agent Factory Functions ---

MAP_PROMPT = """You are an evidence collector for journal metadata extraction.

## TARGET FIELDS
The following list defines the fields you need to find evidence for. Collect evidence for every field that has information in the source text.
{schema}

## RULES
- Output a MapResult containing a list of CollectedEvidence via final_result
- Each item needs: field_hint (exact field path), quote (verbatim text), source (filename), note (brief relevance)
- The source filename is in the "[Source: filename]" marker at the top of the input
- field_hint must match a field path from the target fields above, e.g. "issn.print", "pricing.article_processing_charges.fee.value"
- Quotes must be EXACT, verbatim text from the page
- Do not paraphrase or summarize in quotes
- Only collect evidence for fields that exist in your target fields above. If the chunk contains information for fields NOT in your target fields, ignore it.
- If nothing relevant to your target fields, return MapResult with an empty evidence list
"""


def make_map_agent(pass_config: PassConfig) -> Agent[None, MapResult]:
    return Agent(
        name=f"{pass_config.name} - Map",
        model=map_model,
        output_type=MapResult,
        system_prompt=MAP_PROMPT.format(
            schema=get_slimmer_schema(pass_config.output_type),
        ),
        output_retries=2,
    )


REDUCE_PROMPT = """You assemble journal metadata from collected evidence.
You receive a list of evidence quotes with field hints and source files.

## RULES
- Map each piece of evidence to the correct schema field
- If multiple pieces of evidence support the same field, combine them
- If a field has no evidence, output null or empty list
- Do not hallucinate values not supported by evidence
- **CRITICAL: If a field has an "evidence" sub-field, populate it using the CollectedEvidence item that supports this value:**
  - `CollectedEvidence.quote` → `Evidence.quote` (verbatim text)
  - `CollectedEvidence.source` → `Evidence.source` (filename)
- Output via final_result with the complete extraction
"""


def make_reduce_agent(pass_config: PassConfig) -> Agent:
    return Agent(
        name=f"{pass_config.name} - Reduce",
        model=reduce_model,
        output_type=pass_config.output_type,
        system_prompt=REDUCE_PROMPT + pass_config.domain_guidelines,
    )


COMPLETENESS_PROMPT = """You complete a partially-filled journal metadata extraction.
You will receive the current state of the schema.

## RULES
- Only output fields that are currently null or empty in the current state
- For list fields, only emit items that are NOT already present
- For scalar fields, only populate fields whose value is null
- Fields already populated in the current state should NOT be repeated in your output
- Never re-emit or paraphrase existing data
- Use the Journal Search tool to find missing information
- Output via final_result with only the new fields found
"""


def make_completeness_agent(pass_config: PassConfig) -> Agent:
    return Agent(
        name=f"{pass_config.name} - Completeness",
        model=completeness_model,
        output_type=pass_config.output_type,
        system_prompt=COMPLETENESS_PROMPT + pass_config.domain_guidelines,
        instructions=context_instructions,
        deps_type=JournalSourcesDeps,
        tools=[journal_search_tool],
    )


VERIFICATION_PROMPT = """You verify extracted journal metadata against evidence quotes.
You receive a list of (field_path, value, evidence) triples.

## RULES
- For each triple, check if the value is correctly derived from the evidence quote
- Only check fields that have non-null evidence
- If a value is not supported by the evidence, add a FieldError
- If all values are correct, return is_correct=True with empty errors
- Be strict: the value must be explicitly stated or clearly derivable from the evidence
"""


def make_verification_agent() -> Agent[None, VerificationResult]:
    return Agent(
        name="Verification Agent",
        model=verification_model,
        output_type=VerificationResult,
        system_prompt=VERIFICATION_PROMPT,
    )


CORRECTION_PROMPT = """You correct flagged fields in a journal metadata extraction.
You receive the current extraction, a list of field errors, and access to a search tool.

## RULES
- Only output the fields listed in the errors that you were able to correct
- Do NOT repeat fields that are not being corrected
- Use the Journal Search tool to find the correct values
- If you cannot find a correct value for a flagged field, omit it from your output
- Output via final_result with only the corrected fields
"""


def make_correction_agent(pass_config: PassConfig) -> Agent:
    return Agent(
        name=f"{pass_config.name} - Correction",
        model=correction_model,
        output_type=pass_config.output_type,
        system_prompt=CORRECTION_PROMPT + pass_config.domain_guidelines,
        instructions=context_instructions,
        deps_type=JournalSourcesDeps,
        tools=[journal_search_tool],
    )

