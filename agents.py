import os

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from models.journal import (
    BasicInfoExtraction,
    PoliciesExtraction,
    FeesExtraction,
    PeopleExtraction,
)
from search import JournalSearchDeps, journal_search

logfire.configure()
logfire.instrument_pydantic_ai()

load_dotenv()

# --- Extraction Queries ---

EXTRACTION_QUERIES = {
    "basic_info": [
        "Journal title, publisher, about this journal, mission, scope, sections",
        "ISSN, print ISSN, online ISSN, indexed in, abstracting and indexing databases",
        "Impact factor, journal metrics, citation score, cite score",
    ],
    "policies_submissions": [
        "Publication frequency, issues per year, submission guidelines, author instructions, article types accepted",
        "Peer review process, blind review, open access policy statement, copyright, quality assurance",
    ],
    "fees_membership": [
        "Article Processing Charge, APC, publication fees, cost, waivers, discounts, society membership, institutional membership"
    ],
    "people_metrics": [
        "Editorial board, Editor in Chief, managing editor, editorial team"
    ],
}

# --- Prompts & Agents ---
# Common system prompt base
BASE_SYSTEM_PROMPT = """
You are an expert data extraction assistant. Your task is to extract highly accurate, structured metadata for an academic journal based ONLY on the provided context chunks.

CRITICAL RULES:
1. NO HALLUCINATION: If a piece of information is not explicitly stated in the context, you MUST output null or an empty list. Do not guess or infer.
2. VERBATIM EVIDENCE: For every extracted value, you must provide the exact, verbatim quote from the text in the `quote` field. 
3. SOURCE TRACKING: The context will be provided in blocks separated by "--- [Source: <filename>] ---". You MUST copy the exact <filename> into the `source` field for your evidence.
4. STRICT FORMATTING: 
   - Currencies MUST be 3-letter ISO codes (e.g., "USD", "EUR").
   - ISSNs MUST strictly follow the "NNNN-NNNN" format.
   - Review types must match the allowed canonical values.
5. Always respond with a JSON object matching specified schema without returning extra json keys.

Read the context carefully and extract the data into the requested JSON schema.
"""

llm_model = None

if os.getenv("OLLAMA_BASE_URL"):
    llm_model = OllamaModel(os.getenv("OLLAMA_MODEL", ""))
elif os.getenv("OPENROUTER_MODEL") and os.getenv("OPENROUTER_API_KEY"):
    llm_model = OpenRouterModel(
        os.getenv("OPENROUTER_MODEL", ""),
        provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
    )

FALLBACK_SYSTEM_HINT = (
"If the provided context does not contain the information you need, call the `journal_search` tool to find more relevant "
"documents about this journal. Use specific search terms related to the missing information."
)

basic_info_agent = Agent(
    model=llm_model,
    output_type=BasicInfoExtraction,
    system_prompt=BASE_SYSTEM_PROMPT
    + FALLBACK_SYSTEM_HINT
    + "\nFocus purely on extracting basic information, scope, identifiers (ISSN), facts and metrics.",
    output_retries=3,
    deps_type=JournalSearchDeps,
    tools=[journal_search],
)

policies_agent = Agent(
    model=llm_model,
    output_type=PoliciesExtraction,
    system_prompt=BASE_SYSTEM_PROMPT
    + FALLBACK_SYSTEM_HINT
    + "\nFocus purely on extracting publication frequency, submission guidelines, accepted article types, and review policies.",
    output_retries=3,
    deps_type=JournalSearchDeps,
    tools=[journal_search],
)

fees_agent = Agent(
    model=llm_model,
    output_type=FeesExtraction,
    system_prompt=BASE_SYSTEM_PROMPT
    + FALLBACK_SYSTEM_HINT
    + "\nFocus purely on extracting article processing charges (APCs), fee waivers, discounts, and society/institutional membership models.",
    output_retries=3,
    deps_type=JournalSearchDeps,
    tools=[journal_search],
)

people_agent = Agent(
    model=llm_model,
    output_type=PeopleExtraction,
    system_prompt=BASE_SYSTEM_PROMPT
    + FALLBACK_SYSTEM_HINT
    + "\nFocus purely on extracting editorial board members, their roles/affiliations",
    output_retries=3,
    deps_type=JournalSearchDeps,
    tools=[journal_search],
)
