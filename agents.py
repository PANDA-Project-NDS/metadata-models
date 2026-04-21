from pydantic import BaseModel, Field
from typing import List, Optional
from models.journal import (
    BasicInfoExtraction,
    PoliciesExtraction,
    FeesExtraction,
    PeopleMetricsExtraction
)

# --- Extraction Queries ---

EXTRACTION_QUERIES = {
    "basic_info": [
        "Journal title, publisher, about this journal, mission, scope, sections",
        "ISSN, print ISSN, online ISSN, indexed in, abstracting and indexing databases"
    ],
    "policies_submissions": [
        "Publication frequency, issues per year, submission guidelines, author instructions, article types accepted",
        "Peer review process, blind review, open access policy statement, copyright, quality assurance"
    ],
    "fees_membership": [
        "Article Processing Charge, APC, publication fees, cost, waivers, discounts, society membership, institutional membership"
    ],
    "people_metrics": [
        "Editorial board, Editor in Chief, managing editor, editorial team",
        "Impact factor, journal metrics, citation score, cite score"
    ]
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

Read the context carefully and extract the data into the requested JSON schema.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    base_url='http://127.0.0.1:1234/v1',
    api_key='local-dev'
)

llm_model = OpenAIModel(
    'qwen/qwen3-1.7b',
    provider=provider
)

basic_info_agent = Agent(
    model=llm_model,
    output_type=BasicInfoExtraction,
    system_prompt=BASE_SYSTEM_PROMPT + "\nFocus purely on extracting basic information, scope, identifiers (ISSN), and facts."
)

policies_agent = Agent(
    model=llm_model,
    output_type=PoliciesExtraction,
    system_prompt=BASE_SYSTEM_PROMPT + "\nFocus purely on extracting publication frequency, submission guidelines, accepted article types, and review policies."
)

fees_agent = Agent(
    model=llm_model,
    output_type=FeesExtraction,
    system_prompt=BASE_SYSTEM_PROMPT + "\nFocus purely on extracting article processing charges (APCs), fee waivers, discounts, and society/institutional membership models."
)

people_metrics_agent = Agent(
    model=llm_model,
    output_type=PeopleMetricsExtraction,
    system_prompt=BASE_SYSTEM_PROMPT + "\nFocus purely on extracting editorial board members, their roles/affiliations, and journal impact metrics."
)
