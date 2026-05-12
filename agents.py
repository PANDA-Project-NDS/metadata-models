import os
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models.journal import (
    BasicInfoExtraction,
    EditorialExtraction,
    FeesExtraction,
    PoliciesExtraction,
)
from search import JournalSourcesDeps, journal_search

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


@dataclass
class PassConfig:
    """RAG Extraction pass config"""

    name: str
    output_type: type[BaseModel]
    queries: list[str]


PASSES: list[PassConfig] = [
    PassConfig(
        "Info Agent",
        BasicInfoExtraction,
        [
            "Journal title, publisher, about this journal, mission, scope, sections",
            "ISSN, print ISSN, online ISSN, indexed in, abstracting and indexing databases",
            "Impact factor, journal metrics, citation score, cite score",
        ],
    ),
    PassConfig(
        "Policies Agent",
        PoliciesExtraction,
        [
            "Publication frequency, issues per year, submission guidelines, author instructions, article types accepted",
            "Peer review process, blind review, open access policy statement, copyright, quality assurance",
            "diamond open access, community owned, open to all authors",
            "publication languages, languages accepted",
        ],
    ),
    PassConfig(
        "Fees Agent",
        FeesExtraction,
        [
            "Article Processing Charge, APC, publication fees, cost, waivers, discounts, society membership, institutional membership",
        ],
    ),
    PassConfig(
        "Editors Agent",
        EditorialExtraction,
        [
            "Editorial board, Editor in Chief, managing editor, editorial team",
        ],
    ),
]

# --- Prompts & Agents ---
# Common system prompt base
SEARCH_FALLBACK_PROMPT = """
Search Tool:
- If the provided context does not contain the information you need, call the `Journal Search` tool to find more relevant 
documents about this journal.
- Use only if the information is not present in the provided context. Do not use it if the information is already present.
- Use it to fill the blanks, do not default to null output.
- It only searches in documents related to the current journal.
- Create a specific search query related to the missing information.
- After calling the search tool, you will receive additional context chunks. Re-analyze all the context (previous + new) to extract the information.
"""

SYSTEM_PROMPT = f"""
You are an expert data extraction assistant. Your task is to extract highly accurate, structured metadata for an academic journal based ONLY on the provided context chunks.

CRITICAL RULES:
1. NO HALLUCINATION: If a piece of information is not explicitly stated, you MUST output null or an empty list. Do not guess or infer.
2. VERBATIM EVIDENCE: For every extracted value, you must provide the exact, verbatim quote from the text in the `quote` field. 
3. SOURCE TRACKING: The context will be provided in blocks separated by "--- [Source: <filename>] ---". You MUST copy the exact <filename> into the `source` field for your evidence.
4. STRUCTURED OUTPUT: Always respond with JSON matching the schema specified by the `final_result` output tool without returning extra json keys.

{SEARCH_FALLBACK_PROMPT}

Read the context carefully and extract the data into the requested JSON schema. Focus only on the fields present in the output schema.
"""


def context_instructions(ctx: RunContext[JournalSourcesDeps]) -> str:
    return ctx.deps.context_instructions()


llm_model = OpenAIChatModel(
    os.getenv("OPENAI_MODEL", ""),
    provider=OpenAIProvider(
        os.getenv("OPENAI_API_URL"),
        http_client=create_async_http_client(
            timeout=int(os.getenv("OPENAI_HTTP_TIMEOUT", "60"))
        ),
    ),
    # profile=OpenAIModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)
)

journal_search_tool = Tool(
    journal_search,
    name="Journal Search",
    max_retries=1,
)


def make_agent(pass_config: PassConfig) -> Agent:
    return Agent(
        name=pass_config.name,
        model=llm_model,
        output_type=pass_config.output_type,
        system_prompt=SYSTEM_PROMPT,
        instructions=context_instructions,
        output_retries=3,
        deps_type=JournalSourcesDeps,
        tools=[journal_search_tool],
    )
