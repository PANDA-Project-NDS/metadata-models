from pydantic_ai import Agent, RunContext, Tool

from agents.base import llm_model
from agents.config import PassConfig
from agents.prompts import EVIDENCE_INSTRUCTIONS, SEARCH_RULES, SYSTEM_PROMPT
from models.journal import INCLUDE_EVIDENCE
from search import JournalSourcesDeps, journal_search


def context_instructions(ctx: RunContext[JournalSourcesDeps]) -> str:
    return ctx.deps.context_instructions()


journal_search_tool = Tool(
    journal_search,
    name="Journal Search",
    description="Search for additional context, scoped to this journal only.",
    max_retries=1,
    sequential=False,
)


def make_agent(
    pass_config: PassConfig,
    tools: list[Tool] | None = None,
    include_search: bool = True,
) -> Agent[object, str]:
    prompt = SYSTEM_PROMPT.format(
        evidence_instructions=EVIDENCE_INSTRUCTIONS if INCLUDE_EVIDENCE else "",
        search_rules=SEARCH_RULES if include_search else "",
        domain_guidelines=pass_config.domain_guidelines,
    )
    return Agent(
        name=pass_config.name,
        model=llm_model,
        output_type=pass_config.output_type,
        system_prompt=prompt,
        instructions=context_instructions,
        output_retries=3,
        deps_type=JournalSourcesDeps,
        tools=tools
        if tools is not None
        else ([journal_search_tool] if include_search else []),
    )
