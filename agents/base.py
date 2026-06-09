import os

from dotenv import load_dotenv
from pydantic_ai import ModelSettings, InstrumentationSettings
from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

_langfuse_available = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

if _langfuse_available:
    from langfuse import get_client
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from pydantic_ai.agent import Agent

    langfuse = get_client()
    LlamaIndexInstrumentor().instrument()
    Agent.instrument_all(InstrumentationSettings(version = 3))
else:
    import logfire
    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    langfuse = None

def get_model(role: str | None = None) -> OpenAIChatModel:
    """Create an OpenAIChatModel, with optional per-role env var override.

    Falls back to OPENAI_MODEL when the role-specific var isn't set.
    """
    env_var = f"{role.upper()}_MODEL" if role else "OPENAI_MODEL"
    model_name = os.getenv(env_var) or os.getenv("OPENAI_MODEL", "")
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            os.getenv("OPENAI_API_URL"),
            http_client=create_async_http_client(
                timeout=int(os.getenv("OPENAI_HTTP_TIMEOUT", "60"))
            ),
        ),
        settings=ModelSettings(
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        ),
    )
