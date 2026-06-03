import os

import logfire
from dotenv import load_dotenv
from pydantic_ai import ModelSettings
from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

llm_model = OpenAIChatModel(
    os.getenv("OPENAI_MODEL", ""),
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
