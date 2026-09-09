import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Chat model routed through OpenRouter (OpenAI-compatible endpoint)."""
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )
