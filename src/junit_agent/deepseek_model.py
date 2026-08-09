from __future__ import annotations

import os
from langchain_openai import ChatOpenAI


def build_deepseek_llm(
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    model: str = "deepseek-chat",
) -> ChatOpenAI:
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "DeepSeek API key not provided. Either pass it as api_key parameter "
            "or set DEEPSEEK_API_KEY environment variable."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    return llm


def build_openrouter_llm(
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    model: str = "deepseek/deepseek-v3.2",
) -> ChatOpenAI:
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenRouter API key not provided. Either pass it as api_key parameter "
            "or set OPENROUTER_API_KEY environment variable."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    return llm
