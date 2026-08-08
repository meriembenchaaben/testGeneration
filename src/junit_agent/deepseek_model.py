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


def build_chutes_deepseek_llm(
    api_key: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    model: str = "deepseek-ai/DeepSeek-V3.2-TEE",
) -> ChatOpenAI:
    if api_key is None:
        api_key = os.environ.get("CHUTES_API_KEY")

    if not api_key:
        raise ValueError(
            "Chutes API key not provided. Either pass it as api_key parameter "
            "or set CHUTES_API_KEY environment variable."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1",
    )

    return llm
