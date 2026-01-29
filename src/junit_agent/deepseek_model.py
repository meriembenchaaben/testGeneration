from __future__ import annotations

import os
from langchain_openai import ChatOpenAI


def build_deepseek_llm(
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    model: str = "deepseek-chat",
) -> ChatOpenAI:
    """
    Build a DeepSeek API LLM for LangChain/LangGraph usage.
    
    Args:
        api_key: DeepSeek API key. If not provided, will try to read from DEEPSEEK_API_KEY env var.
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum number of tokens to generate.
        model: Model name (default: "deepseek-chat" for DeepSeek V3).
        
    Returns:
        ChatOpenAI instance configured for DeepSeek API.
        
    Raises:
        ValueError: If no API key is provided or found in environment.
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError(
            "DeepSeek API key not provided. Either pass it as api_key parameter "
            "or set DEEPSEEK_API_KEY environment variable."
        )
    
    # DeepSeek uses OpenAI-compatible API
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    
    return llm
