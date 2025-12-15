from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


def build_local_hf_llm(
    model_id: str,
    temperature: float,
    max_new_tokens: int,
) -> HuggingFacePipeline:
    """
    Minimal local (open-source) Hugging Face LLM for LangChain/LangGraph usage.
    Uses a Transformers text-generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)
