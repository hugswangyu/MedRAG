"""Medical query intent recognition.

Extracted from KGRetriever so it can be reused independently of the
Neo4j pipeline.  Uses a few-shot LLM prompt to classify a user question
into one or more of 15 predefined medical query intents.
"""

from __future__ import annotations

from medrag.config.settings import settings
from medrag.prompts import INTENT_PROMPT_TEMPLATE


def recognize_intents(query: str, llm_client) -> str:
    """Call the LLM for intent recognition.

    Args:
        query: Natural-language medical question.
        llm_client: OpenAI-compatible client (chat.completions.create).

    Returns:
        Raw API response string (e.g.
        ``["查询疾病简介","查询疾病病因"] # comment``), or ``""`` on failure.
    """
    try:
        prompt = INTENT_PROMPT_TEMPLATE.format(query=query)
        response = llm_client.chat.completions.create(
            model=settings.deepseek_default_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return ""
