"""医疗查询意图识别。

从 KGRetriever 中提取，可独立于 Neo4j 流水线复用。
通过 few-shot LLM 提示词将用户问题分类到 15 个预定义医学查询意图之一。

意图识别使用独立的 DeepSeek 客户端（不受 LLM_PROVIDER 影响），
确保 KG 检索的意图分类稳定可靠。
"""

from __future__ import annotations

from openai import OpenAI

from medrag.config.settings import settings
from medrag.prompts import INTENT_PROMPT_TEMPLATE

_intent_client: OpenAI | None = None


def _get_intent_client() -> OpenAI:
    """获取意图识别专用的 DeepSeek 客户端（延迟初始化，单例）。"""
    global _intent_client
    if _intent_client is None:
        _intent_client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _intent_client


def recognize_intents(query: str, llm_client=None) -> str:
    """调用 DeepSeek 进行意图识别。

    Args:
        query: 自然语言医学问题。
        llm_client: 已废弃，保留参数仅用于向后兼容。

    Returns:
        原始 API 响应字符串（如 ``["查询疾病简介","查询疾病病因"] # 注释``），
        失败时返回 ``""``。
    """
    try:
        prompt = INTENT_PROMPT_TEMPLATE.format(query=query)
        client = _get_intent_client()
        response = client.chat.completions.create(
            model=settings.deepseek_intent_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return ""
