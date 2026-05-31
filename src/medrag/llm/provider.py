"""LLMProvider：将 client + model + provider name 绑定为单一注入对象。"""

from __future__ import annotations

from dataclasses import dataclass

from medrag.llm.clients import get_llm_client


@dataclass(frozen=True)
class LLMProvider:
    """捆绑 provider name、兼容 OpenAI 的 client 以及默认 model。"""

    name: str
    client: object
    default_model: str


def get_llm_provider(provider: str | None = None) -> LLMProvider:
    """返回 *provider* 对应的 LLMProvider。

    若 *provider* 为 None，回退到 ``settings.llm_provider``。
    """
    from medrag.config.settings import settings

    name = (provider or settings.llm_provider).strip().lower()

    if name == "deepseek":
        model = settings.deepseek_default_model
    elif name == "zhipuai":
        model = settings.zhipuai_model
    elif name == "qwen":
        model = settings.qwen_model
    elif name == "ollama":
        model = settings.ollama_model
    else:
        raise ValueError(
            f"不支持的 LLM_PROVIDER: {name!r}，可选值为 deepseek / zhipuai / qwen / ollama"
        )

    return LLMProvider(
        name=name,
        client=get_llm_client(name),
        default_model=model,
    )
