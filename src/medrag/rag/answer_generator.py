"""答案生成器：封装 LLM 调用以生成最终回答。"""

from __future__ import annotations

from medrag.llm import get_llm_provider
from medrag.llm.provider import LLMProvider


class AnswerGenerator:
    """调用配置好的 LLM，根据提示词生成最终回答。

    用法::

        generator = AnswerGenerator()
        answer = generator.generate(prompt)

        # 注入自定义 provider:
        generator = AnswerGenerator(llm_provider=get_llm_provider("ollama"))
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._provider = llm_provider or get_llm_provider()
        self._client = self._provider.client
        self._model = self._provider.default_model

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """发送 *prompt* 到 LLM，返回生成的回答文本。

        Args:
            prompt: 完整的提示词字符串（通常来自
                    :meth:`PromptBuilder.build_answer_prompt`）。

        Returns:
            LLM 的响应文本，失败时返回友好的错误消息。
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            provider = self.provider
            return (
                f"抱歉，调用 {self._provider.name} 生成回答时出错：{exc}\n"
                f"请检查 API Key 是否正确、网络是否通畅。"
            )

    def generate_stream(self, prompt: str):
        """流式调用 LLM，逐 token yield 文本片段。"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            yield f"\n[流式生成错误: {exc}]"
