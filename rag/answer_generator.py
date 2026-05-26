"""Answer Generator: encapsulates LLM calls for final answer generation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings


class AnswerGenerator:
    """Call the configured LLM to generate a final answer from a prompt.

    Usage::

        generator = AnswerGenerator()
        answer = generator.generate(prompt)
    """

    def __init__(self, provider: str | None = None):
        """
        Args:
            provider: One of ``"deepseek"``, ``"zhipuai"``, ``"ollama"``.
                      Defaults to ``settings.llm_provider``.
        """
        self.provider = (provider or settings.llm_provider).strip().lower()

        if self.provider == "deepseek":
            self._model = settings.deepseek_answer_model
            self._client = self._create_deepseek_client()
        elif self.provider == "zhipuai":
            self._model = settings.zhipuai_model
            self._client = self._create_zhipuai_client()
        elif self.provider == "ollama":
            self._model = settings.ollama_model
            self._client = self._create_ollama_client()
        else:
            raise ValueError(
                f"不支持的 LLM_PROVIDER: {self.provider!r}，"
                f"可选值为 deepseek / zhipuai / ollama"
            )

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    @staticmethod
    def _create_deepseek_client():
        from openai import OpenAI

        return OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    @staticmethod
    def _create_zhipuai_client():
        from zhipuai import ZhipuAI

        return ZhipuAI(api_key=settings.zhipuai_api_key)

    @staticmethod
    def _create_ollama_client():
        from openai import OpenAI

        return OpenAI(
            api_key="ollama",  # ollama doesn't require a real key
            base_url=settings.ollama_base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the generated answer text.

        Args:
            prompt: The full prompt string (typically from
                    :meth:`PromptBuilder.build_answer_prompt`).

        Returns:
            The LLM response text, or a friendly error message on failure.
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
                f"抱歉，调用 {provider} 生成回答时出错：{exc}\n"
                f"请检查 API Key 是否正确、网络是否通畅。"
            )
