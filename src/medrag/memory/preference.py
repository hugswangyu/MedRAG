"""User preference store — key-value pairs extracted from conversation.

Two extraction paths:
  1. LLM-based (primary): calls DeepSeek to extract structured preferences.
  2. Rule-based (fallback): regex patterns for common Chinese patterns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PREFERENCE_EXTRACT_PROMPT = (
    "你是一个用户偏好提取器。从用户的对话消息中提取用户的个人偏好信息。\n\n"
    "只提取明确陈述的事实，不要猜测或推断。\n\n"
    '返回 JSON 格式，key 用中文，value 是提取到的内容。'
    '如果没有找到任何偏好信息，返回 {{"preferences": []}}。\n\n'
    "示例：\n"
    "用户：我叫张三，今年28岁，平时喜欢打篮球\n"
    '输出：{{"preferences": [["姓名", "张三"], ["年龄", "28"], ["喜好", "打篮球"]]}}\n\n'
    "用户：我住在北京\n"
    '输出：{{"preferences": [["城市", "北京"]]}}\n\n'
    "用户消息：{text}\n\n"
    "请只输出 JSON，不要加任何其他文字："
)


class PreferenceStore:
    """User preference key-value store with rule-based extraction.

    Supports LLM-based extraction as an upgrade path — for now,
    uses regex rules to capture common preference patterns from user messages.
    Persisted to PostgreSQL for multi-tenant isolation.
    """

    def __init__(self, username: str = ""):
        self._username = username
        self._data: Dict[str, str] = {}
        if username:
            self._load_from_pg()

        # Rule patterns: (regex, key_name) — mirrors AGI-saber ExtractAndSave
        self._rules: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"我(?:叫|是|的名字(?:是|为)?)\s*(.+?)(?:[，。\.]|$)"), "姓名"),
            (re.compile(r"我(?:喜欢|爱)\s*(.+?)(?:[，。\.]|$)"), "喜好"),
            (re.compile(r"我(?:住在|来自|在)\s*(.+?)(?:[，。\.]|$)"), "城市"),
            (re.compile(r"我(?:今年|年龄)\s*(\d+)\s*岁"), "年龄"),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _save_to_pg(self, key: str, value: str) -> None:
        if not self._username:
            return
        try:
            from medrag.infrastructure.storage.postgres_client import pref_save
            pref_save(self._username, key, value)
        except Exception:
            logger.debug("Failed to persist preference", exc_info=True)

    def _load_from_pg(self) -> None:
        try:
            from medrag.infrastructure.storage.postgres_client import pref_load_all
            self._data = pref_load_all(self._username)
        except Exception:
            logger.debug("Failed to load preferences from PG", exc_info=True)

    def save(self, key: str, value: str) -> None:
        if key and value:
            self._data[key] = value
            self._save_to_pg(key, value)

    def save_batch(self, kvs: Dict[str, str]) -> None:
        for k, v in kvs.items():
            if k and v:
                self._data[k] = v
                self._save_to_pg(k, v)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(key, default)

    def all(self) -> Dict[str, str]:
        return dict(self._data)

    def extract_and_save(self, text: str) -> Optional[Tuple[str, str]]:
        """Try to extract a preference from *text* using rules.

        Returns (key, value) if matched, else None.
        Mirrors AGI-saber Preference.ExtractAndSave.
        """
        for pattern, key in self._rules:
            m = pattern.search(text)
            if m:
                value = m.group(1).strip()
                self.save(key, value)
                return (key, value)
        return None

    def llm_extract(self, text: str) -> None:
        """Extract preferences via DeepSeek and save all found key-value pairs.

        Mirrors AGI-saber a.llmExtractPreferences() in agent.go.
        Non-blocking: failures are silently logged, never raised.
        """
        try:
            from openai import OpenAI
            from medrag.config.settings import settings

            client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            prompt = _PREFERENCE_EXTRACT_PROMPT.format(text=text)
            response = client.chat.completions.create(
                model=settings.deepseek_intent_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            if not raw:
                return
            data = json.loads(raw)
            for pair in data.get("preferences", []):
                if len(pair) == 2 and pair[0] and pair[1]:
                    self.save(str(pair[0]).strip(), str(pair[1]).strip())
        except Exception:
            logger.debug("LLM preference extraction failed", exc_info=True)

    def build_context(self) -> str:
        """Format stored preferences as a context string for LLM prompts.

        Mirrors AGI-saber Preference.BuildContext.
        Returns empty string if no preferences stored.
        """
        if not self._data:
            return ""
        lines = [f"{k}: {v}" for k, v in self._data.items()]
        return "【用户偏好】\n" + "\n".join(lines)

    @property
    def data(self) -> Dict[str, str]:
        return self._data
