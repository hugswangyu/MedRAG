"""Schema-Driven Context Assembly — Slot + priority-based budget management.

Mirrors AGI-saber runtime/context_schema.go RuntimeContextSchema + ContextAssembler.

Usage::

    assembler = ContextAssembler(budget=4096)
    assembler.add("memory", "【用户偏好】\\n姓名: 张三", priority=100)
    assembler.add("kg", "【知识图谱】\\n- 高血压需要低盐饮食", priority=80)
    assembler.add("qa", "【问答匹配】\\n- 多喝水休息", priority=70)

    context = assembler.assemble()
    # → drops QA if over budget, keeps memory + KG
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
# Rough heuristic: Chinese chars ~1.5 tokens, English ~4 chars per token.
# Used only for budget-aware pruning; exact tokenizer is the LLM's job.
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count for Chinese-English mixed text.

    Chinese characters: ~1 token per 1.5 chars.
    English/ASCII words: ~1 token per 4 chars.
    """
    if not text:
        return 0
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - chinese
    return int(chinese / 1.5 + other / 4)


# ---------------------------------------------------------------------------
# Slot
# ---------------------------------------------------------------------------

# Priority levels for medical RAG (higher = more important, kept first)
PRIORITY_MEMORY = 100       # user preferences + conversation history
PRIORITY_CASE_SUMMARY = 90  # patient case summary
PRIORITY_KG = 80            # knowledge graph facts
PRIORITY_QA = 70            # QA pair results
PRIORITY_CASE_CHUNKS = 60   # case document fragments
PRIORITY_QUERY = 50         # user question


@dataclass
class Slot:
    """A single context slot with priority and content.

    Attributes:
        kind: Section identifier (memory, kg, qa, case_summary, ...).
        content: Pre-formatted text content (None = empty/dropped).
        priority: Higher = more important when pruning under budget.
        token_count: Estimated token count (auto-computed).
    """

    kind: str
    content: Optional[str]
    priority: int = 50
    token_count: int = 0

    def __post_init__(self):
        if self.content and self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Priority-based context assembly with global token budget.

    Mirrors AGI-saber runtime.ContextAssembler.

    1. Add pre-formatted slots with priorities.
    2. Call ``assemble()`` to sort, prune, and join.
    """

    def __init__(self, budget: int = 4096):
        self._slots: List[Slot] = []
        self._budget = budget

    def add(self, kind: str, content: Optional[str],
            priority: int = 50) -> None:
        """Register a pre-formatted context slot."""
        if not content or not content.strip():
            return
        self._slots.append(Slot(
            kind=kind,
            content=content.strip(),
            priority=priority,
        ))

    def assemble(self) -> str:
        """Sort by priority, prune to budget, return joined context.

        Returns empty string if nothing survives pruning.
        """
        if not self._slots:
            return ""

        # 1. Sort by priority descending
        self._slots.sort(key=lambda s: s.priority, reverse=True)

        # 2. Prune to budget (drop lowest-priority slots first)
        total = sum(s.token_count for s in self._slots)
        if total > self._budget:
            for slot in reversed(self._slots):
                if total <= self._budget:
                    break
                total -= slot.token_count
                slot.content = None  # drop
                logger.debug(
                    "ContextAssembler pruned slot '%s' "
                    "(priority=%d, %d tokens)",
                    slot.kind, slot.priority, slot.token_count,
                )

        # 3. Assemble remaining
        parts = [s.content for s in self._slots if s.content]
        return "\n\n".join(parts)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self._slots if s.content)

    @property
    def used_slots(self) -> List[str]:
        return [s.kind for s in self._slots if s.content]

    @property
    def dropped_slots(self) -> List[str]:
        return [s.kind for s in self._slots if s.content is None]

    def reset(self) -> None:
        self._slots.clear()


__all__ = [
    "Slot",
    "ContextAssembler",
    "estimate_tokens",
    "PRIORITY_MEMORY",
    "PRIORITY_CASE_SUMMARY",
    "PRIORITY_KG",
    "PRIORITY_QA",
    "PRIORITY_CASE_CHUNKS",
    "PRIORITY_QUERY",
]
