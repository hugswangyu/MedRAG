"""Rerankers for multi-source retrieval results.

Three strategies, all sharing the same ``rerank(query, results, top_k)``
interface so they are drop-in interchangeable:

  - SimpleReranker      rule-based (zero added latency)
  - CrossEncoderReranker  neural cross-encoder model
  - LLMReranker         DeepSeek/OpenAI pointwise scoring
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SOURCE_BONUS: Dict[str, float] = {
    "neo4j_kg": 0.20,
    "toyhom_qa": 0.0,
    "user_case": 0.30,
}
DEFAULT_SCORE = 0.50


def _result_text(result: Dict) -> str:
    """Combine all text-like fields in a result for matching / cross-encoding."""
    parts: list[str] = []
    for key in ("answer", "text", "title", "question"):
        v = result.get(key, "")
        if v:
            parts.append(str(v))
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        parts.append(" ".join(str(x) for x in evidence))
    elif evidence:
        parts.append(str(evidence))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 1. SimpleReranker  (rule-based)
# ---------------------------------------------------------------------------

KEYWORD_HIT_BONUS = 0.10
MAX_KEYWORD_BONUS = 0.30


class SimpleReranker:
    """Rule-based: source prior + keyword overlap."""

    def rerank(self, query: str, results: List[Dict],
               top_k: int = 8) -> List[Dict]:
        if not results:
            return []
        keywords = _extract_keywords(query)
        for r in results:
            base = SOURCE_BONUS.get(r.get("source", ""), 0.0) + r.get("score", DEFAULT_SCORE)
            kw_hits = sum(1 for kw in keywords if kw in _result_text(r)) if keywords else 0
            kw_bonus = min(kw_hits * KEYWORD_HIT_BONUS, MAX_KEYWORD_BONUS)
            r["final_score"] = round(base + kw_bonus, 4)
            r["rerank_reason"] = (
                f"source={r.get('source')}, base={base:.3f}"
                + (f", kw×{kw_hits} +{kw_bonus:.2f}" if kw_bonus else "")
            )
        return sorted(results, key=lambda r: r["final_score"], reverse=True)[:top_k]


def _extract_keywords(query: str) -> List[str]:
    q = query.replace("？", "").replace("?", "").replace("，", "").replace("。", "")
    if len(q) <= 2:
        return [q]
    kw: list[str] = []
    for n in (2, 3, 4):
        if len(q) >= n:
            kw.extend(q[i:i + n] for i in range(len(q) - n + 1))
    return kw


# ---------------------------------------------------------------------------
# 2. CrossEncoderReranker  (neural)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Neural cross-encoder for relevance scoring.

    Uses ``sentence-transformers`` CrossEncoder.  The model is downloaded
    on first instantiation (~1 GB for ``BAAI/bge-reranker-base``).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None  # lazy load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: List[Dict],
               top_k: int = 8) -> List[Dict]:
        if not results:
            return []
        pairs = [(query, _result_text(r)) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for r, ce_score in zip(results, scores):
            src_bonus = SOURCE_BONUS.get(r.get("source", ""), 0.0)
            r["final_score"] = round(float(ce_score) + src_bonus, 4)
            r["rerank_reason"] = (
                f"source={r.get('source')}, ce={ce_score:.4f}"
                + (f" +src_bonus={src_bonus:.2f}" if src_bonus else "")
            )

        return sorted(results, key=lambda r: r["final_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# 3. LLMReranker  (pointwise LLM scoring)
# ---------------------------------------------------------------------------

_RANKER_SYSTEM = """你是一个医疗检索结果的相关性评估专家。

你的任务：对给定的用户问题和每个检索结果，评估其相关性。

评分标准（0-1 浮点数）：
- 1.0: 非常相关，直接回答了用户的问题
- 0.7-0.9: 比较相关，提供了有用的参考信息
- 0.4-0.6: 部分相关，但不够直接或不够完整
- 0.1-0.3: 弱相关，仅有微量关联
- 0.0: 完全不相关

请只输出一个 JSON 数组，包含每个结果的相关性分数，格式： [0.9, 0.3, 0.7, ...]
不要输出任何其他文字。"""


class LLMReranker:
    """Pointwise LLM relevance scoring.

    Sends (query, result_text) pairs to an OpenAI-compatible LLM for
    relevance scoring.  Higher accuracy at the cost of latency + token $$.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def rerank(self, query: str, results: List[Dict],
               top_k: int = 8) -> List[Dict]:
        if not results:
            return []

        llm_scores = self._score_batch(query, results)
        if llm_scores is None or len(llm_scores) != len(results):
            logger.warning("LLM scoring failed, falling back to source-prior-only")
            llm_scores = [0.5] * len(results)

        for r, llm_s in zip(results, llm_scores):
            src_bonus = SOURCE_BONUS.get(r.get("source", ""), 0.0)
            r["final_score"] = round(float(llm_s) + src_bonus, 4)
            r["rerank_reason"] = (
                f"source={r.get('source')}, llm={llm_s:.3f}"
                + (f" +src_bonus={src_bonus:.2f}" if src_bonus else "")
            )

        return sorted(results, key=lambda r: r["final_score"], reverse=True)[:top_k]

    def _score_batch(self, query: str,
                     results: List[Dict]) -> Optional[List[float]]:
        items = []
        for i, r in enumerate(results):
            text = _result_text(r)
            items.append(f"[{i}] {text[:300]}")

        user_msg = (
            f"用户问题：{query}\n\n"
            + "待评估的检索结果：\n" + "\n\n".join(items)
            + "\n\n请输出每个结果的相关性分数（JSON 数组）："
        )

        try:
            resp = self.llm.chat.completions.create(
                model=settings.deepseek_default_model,
                messages=[
                    {"role": "system", "content": _RANKER_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content
            return self._parse_scores(raw)
        except Exception:
            logger.debug("LLM rerank call failed", exc_info=True)
            return None

    @staticmethod
    def _parse_scores(raw: str) -> Optional[List[float]]:
        import json, re
        raw = raw.strip()
        m = re.search(r"\[[0-9.,\s]+\]", raw)
        if m:
            try:
                return [float(x) for x in json.loads(m.group(0))]
            except (json.JSONDecodeError, ValueError):
                pass
        return None


# ---------------------------------------------------------------------------
# Factory: best-available reranker with fallback chain
# ---------------------------------------------------------------------------

_DEFAULT_CROSS_MODEL = "BAAI/bge-reranker-base"


def get_reranker(llm_client=None,
                 cross_model: str = _DEFAULT_CROSS_MODEL):
    """Return the best available reranker.

    Priority: CrossEncoder → SimpleReranker (zero-dep fallback).

    Pass ``llm_client`` if you also want LLMReranker as an option
    (selectable via ``.llm`` after construction).

    Usage::

        reranker = get_reranker(llm_client=my_llm)
        ranked = reranker.rerank(query, results, top_k=8)
    """
    try:
        cross = CrossEncoderReranker(model_name=cross_model)
        _ = cross.model  # trigger download / validation
        logger.info("Using CrossEncoderReranker (model=%s)", cross_model)
        return cross
    except Exception:
        logger.warning(
            "CrossEncoder unavailable, falling back to SimpleReranker",
            exc_info=True,
        )
        return SimpleReranker()
