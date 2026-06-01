"""医疗查询规范化。

将口语化问题转换为更适合 KG / 向量库检索的中文医学查询，同时保留
原始问题供最终回答使用。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable


_TERM_REWRITES: tuple[tuple[str, str], ...] = (
    ("嗓子疼", "咽痛"),
    ("喉咙疼", "咽痛"),
    ("拉肚子", "腹泻"),
    ("肚子疼", "腹痛"),
    ("发烧", "发热"),
    ("高烧", "高热"),
    ("低烧", "低热"),
    ("头疼", "头痛"),
    ("流鼻涕", "流涕"),
    ("鼻塞", "鼻塞"),
    ("咳嗽", "咳嗽"),
    ("心口疼", "胸痛"),
    ("喘不上气", "呼吸困难"),
    ("恶心想吐", "恶心 呕吐"),
    ("血压高", "高血压"),
    ("血糖高", "高血糖"),
    ("尿酸高", "高尿酸血症"),
    ("胆固醇高", "高胆固醇血症"),
)

_FILLER_PATTERNS: tuple[str, ...] = (
    r"请问",
    r"麻烦问一下",
    r"我想问(一下)?",
    r"帮我看看",
    r"能不能",
    r"可以不可以",
    r"该不该",
    r"是不是",
    r"有没有必要",
)


@dataclass(frozen=True)
class NormalizedQuery:
    """规范化查询结果。"""

    original_query: str
    normalized_query: str
    medical_terms: list[str]
    rewrite_reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


class QueryNormalizer:
    """轻量级中文医学 query normalizer。

    当前版本优先使用确定性规则，避免把检索前置步骤绑定到额外 LLM 调用。
    后续如果需要中英翻译或多查询扩展，可以在此类中增加 LLM 分支。
    """

    def normalize(self, query: str) -> NormalizedQuery:
        original = (query or "").strip()
        normalized = _normalize_space(original)
        normalized = _strip_fillers(normalized)

        medical_terms: list[str] = []
        reasons: list[str] = []
        for colloquial, canonical in _TERM_REWRITES:
            if colloquial in normalized:
                normalized = normalized.replace(colloquial, canonical)
                medical_terms.extend(_split_terms(canonical))
                reasons.append(f"{colloquial}->{canonical}")
            elif canonical in normalized:
                medical_terms.extend(_split_terms(canonical))

        normalized = _normalize_space(normalized)
        if not normalized:
            normalized = original

        seen: set[str] = set()
        deduped_terms = []
        for term in medical_terms:
            if term and term not in seen:
                seen.add(term)
                deduped_terms.append(term)

        rewrite_reason = "；".join(reasons) if reasons else "未发现需要改写的口语医学表达"
        return NormalizedQuery(
            original_query=original,
            normalized_query=normalized,
            medical_terms=deduped_terms,
            rewrite_reason=rewrite_reason,
        )


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("？", "?").strip())


def _strip_fillers(text: str) -> str:
    result = text
    for pattern in _FILLER_PATTERNS:
        result = re.sub(pattern, "", result)
    return result.strip(" ，,。？?")


def _split_terms(text: str) -> Iterable[str]:
    return [part.strip() for part in text.split() if part.strip()]
