"""混合多源检索器。

通过 QueryRouter 编排两个检索后端：
  1. KGRetriever      — Neo4j 医学知识图谱
  2. ToyhomQARetriever — Milvus 支持的医学问答库

下游执行策略由路由决策决定：

+---------------------+---------------+-------------+------------------+
| 路由决策             | 检索方式       | 融合        | 重排              |
+---------------------+---------------+-------------+------------------+
| 仅 KG               | 只调 KG        | 跳过 RRF    | CrossEncoder      |
| 仅 Toyhom           | 只调 Toyhom    | 跳过 RRF    | CrossEncoder      |
| KG + Toyhom 同时开  | 并行检索两者   | RRF (c=60)  | CrossEncoder      |
| 两者都不开           | 不检索          | —           | —                 |
+---------------------+---------------+-------------+------------------+

病例上下文不参与上述任何流程，作为独立的优先上下文通道直通 PromptBuilder。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

logger = logging.getLogger(__name__)

# RRF 常数：控制排名差异对最终分数的影响程度，60 是 TREC 等 IR 评测的标准取值
RRF_C = 60


def _rrf_fuse(
    kg_results: List[Dict],
    toyhom_results: List[Dict],
) -> List[Dict]:
    """对两个独立排序的检索结果进行倒数排名融合。

    RRF_score(d) = Σ 1/(c + rank_s(d))

    仅当两个源都有结果时调用；单源场景不需要融合。

    Args:
        kg_results: KG 检索结果（按 score 降序，已排名）。
        toyhom_results: Toyhom 检索结果（按 score 降序，已排名）。

    Returns:
        按 RRF 分数降序排列的融合结果列表，每条结果附加 ``rrf_score`` 和 ``rrf_source_rank``。
    """
    fused: List[Dict] = []

    for rank, r in enumerate(kg_results, start=1):
        r = dict(r)
        r["rrf_score"] = round(1.0 / (RRF_C + rank), 6)
        r["rrf_source_rank"] = rank
        fused.append(r)

    for rank, r in enumerate(toyhom_results, start=1):
        r = dict(r)
        r["rrf_score"] = round(1.0 / (RRF_C + rank), 6)
        r["rrf_source_rank"] = rank
        fused.append(r)

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused


def _tag_single_source(results: List[Dict], source_tag: str) -> List[Dict]:
    """为单源结果附加排名标记（不执行 RRF，仅保留源内排名信息）。"""
    tagged: List[Dict] = []
    for rank, r in enumerate(results, start=1):
        r = dict(r)
        r["rrf_score"] = None           # 单源无融合，下游不应依赖此字段
        r["rrf_source_rank"] = rank
        tagged.append(r)
    return tagged


class HybridRetriever:
    """带路由的统一多源检索。

    根据 QueryRouter 的决策动态选择执行路径：
    单源 → 直通 CrossEncoder；双源 → RRF 融合 → CrossEncoder。

    用法::

        hybrid = HybridRetriever(kg_retriever=kg, toyhom_retriever=toy, router=router)
        result = hybrid.retrieve("感冒了怎么办")
        # result["all_results"] → 根据路由决策处理后的结果列表
    """

    def __init__(self, kg_retriever, toyhom_retriever, router):
        self.kg = kg_retriever
        self.toyhom = toyhom_retriever
        self.router = router

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        department: str | None = None,
    ) -> Dict:
        """路由 *query* 并从合适的源获取结果。

        根据路由决策分三种路径处理检索结果：
        - 单源：跳过 RRF，标记源内排名后直通下游 CrossEncoder。
        - 双源：并行检索 + RRF (c=60) 融合。
        - 无源：返回空列表。

        Args:
            query: 自然语言医学问题。
            top_k: 向量库返回的最大结果数。
            department: 可选科室过滤，透传给 ToyhomQARetriever.search()。

        Returns:
            字典，键为：route、kg_results、toyhom_results、all_results、
            ``fusion_mode``（"rrf" / "single" / "none"）。
        """
        route = self.router.route(query)

        use_kg = route["use_kg"]
        use_qa = route["use_toyhom_qa"]

        # 两者都不开 → 直接返回
        if not use_kg and not use_qa:
            return {
                "route": route,
                "kg_results": [],
                "toyhom_results": [],
                "all_results": [],
                "fusion_mode": "none",
            }

        kg_results, toyhom_results = self._run_retrieval(
            query, use_kg, use_qa, top_k, department,
        )

        # 根据实际命中源的数量选择融合策略
        both_hit = bool(kg_results) and bool(toyhom_results)
        if both_hit:
            all_results = _rrf_fuse(kg_results, toyhom_results)
            fusion_mode = "rrf"
        elif kg_results:
            all_results = _tag_single_source(kg_results, "kg")
            fusion_mode = "single"
        elif toyhom_results:
            all_results = _tag_single_source(toyhom_results, "toyhom")
            fusion_mode = "single"
        else:
            all_results = []
            fusion_mode = "none"

        return {
            "route": route,
            "kg_results": kg_results,
            "toyhom_results": toyhom_results,
            "all_results": all_results,
            "fusion_mode": fusion_mode,
        }

    # ------------------------------------------------------------------
    # 检索执行
    # ------------------------------------------------------------------

    def _run_retrieval(
        self,
        query: str,
        use_kg: bool,
        use_qa: bool,
        top_k: int,
        department: str | None,
    ) -> tuple[List[Dict], List[Dict]]:
        """按路由决策执行检索：仅 KG、仅 Toyhom，或两者并行。"""
        if use_kg and use_qa:
            return self._parallel_retrieve(query, top_k, department)
        if use_kg:
            return (self._safe_kg_search(query), [])
        return ([], self._safe_toyhom_search(query, top_k, department))

    def _parallel_retrieve(
        self, query: str, top_k: int, department: str | None,
    ) -> tuple[List[Dict], List[Dict]]:
        """双源并行检索，任一失败不影响另一个。"""
        kg_results: List[Dict] = []
        toyhom_results: List[Dict] = []

        futures: dict = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures[executor.submit(self._safe_kg_search, query)] = "kg"
            futures[
                executor.submit(self._safe_toyhom_search, query, top_k, department)
            ] = "toyhom"

            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.warning(
                        "%s parallel retrieval failed", source, exc_info=True,
                    )
                    result = []

                if source == "kg":
                    kg_results = result
                else:
                    toyhom_results = result

        return kg_results, toyhom_results

    # ------------------------------------------------------------------
    # 内部辅助方法（每个源独立 try/except）
    # ------------------------------------------------------------------

    def _safe_kg_search(self, query: str) -> List[Dict]:
        try:
            return self.kg.search(query)
        except Exception:
            logger.warning("KG retrieval failed", exc_info=True)
            return []

    def _safe_toyhom_search(
        self, query: str, top_k: int, department: str | None = None,
    ) -> List[Dict]:
        try:
            return self.toyhom.search(query, top_k=top_k, department=department)
        except Exception:
            logger.warning("Toyhom retrieval failed", exc_info=True)
            return []
