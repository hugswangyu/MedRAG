"""混合多源检索器。

通过 QueryRouter 编排两个检索后端：
  1. KGRetriever      — Neo4j 医学知识图谱
  2. ToyhomQARetriever — Milvus 支持的医学问答库

两个源**并行检索**，结果通过 **RRF（Reciprocal Rank Fusion, c=60）** 融合，
再交由下游 Cross-Encoder 做最终重排序。

病例上下文不参与 RRF + Cross-Encoder 流程，
作为独立的优先上下文通道直通 PromptBuilder。
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

    即使文档在各源间无重叠，RRF 也能将不同评分尺度下的排名统一映射到
    可比较的分数空间，实现公平的交错排序。

    Args:
        kg_results: KG 检索结果（按 score 降序，已排名）。
        toyhom_results: Toyhom 检索结果（按 score 降序，已排名）。

    Returns:
        按 RRF 分数降序排列的融合结果列表，每条结果附加 ``rrf_score`` 和 ``rrf_source_rank``。
    """
    if not kg_results and not toyhom_results:
        return []

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


class HybridRetriever:
    """带路由的统一多源检索。

    用法::

        hybrid = HybridRetriever(kg_retriever=kg, toyhom_retriever=toy, router=router)
        result = hybrid.retrieve("感冒了怎么办")
        # result["all_results"] → RRF 融合后的结果列表
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

        两个源通过 ThreadPoolExecutor **并行检索**，任一源失败不影响另一个。
        检索结果经 RRF（c=60）融合后返回。

        Args:
            query: 自然语言医学问题。
            top_k: 向量库返回的最大结果数。
            department: 可选科室过滤，透传给 ToyhomQARetriever.search()。

        Returns:
            字典，键为：route、kg_results、toyhom_results、all_results。
        """
        route = self.router.route(query)

        kg_results: List[Dict] = []
        toyhom_results: List[Dict] = []

        futures: dict = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if route["use_kg"]:
                futures[executor.submit(self._safe_kg_search, query)] = "kg"
            if route["use_toyhom_qa"]:
                futures[
                    executor.submit(self._safe_toyhom_search, query, top_k, department)
                ] = "toyhom"

            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.warning("%s parallel retrieval failed", source, exc_info=True)
                    result = []

                if source == "kg":
                    kg_results = result
                else:
                    toyhom_results = result

        all_results = _rrf_fuse(kg_results, toyhom_results)

        return {
            "route": route,
            "kg_results": kg_results,
            "toyhom_results": toyhom_results,
            "all_results": all_results,
        }

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
        self, query: str, top_k: int, department: str | None = None
    ) -> List[Dict]:
        try:
            return self.toyhom.search(query, top_k=top_k, department=department)
        except Exception:
            logger.warning("Toyhom retrieval failed", exc_info=True)
            return []
