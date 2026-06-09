"""Elasticsearch BM25 检索器：对 cMedQA2 语料库做关键词 / 医学实体召回。

用于与 Milvus ANN（BGE dense）构成双路检索：
  Dense（Milvus）+ Sparse（ES BM25）→ RRF → Cross-Encoder
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from elasticsearch import Elasticsearch, helpers

from medrag.config.settings import settings
logger = logging.getLogger(__name__)

# ES 索引 mapping：三个文本字段做 BM25，department 做 keyword 过滤
INDEX_MAPPINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "rag_standard": {
                    "type": "standard",
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "pk": {"type": "keyword"},
            "department": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
            "question": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
            "answer": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
            "text": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
        }
    },
}


class ESBM25Retriever:
    """Elasticsearch BM25 关键词检索器。

    用法::

        retriever = ESBM25Retriever()
        results = retriever.search("感冒吃什么药", top_k=10)
    """

    def __init__(
        self,
        hosts: str = settings.es_hosts,
        index_name: str = settings.es_index_name,
    ):
        self.hosts = hosts
        self.index_name = index_name
        self._client: Elasticsearch | None = None

    @property
    def client(self) -> Elasticsearch:
        if self._client is None:
            self._client = Elasticsearch(hosts=self.hosts)
        return self._client

    def search(
        self,
        query: str,
        top_k: int = 10,
        department: Optional[str] = None,
    ) -> List[Dict]:
        """BM25 检索，返回统一格式的结果列表。"""
        if not self.client.indices.exists(index=self.index_name):
            logger.warning("ES index %s does not exist", self.index_name)
            return []

        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "title^2", "answer"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ]
        filter_clauses = []
        if department:
            filter_clauses.append({"term": {"department": department}})

        body = {
            "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
            "size": top_k,
            "_source": True,
        }

        resp = self.client.search(index=self.index_name, body=body)
        hits: List[Dict] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            hits.append(
                {
                    "source": "cmedqa2_es",
                    "id": src.get("pk", hit["_id"]),
                    "score": float(hit["_score"]),
                    "department": src.get("department", ""),
                    "title": src.get("title", ""),
                    "question": src.get("question", ""),
                    "answer": src.get("answer", ""),
                    "text": src.get("text", ""),
                }
            )
        return hits

    @staticmethod
    def _result_text(result: Dict) -> str:
        parts = []
        for key in ("answer", "text", "title", "question"):
            v = result.get(key, "")
            if v:
                parts.append(str(v))
        return " ".join(parts)
