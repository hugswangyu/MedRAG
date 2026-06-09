from medrag.vectors.milvus_client import MilvusClientWrapper

try:
    from medrag.vectors.embedding import EmbeddingModel
except Exception:  # pragma: no cover - optional sentence-transformers dependency
    EmbeddingModel = None  # type: ignore[assignment]

try:
    from medrag.vectors.qa_retriever import QARetriever
except Exception:  # pragma: no cover - optional vector runtime dependency
    QARetriever = None  # type: ignore[assignment]

__all__ = ["EmbeddingModel", "MilvusClientWrapper", "QARetriever"]
