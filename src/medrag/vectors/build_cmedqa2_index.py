"""为 cMedQA2 医学问答数据集构建 Milvus 向量索引 + ES BM25 索引。"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

from medrag.config.settings import settings
from medrag.data.cmedqa2_loader import load_cmedqa2_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 128


def _batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_milvus_index(
    limit: int | None = None,
    recreate: bool = False,
    batch_size: int = BATCH_SIZE,
) -> int:
    """在 Milvus 中构建 cMedQA2 向量索引（嵌入 question + answer）。"""
    docs = load_cmedqa2_dataset(limit=limit)
    logger.info("Loaded cMedQA2 docs: %d", len(docs))
    if not docs:
        return 0

    from medrag.vectors.embedding import EmbeddingModel
    from medrag.vectors.milvus_client import MilvusClientWrapper

    embedding_model = EmbeddingModel(settings.embedding_model_name)
    milvus = MilvusClientWrapper()
    milvus.connect()
    milvus.create_collection(embedding_model.embedding_dim, recreate=recreate)

    inserted = 0
    skipped = 0
    total = len(docs)
    from tqdm import tqdm
    import time
    t0 = time.time()
    pbar = tqdm(total=total, unit="doc", desc="Milvus", ncols=80)
    for batch_docs in _batched(docs, batch_size):
        texts = [
            f"{doc['question']} {doc['answer']}" for doc in batch_docs
        ]
        embeddings = embedding_model.encode(texts, batch_size=batch_size, is_query=False)
        ok = milvus.insert_batch(batch_docs, embeddings)
        if ok:
            inserted += len(batch_docs)
        else:
            skipped += len(batch_docs)
        pbar.update(len(batch_docs))

    pbar.close()
    milvus.flush()
    milvus.create_index()
    milvus.load_collection()
    elapsed = time.time() - t0
    logger.info(
        "Milvus index build finished in %.0fs. inserted=%d, skipped=%d, total=%d",
        elapsed, inserted, skipped, total,
    )
    return inserted


def build_es_index(
    limit: int | None = None,
    recreate: bool = False,
) -> int:
    """在 Elasticsearch 中构建 cMedQA2 BM25 索引。"""
    from medrag.retrieval.es_retriever import ESBM25Retriever, INDEX_MAPPINGS

    docs = load_cmedqa2_dataset(limit=limit)
    logger.info("Loaded cMedQA2 docs: %d", len(docs))
    if not docs:
        return 0

    retriever = ESBM25Retriever(
        hosts=settings.es_hosts,
        index_name=settings.es_index_name,
    )

    if recreate and retriever.client.indices.exists(index=settings.es_index_name):
        retriever.client.indices.delete(index=settings.es_index_name)
        logger.info("Deleted existing ES index: %s", settings.es_index_name)

    if not retriever.client.indices.exists(index=settings.es_index_name):
        retriever.client.indices.create(
            index=settings.es_index_name, body=INDEX_MAPPINGS
        )
        logger.info("Created ES index: %s", settings.es_index_name)

    from elasticsearch import helpers

    def _gen_actions():
        for doc in docs:
            yield {
                "_index": settings.es_index_name,
                "_id": doc["id"],
                "_source": {
                    "pk": doc["id"],
                    "department": "cmedqa2",
                    "title": doc["question"][:80],
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "text": doc.get("text", ""),
                },
            }

    success, errors = helpers.bulk(
        retriever.client,
        _gen_actions(),
        chunk_size=128,
        max_retries=3,
        request_timeout=60,
        raise_on_error=False,
    )
    if errors:
        logger.warning("ES indexing errors: %s", errors[:3])
    retriever.client.indices.refresh(index=settings.es_index_name)
    logger.info(
        "ES index build finished: index=%s, docs=%d, errors=%d",
        settings.es_index_name, success, len(errors) if errors else 0,
    )
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="为 cMedQA2 构建 Milvus + ES 索引。")
    parser.add_argument("--milvus-only", action="store_true", help="仅构建 Milvus 索引")
    parser.add_argument("--es-only", action="store_true", help="仅构建 ES 索引")
    parser.add_argument("--limit", type=int, default=None, help="最大文档数")
    parser.add_argument("--recreate", action="store_true", help="删除并重新创建索引")
    args = parser.parse_args()

    do_milvus = not args.es_only
    do_es = not args.milvus_only

    if do_milvus:
        build_milvus_index(limit=args.limit, recreate=args.recreate)
    if do_es:
        build_es_index(limit=args.limit, recreate=args.recreate)


if __name__ == "__main__":
    main()
