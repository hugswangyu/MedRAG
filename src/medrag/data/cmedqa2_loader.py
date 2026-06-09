"""cMedQA2 中文医疗问答数据集加载器（HuggingFace 源）。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

HF_DATASET_PATH = "zirui3/cMedQA2-instructions"


def load_cmedqa2_dataset(
    split: str = "train",
    limit: Optional[int] = None,
) -> List[Dict]:
    """从 HuggingFace 加载 cMedQA2 数据集，返回统一字典列表。

    返回格式::

        {
            "id": "cmedqa2-{question_id}-{answer_id}",
            "question": "...",
            "answer": "...",
            "source": "cmedqa2",
        }
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed; run: pip install datasets")
        return []

    ds = load_dataset(HF_DATASET_PATH, split=split)
    records: List[Dict] = []
    seen = set()

    total = len(ds) if limit is None else min(len(ds), limit)
    pbar = tqdm(total=total, unit="row", desc="cMedQA2", ncols=80)

    for i, row in enumerate(ds):
        if limit is not None and len(records) >= limit:
            break

        question = (row.get("question") or "").strip()
        answer = (row.get("anwser") or row.get("answer") or "").strip()

        if len(question) < 2 or len(answer) < 5:
            pbar.update(1)
            continue

        qid = row.get("question_id", "")
        aid = row.get("answer_id", "")

        dedup_key = (question, answer)
        if dedup_key in seen:
            pbar.update(1)
            continue
        seen.add(dedup_key)

        record = {
            "id": f"cmedqa2-{qid}-{aid}",
            "question": question,
            "answer": answer,
            "source": "cmedqa2",
        }
        record["text"] = f"问题：{question}\n回答：{answer}"
        records.append(record)
        pbar.update(1)

    pbar.close()
    logger.info(
        "Loaded %d / %d cMedQA2 records from split=%s",
        len(records),
        i + 1,
        split,
    )
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records = load_cmedqa2_dataset(limit=5)
    for r in records:
        print(r["id"])
        print(f"  Q: {r['question'][:80]}")
        print(f"  A: {r['answer'][:80]}")
        print()
