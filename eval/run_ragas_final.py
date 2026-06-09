"""直接调用 RAGAS 评测，使用 Qwen 作为 judge LLM + 本地 embedding。

用法: DASHSCOPE_API_KEY=sk-xxx uv run python eval/run_ragas_final.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 配置 Qwen 作为 judge LLM
os.environ["OPENAI_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")
llm = llm_factory(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本地 embedding 模型
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"))

for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    m.llm = llm
    m.reproducibility = 1  # Qwen 不支持 n > 1
answer_relevancy.embeddings = emb
answer_relevancy.strictness = 1

# 修复 ragas 的 endswith(".") bug：中文句子以 。！？ 结尾，而非英文 .
_create_orig = faithfulness._create_statements_prompt
def _create_patched(self, row):
    answer, question = row["answer"], row["question"]
    sentences = self.sentence_segmenter.segment(answer)
    sentences = [
        s for s in sentences
        if s.strip().endswith((".", "。", "！", "？"))
    ]
    sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
    return self.statement_prompt.format(
        question=question, answer=answer, sentences=sentences
    )
faithfulness._create_statements_prompt = _create_patched.__get__(faithfulness)

# 加载 golden cases
golden_path = _root / "eval" / "golden_cases.jsonl"
cases = []
with open(golden_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            cases.append(json.loads(line))

print(f"Loaded {len(cases)} cases")

# 从已生成的报告中读取 answers 和 contexts
report_path = _root / "eval" / "reports" / "ragas_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
else:
    print("Error: run_ragas_eval.py must be run first to generate answers")
    sys.exit(1)

# 构造 RAGAS 数据集
dataset = Dataset.from_list([
    {
        "question": row["question"],
        "answer": row["answer"],
        "contexts": row["contexts"] or [""],
        "ground_truth": row["ground_truth"],
    }
    for row in rows
])

# 运行 RAGAS 评估
print("Running RAGAS evaluation with Qwen judge...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

if hasattr(result, "to_pandas"):
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
else:
    scores = dict(result)

# 输出
print("\n" + "=" * 50)
print("RAGAS Evaluation Results (30 cases)")
print("=" * 50)
for k, v in scores.items():
    label = {"faithfulness": "Faithfulness（忠实度）",
             "answer_relevancy": "Answer Relevancy（答案相关性）",
             "context_precision": "Context Precision（上下文精确度）",
             "context_recall": "Context Recall（上下文召回率）"}.get(k, k)
    val_str = f"{v:.4f}" if isinstance(v, (int, float)) and not (v != v) else str(v)
    print(f"  {label}: {val_str}")

# 保存到报告
report["ragas"] = {"summary": {k: v for k, v in scores.items() if v == v or k == v}}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nReport updated: {report_path}")
