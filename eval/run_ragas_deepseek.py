"""使用 Qwen 作为 judge LLM 运行 RAGAS 测评，并适配中文 Prompt。

用法: DASHSCOPE_API_KEY=sk-xxx uv run python eval/run_ragas_deepseek.py
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

# 用 Qwen 作为 judge LLM（优先用免费额度模型）
os.environ["OPENAI_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")
_MODEL = "qwen3-max"  # 或 qwen-max
llm = llm_factory(
    model=_MODEL,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本地 embedding
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"))

# 配置指标
for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    m.llm = llm
    m.reproducibility = 1
answer_relevancy.embeddings = emb
answer_relevancy.strictness = 1

# 将 RAGAS 默认的英文 Prompt 适配为中文
# 必须先设 llm 再 adapt（adapt 内部需要 llm 做翻译）
for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    m.adapt(language="chinese")

# 修复 ragas 的 endswith(".") bug：中文句子以 。！？ 结尾
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

dataset = Dataset.from_list([
    {
        "question": row["question"],
        "answer": row["answer"],
        "contexts": row["contexts"] or [""],
        "ground_truth": row["ground_truth"],
    }
    for row in rows
])

print(f"Running RAGAS evaluation with {_MODEL} (Chinese-adapted prompts)...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

if hasattr(result, "to_pandas"):
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
else:
    scores = dict(result)

print("\n" + "=" * 50)
print(f"RAGAS Evaluation Results ({_MODEL}, 中文 Prompt)")
print("=" * 50)
for k, v in scores.items():
    label = {"faithfulness": "Faithfulness（忠实度）",
             "answer_relevancy": "Answer Relevancy（答案相关性）",
             "context_precision": "Context Precision（上下文精确度）",
             "context_recall": "Context Recall（上下文召回率）"}.get(k, k)
    val_str = f"{v:.4f}" if isinstance(v, (int, float)) and not (v != v) else str(v)
    print(f"  {label}: {val_str}")

# 保存结果（保留原有字段，增加 deepseek_ragas 字段）
report[f"{_MODEL.replace('-', '_')}_ragas"] = {
    "summary": {k: v for k, v in scores.items() if v == v or k == v},
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nReport updated: {report_path}")
