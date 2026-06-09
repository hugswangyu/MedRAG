"""验证 RAGAS 评估脚本内部逻辑：度量计算、关键词命中、报告输出。"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保 eval/ 目录可导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eval.run_ragas_eval import (
    _contexts_from_result,
    _custom_metrics,
    _keyword_hit_rate,
    _load_jsonl,
    _write_report,
)


# ============================================================================
# 数据加载
# ============================================================================


def test_load_jsonl(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(
        '{"id":"1","question":"q1"}\n{"id":"2","question":"q2"}\n',
        encoding="utf-8",
    )
    cases = _load_jsonl(f)
    assert len(cases) == 2
    assert cases[0]["id"] == "1"
    assert cases[1]["id"] == "2"


def test_load_jsonl_skips_empty_lines(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text('{"id":"1"}\n\n\n{"id":"2"}\n', encoding="utf-8")
    cases = _load_jsonl(f)
    assert len(cases) == 2


# ============================================================================
# 上下文提取
# ============================================================================


class TestContextsFromResult:
    def test_empty_result(self):
        assert _contexts_from_result({}) == []

    def test_kg_results(self):
        result = {"kg_results": [{"answer": "感冒是病毒感染"}]}
        ctx = _contexts_from_result(result)
        assert "感冒是病毒感染" in ctx

    def test_qa_results(self):
        result = {"qa_results": [{"text": "多喝水休息"}]}
        ctx = _contexts_from_result(result)
        assert "多喝水休息" in ctx

    def test_all_sources_combined(self):
        result = {
            "kg_results": [{"answer": "kg1"}],
            "qa_results": [{"text": "qa1"}],
            "case_results": [{"answer": "case1"}],
        }
        ctx = _contexts_from_result(result)
        assert len(ctx) == 3

    def test_list_value(self):
        result = {"kg_results": [{"answer": ["项1", "项2"]}]}
        ctx = _contexts_from_result(result)
        assert "项1、项2" in ctx


# ============================================================================
# 关键词命中率
# ============================================================================


class TestKeywordHitRate:
    def test_empty_keywords(self):
        assert _keyword_hit_rate([], ["some context"], "answer") == 1.0

    def test_all_hit(self):
        rate = _keyword_hit_rate(["头痛", "发热"], ["头痛相关内容"], "患者有发热症状")
        assert rate == 1.0

    def test_partial_hit(self):
        rate = _keyword_hit_rate(["头痛", "发热", "咳嗽"], ["头痛相关内容"], "患者有发热症状")
        assert rate == pytest.approx(2 / 3)

    def test_no_hit(self):
        rate = _keyword_hit_rate(["头痛"], ["腹痛内容"], "腹痛患者")
        assert rate == 0.0


# ============================================================================
# 自定义指标计算
# ============================================================================


class TestCustomMetrics:
    def test_route_accuracy(self):
        rows = [
            {"expected_route": "disease_fact", "route": {"query_type": "disease_fact"}},
            {"expected_route": "medication", "route": {"query_type": "medication"}},
            {"expected_route": "diet", "route": {"query_type": "medication"}},  # wrong
        ]
        metrics = _custom_metrics(rows)
        assert metrics["route_accuracy"] == pytest.approx(2 / 3)

    def test_keyword_hit_rate(self):
        rows = [
            {
                "expected_context_keywords": ["头痛"],
                "contexts": ["头痛内容"],
                "answer": "回答",
            },
        ]
        metrics = _custom_metrics(rows)
        assert metrics["context_keyword_hit_rate"] == 1.0

    def test_empty_context_rate(self):
        rows = [
            {"contexts": ["内容"], "answer": "回答"},
            {"contexts": [], "answer": "回答"},
        ]
        metrics = _custom_metrics(rows)
        assert metrics["empty_context_rate"] == 0.5

    def test_safety_hit_rate(self):
        """验证高危问题中安全提醒命中率。"""
        rows = [
            {
                "id": "safety_001",
                "answer": "胸痛请立即拨打120",
                "risk_info": {"is_high_risk": True, "is_moderate_risk": False},
                "contexts": [],
            },
            {
                "id": "safety_002",
                "answer": "建议尽快就医",
                "risk_info": {"is_high_risk": True, "is_moderate_risk": False},
                "contexts": [],
            },
            {
                "id": "normal_001",
                "answer": "多喝水休息",
                "risk_info": {"is_high_risk": False, "is_moderate_risk": False},
                "contexts": [],
            },
        ]
        metrics = _custom_metrics(rows)
        assert metrics["safety_hit_rate"] == 1.0  # safety_001 和 safety_002 都命中

    def test_no_expected_route_skips_accuracy(self):
        rows = [
            {"route": {"query_type": "disease_fact"}},  # 没有 expected_route
        ]
        metrics = _custom_metrics(rows)
        assert metrics["route_accuracy"] is None


# ============================================================================
# 报告输出
# ============================================================================


def test_write_report(tmp_path):
    report_path = tmp_path / "subdir" / "report.json"
    payload = {"case_count": 3, "custom_metrics": {"route_accuracy": 0.8}}
    _write_report(report_path, payload)

    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["case_count"] == 3
    assert loaded["custom_metrics"]["route_accuracy"] == 0.8


# ============================================================================
# 集成：完整评估流水线（mock service）
# ============================================================================


class TestFullEvalPipeline:
    """mock 掉 MedicalChatService 后验证评估全流程可跑通。"""

    @patch("eval.run_ragas_eval._run_pipeline")
    def test_main_creates_report(self, mock_run_pipeline, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create a small golden file
        golden = tmp_path / "golden_cases.jsonl"
        golden.write_text(
            '{"id":"1","question":"q1","ground_truth":"gt1","expected_route":"disease_fact"}\n',
            encoding="utf-8",
        )

        # Mock the pipeline to return controlled results
        mock_run_pipeline.return_value = [
            {
                "id": "1",
                "question": "q1",
                "answer": "答案是A",
                "ground_truth": "gt1",
                "contexts": ["相关上下文"],
                "route": {"query_type": "disease_fact"},
                "expected_route": "disease_fact",
                "expected_context_keywords": ["关键词"],
                "risk_info": {"is_high_risk": False, "is_moderate_risk": False},
            }
        ]

        # Run eval
        from eval.run_ragas_eval import main
        import sys

        monkeypatch.setattr(
            sys, "argv",
            ["run_ragas_eval.py", "--golden", str(golden), "--out",
             str(tmp_path / "report.json"), "--skip-ragas"],
        )
        main()

        # Verify report was generated
        report_file = tmp_path / "report.json"
        assert report_file.exists()
        report = json.loads(report_file.read_text(encoding="utf-8"))
        assert report["case_count"] == 1
        assert report["custom_metrics"]["route_accuracy"] == 1.0
        assert report["ragas"]["skipped"] is True
