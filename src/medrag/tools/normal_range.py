"""正常值范围查询工具 — 常见检验/检查指标的正常参考范围。

数据来源：《临床检验诊断学》中国标准、全国临床检验操作规程。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .base import BaseTool

# ---------------------------------------------------------------------------
# 正常值数据库
# ---------------------------------------------------------------------------

_NORMAL_RANGES: List[Dict] = [
    # === 血常规 ===
    {"name": "白细胞", "aliases": ["WBC", "白血球"], "range": "3.5-9.5", "unit": "×10⁹/L"},
    {"name": "红细胞", "aliases": ["RBC", "红血球"], "range": "男4.3-5.8 / 女3.8-5.1", "unit": "×10¹²/L"},
    {"name": "血红蛋白", "aliases": ["Hb", "HGB", "血色素"], "range": "男130-175 / 女115-150", "unit": "g/L"},
    {"name": "血小板", "aliases": ["PLT"], "range": "125-350", "unit": "×10⁹/L"},
    {"name": "中性粒细胞百分比", "aliases": ["NEUT%"], "range": "40-75", "unit": "%"},
    {"name": "淋巴细胞百分比", "aliases": ["LYMPH%"], "range": "20-50", "unit": "%"},
    {"name": "中性粒细胞绝对值", "aliases": ["NEUT#"], "range": "1.8-6.3", "unit": "×10⁹/L"},
    {"name": "C反应蛋白", "aliases": ["CRP"], "range": "＜10", "unit": "mg/L"},
    # === 生化 ===
    {"name": "空腹血糖", "aliases": ["GLU", "血糖"], "range": "3.9-6.1", "unit": "mmol/L"},
    {"name": "糖化血红蛋白", "aliases": ["HbA1c"], "range": "4.0-6.0", "unit": "%"},
    {"name": "总胆固醇", "aliases": ["TC", "CHOL"], "range": "＜5.2", "unit": "mmol/L"},
    {"name": "甘油三酯", "aliases": ["TG"], "range": "＜1.7", "unit": "mmol/L"},
    {"name": "高密度脂蛋白", "aliases": ["HDL-C"], "range": "≥1.0", "unit": "mmol/L"},
    {"name": "低密度脂蛋白", "aliases": ["LDL-C"], "range": "＜3.4", "unit": "mmol/L"},
    {"name": "谷丙转氨酶", "aliases": ["ALT", "GPT"], "range": "＜40", "unit": "U/L"},
    {"name": "谷草转氨酶", "aliases": ["AST", "GOT"], "range": "＜40", "unit": "U/L"},
    {"name": "肌酐", "aliases": ["Cr", "CREA"], "range": "男44-104 / 女44-97", "unit": "μmol/L"},
    {"name": "尿素", "aliases": ["UREA", "BUN"], "range": "2.9-8.2", "unit": "mmol/L"},
    {"name": "尿酸", "aliases": ["UA"], "range": "男208-428 / 女155-357", "unit": "μmol/L"},
    # === 电解质 ===
    {"name": "钾", "aliases": ["K+"], "range": "3.5-5.3", "unit": "mmol/L"},
    {"name": "钠", "aliases": ["Na+"], "range": "137-147", "unit": "mmol/L"},
    {"name": "氯", "aliases": ["Cl-"], "range": "99-110", "unit": "mmol/L"},
    {"name": "钙", "aliases": ["Ca2+"], "range": "2.11-2.52", "unit": "mmol/L"},
    # === 凝血 ===
    {"name": "凝血酶原时间", "aliases": ["PT"], "range": "11-14", "unit": "秒"},
    {"name": "活化部分凝血活酶时间", "aliases": ["APTT"], "range": "28-42", "unit": "秒"},
    # === 尿常规 ===
    {"name": "尿蛋白", "aliases": ["PRO"], "range": "阴性（-）", "unit": ""},
    {"name": "尿糖", "aliases": ["GLU-U"], "range": "阴性（-）", "unit": ""},
    {"name": "尿潜血", "aliases": ["BLD", "ERY"], "range": "阴性（-）", "unit": ""},
    {"name": "尿白细胞", "aliases": ["LEU"], "range": "阴性（-）或＜5", "unit": "/HPF"},
]

# 查询关键词（触发条件）
_QUERY_KEYWORDS = [
    "正常值", "正常范围", "参考范围", "参考值",
    "多少正常", "指标", "检查结果",
]

# 中文数字转数值（用于解释高于/低于正常）
_NUM_PATTERN = re.compile(r"(\d+\.?\d*)")


class NormalRangeTool(BaseTool):
    name = "正常值查询"
    description = "查询常见检查指标的正常参考范围"

    def match(self, query: str) -> Optional[Dict[str, str]]:
        if not any(kw in query for kw in _QUERY_KEYWORDS):
            return None
        test_name = self._find_test(query)
        if test_name is None:
            return None
        return {"test": test_name, "query": query}

    def execute(self, test: str = "", query: str = "") -> str:
        for item in _NORMAL_RANGES:
            if item["name"] == test:
                return self._format_result(item, query)
        return f"暂未收录「{test}」的参考范围"

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _find_test(query: str) -> Optional[str]:
        """在 query 中匹配检验项目名或别名，返回标准名。"""
        for item in _NORMAL_RANGES:
            if item["name"] in query:
                return item["name"]
            for alias in item["aliases"]:
                if alias in query:
                    return item["name"]
        return None

    @staticmethod
    def _format_result(item: Dict, query: str) -> str:
        alias_str = "、".join(item["aliases"][:2])
        parts = [
            f"【{item['name']}】",
            f"正常参考范围：{item['range']} {item['unit']}",
        ]
        if alias_str and alias_str != item["name"]:
            parts.insert(1, f"简称：{alias_str}")

        # 如果有数值，尝试解释高于/低于正常
        m = _NUM_PATTERN.search(query)
        if m:
            parts.append("")
            parts.append(f"您的检测值：{m.group(1)}{item['unit']}")
            parts.append("（此数值仅供参考，请结合临床症状由医生综合判断）")

        parts.append("")
        parts.append("（不同实验室或检测方法可能导致参考范围略有差异）")
        return "\n".join(parts)
