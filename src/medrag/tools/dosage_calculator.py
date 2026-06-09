"""剂量计算工具 — 根据药品名/年龄/体重计算推荐用量。

数据来源：药品说明书 + 中国国家处方集（儿童版）。
仅供辅助参考，实际用药请遵医嘱。
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from .base import BaseTool

# ---------------------------------------------------------------------------
# 常用药品剂量数据库
# ---------------------------------------------------------------------------

_DRUG_DOSAGES: Dict[str, Dict] = {
    "阿莫西林": {
        "adult": "一次 0.5g，每 6-8 小时一次",
        "pediatric": "每日 20-40mg/kg，分 3 次口服",
        "forms": ["胶囊 0.25g", "颗粒 0.125g/包"],
        "note": "青霉素皮试阴性后使用",
    },
    "头孢克肟": {
        "adult": "一次 100mg，每日 2 次",
        "pediatric": "每次 1.5-3mg/kg，每日 2 次",
        "forms": ["胶囊 100mg", "颗粒 50mg/包"],
        "note": "",
    },
    "头孢拉定": {
        "adult": "一次 0.25-0.5g，每 6 小时一次",
        "pediatric": "每日 25-50mg/kg，分 3-4 次",
        "forms": ["胶囊 0.25g"],
        "note": "",
    },
    "阿奇霉素": {
        "adult": "一次 0.5g，每日 1 次，连服 3 天",
        "pediatric": "每日 10mg/kg，每日 1 次，连服 3 天",
        "forms": ["片剂 0.25g", "颗粒 0.1g/包"],
        "note": "餐前 1 小时或餐后 2 小时服用",
    },
    "布洛芬": {
        "adult": "一次 200-400mg，每日 3-4 次（餐后）",
        "pediatric": "每次 5-10mg/kg，每 6-8 小时一次",
        "forms": ["片剂 0.1g", "混悬液 100mg/5ml"],
        "note": "发热＞38.5℃ 时使用，24 小时内不超过 4 次",
    },
    "对乙酰氨基酚": {
        "adult": "一次 300-500mg，每日 3-4 次",
        "pediatric": "每次 10-15mg/kg，每 4-6 小时一次",
        "forms": ["片剂 0.5g", "混悬液 160mg/5ml"],
        "note": "24 小时内不超过 4 次",
    },
    "蒙脱石散": {
        "adult": "一次 3g，每日 3 次",
        "pediatric": "＜1 岁：每日 1g；1-2 岁：每日 1-2g；＞2 岁：每日 2-3g，分 3 次",
        "forms": ["散剂 3g/包"],
        "note": "空腹服用，首剂可加倍",
    },
    "硝苯地平": {
        "adult": "控释片 30mg，每日 1 次",
        "pediatric": "",
        "forms": ["控释片 30mg"],
        "note": "不可掰开服用",
    },
    "二甲双胍": {
        "adult": "一次 0.5g，每日 2-3 次（餐后）",
        "pediatric": "",
        "forms": ["片剂 0.5g"],
        "note": "从小剂量开始，逐渐加量",
    },
    "氨氯地平": {
        "adult": "一次 5mg，每日 1 次",
        "pediatric": "",
        "forms": ["片剂 5mg"],
        "note": "最大可增至 10mg/日",
    },
    "阿托伐他汀": {
        "adult": "一次 10-20mg，每日 1 次",
        "pediatric": "",
        "forms": ["片剂 10mg", "片剂 20mg"],
        "note": "晚间服用",
    },
}

# 别名映射
_ALIASES: Dict[str, str] = {
    "阿莫仙": "阿莫西林",
    "再林": "阿莫西林",
    "世福素": "头孢克肟",
    "希舒美": "阿奇霉素",
    "美林": "布洛芬",
    "泰诺林": "对乙酰氨基酚",
    "必奇": "蒙脱石散",
    "拜新同": "硝苯地平",
    "格华止": "二甲双胍",
    "络活喜": "氨氯地平",
    "立普妥": "阿托伐他汀",
}

# 年龄/体重提取
_AGE_PATTERN = re.compile(r"(\d+)\s*岁")
_WEIGHT_PATTERN = re.compile(r"(\d+)\s*kg|公斤|斤")
_MONTH_PATTERN = re.compile(r"(\d+)\s*个月")


class DosageCalculator(BaseTool):
    name = "剂量计算"
    description = "查询常见药品的成人及儿童推荐剂量"

    def match(self, query: str) -> Optional[Dict[str, str]]:
        if not any(kw in query for kw in ["剂量", "用量", "吃多少", "吃几", "怎么吃", "用法"]):
            return None
        drug = self._find_drug(query)
        if drug is None:
            return None
        params = {"drug": drug}
        m = _AGE_PATTERN.search(query)
        if m:
            params["age"] = m.group(1)
        m = _WEIGHT_PATTERN.search(query)
        if m:
            params["weight"] = m.group(1)
        m = _MONTH_PATTERN.search(query)
        if m:
            params["age"] = str(int(m.group(1)) / 12)
        return params

    def execute(self, drug: str, age: str = "", weight: str = "") -> str:
        info = _DRUG_DOSAGES.get(drug)
        if not info:
            return f"暂未收录「{drug}」的剂量信息"

        is_child = self._is_child(age, weight)
        dose = info["pediatric"] if is_child and info["pediatric"] else info["adult"]

        parts = [f"【{drug}】推荐用量"]
        if is_child and info["pediatric"]:
            parts.append(f"儿童（{self._build_desc(age, weight)}）：{dose}")
        else:
            if age or weight:
                parts.append(f"成人（{self._build_desc(age, weight)}）：{dose}")
            else:
                parts.append(f"成人：{dose}")
        if info["forms"]:
            parts.append(f"常见剂型：{' / '.join(info['forms'])}")
        if info["note"]:
            parts.append(f"注意事项：{info['note']}")
        parts.append("（以上为参考剂量，具体请遵医嘱）")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _find_drug(query: str) -> Optional[str]:
        """在 query 中查找已知药品。优先匹配更长的名字。"""
        candidates = []
        for name in _DRUG_DOSAGES:
            if name in query:
                candidates.append(name)
        for alias, real in _ALIASES.items():
            if alias in query and real not in candidates:
                candidates.append(real)
        if not candidates:
            return None
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    @staticmethod
    def _is_child(age: str, weight: str) -> bool:
        if age:
            try:
                return float(age) < 15
            except ValueError:
                pass
        return False

    @staticmethod
    def _build_desc(age: str, weight: str) -> str:
        parts = []
        if age:
            age_f = float(age)
            parts.append(f"{'%d' % age_f if age_f == int(age_f) else '%.1f' % age_f}岁")
        if weight:
            parts.append(f"{weight}kg")
        return "、".join(parts) if parts else "成人"
