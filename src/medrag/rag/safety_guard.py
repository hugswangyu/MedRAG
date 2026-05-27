"""安全卫士：医疗风险分层检测与情境化安全提示注入。"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# 红色信号（立即急诊 — 回答开头插入警告）
# ---------------------------------------------------------------------------

RED_SIGNALS: Dict[str, str] = {
    "胸痛": "胸痛",
    "呼吸困难": "呼吸困难",
    "意识不清": "意识不清",
    "抽搐": "抽搐",
    "大出血": "大出血",
    "休克": "休克",
    "过量服药": "过量服药",
    "自杀": "自杀",
    "咯血": "咯血",
    "呕血": "呕血",
    "严重过敏": "严重过敏",
    "呼吸心跳骤停": "呼吸心跳骤停",
}

# ---------------------------------------------------------------------------
# 黄色信号（尽快就医 — 回答末尾插入提醒）
# ---------------------------------------------------------------------------

YELLOW_SIGNALS: Dict[str, str] = {
    "便血": "便血",
    "黑便": "黑便",
    "高热不退": "高热不退",
    "剧烈腹痛": "剧烈腹痛",
    "孕妇": "孕妇",
    "婴儿": "婴儿",
    "呕血": "呕血",
    "晕厥": "晕厥",
    "视力突然下降": "视力突然下降",
    "突发言语不清": "突发言语不清",
    "一侧肢体无力": "一侧肢体无力",
    "尿潴留": "尿潴留",
    "无尿": "无尿",
}

# ---------------------------------------------------------------------------
# 警示消息
# ---------------------------------------------------------------------------

_RED_WARNING = (
    "你描述的情况可能存在紧急风险，建议立即拨打120或前往最近医院急诊科。"
)

_YELLOW_WARNING = (
    "你描述的情况建议尽快就医（今天或48小时内），不要自行等待观察。"
)

# ---------------------------------------------------------------------------
# 检索质量 → 免责声明映射
# ---------------------------------------------------------------------------

RETRIEVAL_DISCLAIMERS: Dict[str, str] = {
    "high": "以上基于通用医学知识，具体以医生意见为准。",
    "low": "检索到的资料存在差异或不完整，请以医生意见为准。",
    "none": "该问题在知识库中未检索到相关信息，基于通用医学知识提供参考，请务必核实。",
}

_DEFAULT_DISCLAIMER = "以上内容仅用于健康科普和就医参考，不能替代医生面诊。"


def _llm_already_wrote_disclaimer(answer: str) -> bool:
    """检查 LLM 是否已在⑤不确定性说明中写过免责声明。

    若答案中已包含「不确定性说明」标题字样，说明 LLM 已按五层结构
    输出了⑤层，系统不再追加外层免责声明，避免重复。
    """
    return "不确定性说明" in answer or "不确定性" in answer


class SafetyGuard:
    """检测高风险医疗查询并注入分级安全提示。

    用法::

        guard = SafetyGuard()
        risk = guard.detect_risk(query, answer)
        disclaimer = guard.get_retrieval_disclaimer("none")
        safe_answer = guard.append_safety_notice(answer, risk, retrieval_quality="high")
    """

    def detect_risk(self, query: str, answer: str = "") -> Dict:
        """扫描 *query* 和 *answer* 中的风险关键词，分级返回。"""
        combined = f"{query}\n{answer}"
        red_types: List[str] = []
        yellow_types: List[str] = []

        for keyword, label in RED_SIGNALS.items():
            if keyword in combined:
                red_types.append(label)

        for keyword, label in YELLOW_SIGNALS.items():
            if keyword in combined and label not in red_types:
                yellow_types.append(label)

        if red_types:
            return {
                "is_high_risk": True,
                "is_moderate_risk": False,
                "risk_types": red_types,
                "risk_level": "red",
                "safety_message": _RED_WARNING,
            }

        if yellow_types:
            return {
                "is_high_risk": False,
                "is_moderate_risk": True,
                "risk_types": yellow_types,
                "risk_level": "yellow",
                "safety_message": _YELLOW_WARNING,
            }

        return {
            "is_high_risk": False,
            "is_moderate_risk": False,
            "risk_types": [],
            "risk_level": "none",
            "safety_message": "",
        }

    @staticmethod
    def get_retrieval_disclaimer(quality: str) -> str:
        """根据检索质量返回对应的免责声明。"""
        return RETRIEVAL_DISCLAIMERS.get(quality, RETRIEVAL_DISCLAIMERS["high"])

    @staticmethod
    def append_safety_notice(
        answer: str,
        risk_info: Dict,
        retrieval_quality: str = "high",
    ) -> str:
        """根据 *risk_info* 和 *retrieval_quality* 注入分级安全提示。

        - 红色信号 → 在开头添加紧急就医警告。
        - 黄色信号 → 在末尾添加尽快就医提醒。
        - 若 LLM 已输出⑤不确定性说明，不再追加系统免责声明。
        """
        parts: List[str] = []

        # 红色预警：开头插入
        risk_level = risk_info.get("risk_level", "none")
        if risk_level == "red":
            parts.append(risk_info.get("safety_message", _RED_WARNING))

        parts.append(answer.strip())

        # 黄色预警：末尾插入
        if risk_level == "yellow":
            parts.append(risk_info.get("safety_message", _YELLOW_WARNING))

        # 免责声明：若 LLM 已写过⑤则跳过
        if not _llm_already_wrote_disclaimer(answer):
            disclaimer = RETRIEVAL_DISCLAIMERS.get(
                retrieval_quality, _DEFAULT_DISCLAIMER
            )
            parts.append(disclaimer)

        return "\n\n".join(parts)
