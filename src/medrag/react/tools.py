"""ReAct 工具定义 — 包装检索器与原生工具供 LLM 调用。

每个 ``ReActTool`` 包含名称、描述、参数列表和可调用方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolParam:
    """工具参数定义。"""
    name: str
    description: str
    type: str = "string"


@dataclass
class ReActTool:
    """ReAct 可调用工具。

    Attributes:
        name: 工具名（供 LLM 引用）。
        description: 用途说明（生成 LLM 提示词）。
        parameters: 参数列表。
        executor: 实际执行函数，接收 **(param_name → value)**。
    """
    name: str
    description: str
    parameters: List[ToolParam] = field(default_factory=list)
    executor: Optional[Callable] = None

    def to_prompt_block(self) -> str:
        """格式化为 LLM 提示词中的工具描述块。"""
        params_str = ", ".join(
            f"{p.name}: {p.type} — {p.description}" for p in self.parameters
        ) if self.parameters else "无参数"
        return f"{self.name}({params_str})\n   {self.description}"

    def call(self, **kwargs) -> str:
        """执行工具，返回字符串结果。"""
        if self.executor is None:
            return f"工具「{self.name}」不可用"
        try:
            result = self.executor(**kwargs)
            if result is None:
                return "未找到相关信息"
            return str(result)
        except Exception as exc:
            return f"工具执行出错：{exc}"
