"""原生工具包：ToolRegistry + 内置工具注册。

用法::

    from medrag.tools import get_tool_registry

    registry = get_tool_registry()
    name, params = registry.match("阿莫西林儿童用量")
    if name:
        result = registry.execute(name, **params)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .base import BaseTool
from .dosage_calculator import DosageCalculator
from .department_guide import DepartmentGuide
from .normal_range import NormalRangeTool


class ToolRegistry:
    """工具注册表 — 管理所有内置工具的匹配和执行。"""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._register(DosageCalculator())
        self._register(DepartmentGuide())
        self._register(NormalRangeTool())

    def _register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def match(self, query: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
        """依次尝试每个工具，返回首个匹配的 (工具名, 参数)。"""
        for name, tool in self._tools.items():
            params = tool.match(query)
            if params is not None:
                return name, params
        return None, None

    def execute(self, name: str, **params) -> str:
        """执行指定工具。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"工具「{name}」不存在"
        try:
            return tool.execute(**params)
        except Exception as exc:
            return f"工具「{name}」执行失败：{exc}"

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


__all__ = [
    "ToolRegistry",
    "get_tool_registry",
    "BaseTool",
    "DosageCalculator",
    "DepartmentGuide",
    "NormalRangeTool",
]
