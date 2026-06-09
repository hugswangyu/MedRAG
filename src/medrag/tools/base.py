"""工具基类 — 所有原生工具需实现 match/execute 接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseTool(ABC):
    """工具基类。

    子类需实现：
    - match(query) -> Optional[Dict]：判断是否命中，返回参数字典或 None。
    - execute(**params) -> str：执行工具逻辑，返回自然语言结果。
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def match(self, query: str) -> Optional[Dict[str, str]]:
        ...

    @abstractmethod
    def execute(self, **params) -> str:
        ...
