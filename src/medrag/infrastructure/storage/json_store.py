"""通用 JSON 文件读写，消除 session / document / credential 重复的 I/O 模式。"""

from __future__ import annotations

import json
import os
from typing import TypeVar

T = TypeVar("T")


class JsonStore:
    """单个 JSON 文件的读写门面。"""

    def __init__(self, file_path: str) -> None:
        self._path = file_path

    @property
    def path(self) -> str:
        return self._path

    def read(self) -> dict | list:
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write(self, data: dict | list) -> None:
        folder = os.path.dirname(self._path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
