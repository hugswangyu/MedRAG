"""会话持久化：JSON 文件存储。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from medrag.config.settings import settings
from medrag.infrastructure.storage import JsonStore

from .schemas import SessionSummary, SessionMessage, SessionDetailResponse

logger = logging.getLogger(__name__)

_store = JsonStore(str(settings.sessions_path))


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def add_message(
    session_id: str,
    msg_type: str,
    content: str,
    rag_trace: Optional[dict] = None,
) -> None:
    """向会话追加一条消息。"""
    sessions = _store.read()
    entry = {
        "type": msg_type,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if rag_trace:
        entry["rag_trace"] = rag_trace
    sessions.setdefault(session_id, []).append(entry)
    _store.write(sessions)


def get_sessions() -> List[SessionSummary]:
    sessions = _store.read()
    result = []
    for sid, msgs in sessions.items():
        updated = msgs[-1]["timestamp"] if msgs else ""
        result.append(
            SessionSummary(
                session_id=sid,
                message_count=len(msgs),
                updated_at=updated,
            )
        )
    result.sort(key=lambda s: s.updated_at, reverse=True)
    return result


def get_session(session_id: str) -> Optional[SessionDetailResponse]:
    sessions = _store.read()
    msgs = sessions.get(session_id)
    if msgs is None:
        return None
    return SessionDetailResponse(
        session_id=session_id,
        messages=[
            SessionMessage(
                type=m["type"],
                content=m["content"],
                rag_trace=m.get("rag_trace"),
            )
            for m in msgs
        ],
    )


def delete_session(session_id: str) -> bool:
    sessions = _store.read()
    if session_id not in sessions:
        return False
    del sessions[session_id]
    _store.write(sessions)
    return True
