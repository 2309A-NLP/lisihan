# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""聊天模块的工具函数。"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.chat import ChatMessage, Conversation


def _message_timestamp(message: ChatMessage) -> int:
    """把消息创建时间转换为 Unix 时间戳。"""
    created_at = message.created_at or datetime.utcnow()
    return int(created_at.timestamp())


def _conversation_title(message: str, role_name: str) -> str:
    """根据用户首条消息生成简短会话标题。"""
    clean = " ".join((message or "").strip().split())
    if not clean:
        return f"{role_name} new chat"
    return clean[:28]


def _is_placeholder_title(title: str, role_name: str) -> bool:
    """判断标题是否仍是默认占位标题。"""
    normalized = (title or "").strip()
    return normalized in {
        "",
        f"{role_name} new chat",
        f"{role_name}的新对话",
        "新对话",
    }


def _latest_user_question(session, conversation_id: int) -> Optional[str]:
    """查询指定会话中最近一条用户消息。"""
    message = (
        session.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id, ChatMessage.sender == "user")
        .order_by(ChatMessage.id.desc())
        .first()
    )
    return message.content if message else None


def _serialize_conversation(
    conversation: Conversation, last_question: Optional[str] = None
):
    """把 Conversation ORM 对象转换成 API 返回模型。"""
    from app.schemas.chat import ConversationResponse

    updated_at = conversation.updated_at or conversation.created_at
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        role_id=conversation.role_id,
        title=conversation.title,
        updated_at=updated_at.isoformat() if updated_at else None,
        last_question=last_question,
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    """构造 Server-Sent Events 格式的数据块。"""
    import json
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _split_text(text: str, chunk_size: int = 24) -> List[str]:
    """把已有完整文本拆成 SSE 小块。"""
    text = text or ""
    return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size)] or [""]