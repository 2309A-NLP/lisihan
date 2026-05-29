# -*- coding: utf-8 -*-
import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.core.rag import rag_system
from app.models import SessionLocal
from app.models.chat import ChatMessage, Conversation
from app.models.user import User

# 下载界面的API
router = APIRouter()

def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "download"

def _response(content: str, filename: str, media_type: str) -> Response:
    return Response(
        content=content.encode("utf-8-sig"),
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@router.get("/conversation/{conversation_id}")
def export_conversation(
    conversation_id: int,
    format: str = Query("json", pattern="^(json|csv|txt)$"),
):
    """导出对话记录，支持 json/csv/txt。"""
    session = SessionLocal()
    try:
        conversation = session.get(Conversation, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        user = session.get(User, conversation.user_id)
        messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )

        payload = {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "user_id": conversation.user_id,
                "username": user.username if user else "未知",
                "role_id": conversation.role_id,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
            },
            "messages": [
                {
                    "id": item.id,
                    "sender": item.sender,
                    "content": item.content,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                }
                for item in messages
            ],
            "exported_at": datetime.utcnow().isoformat(),
        }
        base_name = _safe_filename(f"conversation_{conversation_id}_{conversation.title}")

        if format == "json":
            return _response(
                json.dumps(payload, ensure_ascii=False, indent=2),
                f"{base_name}.json",
                "application/json",
            )

        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["message_id", "sender", "content", "created_at"])
            for item in payload["messages"]:
                writer.writerow([item["id"], item["sender"], item["content"], item["created_at"]])
            return _response(output.getvalue(), f"{base_name}.csv", "text/csv")

        lines = [
            f"对话 ID: {payload['conversation']['id']}",
            f"标题: {payload['conversation']['title']}",
            f"用户: {payload['conversation']['username']} ({payload['conversation']['user_id']})",
            f"角色 ID: {payload['conversation']['role_id']}",
            f"导出时间: {payload['exported_at']}",
            "",
        ]
        for item in payload["messages"]:
            lines.append(f"[{item['created_at'] or '--'}] {item['sender']}:")
            lines.append(item["content"])
            lines.append("")
        return _response("\n".join(lines), f"{base_name}.txt", "text/plain")
    finally:
        session.close()

@router.get("/knowledge/{knowledge_base_id}")
def export_knowledge(
    knowledge_base_id: int,
    format: str = Query("json", pattern="^(json|txt)$"),
):
    """按角色/知识库 ID 导出知识片段，支持 json/txt。"""
    items = [
        {
            "index": index,
            "content": item.get("content", ""),
            "role_ids": item.get("role_ids", []),
            "source_file": item.get("source_file", "未知来源"),
        }
        for index, item in enumerate(getattr(rag_system, "knowledge_items", []), start=1)
        if knowledge_base_id in item.get("role_ids", [])
    ]

    if not items:
        raise HTTPException(status_code=404, detail="未找到该知识库或角色对应的知识片段")

    payload = {
        "knowledge_base_id": knowledge_base_id,
        "total": len(items),
        "items": items,
        "exported_at": datetime.utcnow().isoformat(),
    }
    base_name = _safe_filename(f"knowledge_{knowledge_base_id}")

    if format == "json":
        return _response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            f"{base_name}.json",
            "application/json",
        )

    lines = [
        f"知识库/角色 ID: {knowledge_base_id}",
        f"片段数量: {len(items)}",
        f"导出时间: {payload['exported_at']}",
        "",
    ]
    for item in items:
        lines.append(f"#{item['index']} 来源: {item['source_file']} | 角色: {','.join(map(str, item['role_ids'])) or '--'}")
        lines.append(item["content"])
        lines.append("")
    return _response("\n".join(lines), f"{base_name}.txt", "text/plain")
