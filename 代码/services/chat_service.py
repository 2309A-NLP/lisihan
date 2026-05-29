# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""聊天业务逻辑层。

负责：
- 角色和会话验证
- 数据获取和组装
- 调用 RAG 生成回复
- 保存结果到三层存储
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import HTTPException

from app.core.chat_context import (
    _build_context,
    _build_response_context,
    _memory_for_conversation,
)
from app.core.memory import memory
from app.core.rag import rag_system
from app.models import SessionLocal
from app.models.chat import ChatMessage, Conversation
from app.schemas.chat import ChatRequest, ChatResult
from app.services.role_service import RoleService
from app.utils.chat_utils import _conversation_title, _is_placeholder_title, _message_timestamp

logger = logging.getLogger(__name__)


def _get_relevant_long_memory(
    user_id: int, role_id: int, message: str, top_k: int, use_embedding: bool
) -> List[Dict[str, Any]]:
    """按开关选择长期记忆策略：embedding 相关检索优先，失败后回退到时间倒序。"""
    if top_k <= 0:
        return []

    if use_embedding:
        query_embedding = rag_system.get_embedding(message)
        if query_embedding:
            results = memory.long.search(user_id, role_id, query_embedding, limit=top_k)
            if results:
                return results

    return memory.long.get(user_id, role_id, limit=top_k)


def _save_chat_result(
    session,
    request: ChatRequest,
    conversation: Conversation,
    role_name: str,
    response: str
) -> ChatResult:
    """保存 MySQL 会话消息，并更新会话标题和时间。"""
    now = int(time.time())
    conversation_id = conversation.id

    # 保存到 MySQL
    session.add(ChatMessage(
        conversation_id=conversation_id,
        user_id=request.user_id,
        role_id=request.role_id,
        sender="user",
        content=request.message,
    ))
    session.add(ChatMessage(
        conversation_id=conversation_id,
        user_id=request.user_id,
        role_id=request.role_id,
        sender="role",
        content=response,
    ))

    # 更新会话标题
    if _is_placeholder_title(conversation.title, role_name):
        conversation.title = _conversation_title(request.message, role_name)
    conversation.updated_at = datetime.utcnow()
    session.commit()

    # 准备记忆数据
    user_message = {
        "sender": "user",
        "content": request.message,
        "timestamp": now,
        "conversation_id": conversation_id,
    }
    role_message = {
        "sender": "role",
        "content": response,
        "timestamp": int(time.time()),
        "conversation_id": conversation_id,
    }

    # 保存到短期记忆（Redis）
    memory.short.save(request.user_id, request.role_id, user_message)
    memory.short.save(request.user_id, request.role_id, role_message)

    # 保存到长期记忆（Milvus）
    memory.long.save(request.user_id, request.role_id, conversation_id, user_message)
    memory.long.save(request.user_id, request.role_id, conversation_id, role_message)

    return ChatResult(response=response, conversation_id=conversation_id)


def run_chat_sync(request: ChatRequest) -> ChatResult:
    """执行完整聊天流程。

    流程：
    1. 验证角色和会话权限
    2. 获取历史对话、短期记忆、长期记忆
    3. 构建并裁剪上下文
    4. 检索知识库
    5. 调用 LLM 生成回复
    6. 保存结果到数据库和缓存
    """
    from app.core.chat_context import _build_response_context
    from app.models import SessionLocal

    SessionLocal()  # 确保导入
    session = SessionLocal()
    try:
        role_service = RoleService()
        role = role_service.get_role_by_id(request.role_id)
        if not role:
            raise HTTPException(status_code=404, detail="Role not found")

        role_template = role_service.get_role_template(request.role_id)
        if not role_template:
            raise HTTPException(status_code=404, detail="Role template not found")

        # 获取或创建会话
        conversation = session.get(Conversation, request.conversation_id) if request.conversation_id else None
        if conversation and conversation.user_id != request.user_id:
            raise HTTPException(status_code=403, detail="No permission for this conversation")
        if conversation and conversation.role_id != request.role_id:
            raise HTTPException(status_code=400, detail="Conversation role mismatch")

        if not conversation:
            conversation = Conversation(
                user_id=request.user_id,
                role_id=request.role_id,
                title=_conversation_title(request.message, role.name),
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)

        # 获取历史对话（MySQL）
        history_rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation.id)
            .order_by(ChatMessage.id.desc())
            .limit(request.history_limit)
            .all()
        )
        history = [
            {"sender": item.sender, "content": item.content, "timestamp": _message_timestamp(item)}
            for item in reversed(history_rows)
        ]

        # 获取短期记忆（Redis）
        short_memory = _memory_for_conversation(
            memory.short.get(request.user_id, request.role_id),
            conversation.id,
        )

        # 获取长期记忆（Milvus）
        long_memory = _memory_for_conversation(
            _get_relevant_long_memory(
                request.user_id,
                request.role_id,
                request.message,
                request.memory_top_k,
                request.use_embedding_memory,
            ),
            conversation.id,
        )

        # 构建并裁剪上下文
        context_bundle = _build_context(history, short_memory, long_memory, request.context_budget)

        # 构建响应上下文（包含知识库）
        response_context = _build_response_context(
            request, context_bundle, rag_system
        )

        # 生成回复
        effective_request = request.model_copy(update={"conversation_id": conversation.id})
        response = rag_system.generate_response(
            effective_request.message,
            response_context,
            role_template,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            history=context_bundle["history"],
            short_memory=context_bundle["short_memory"],
            long_memory=context_bundle["long_memory"],
            combined_context=context_bundle["combined"],
        )

        # 保存结果
        return _save_chat_result(session, effective_request, conversation, role.name, response)

    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        logger.exception("聊天流程异常", exc_info=exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")
    finally:
        session.close()
