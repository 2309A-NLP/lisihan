# -*- coding: utf-8 -*-
"""
管理员API模块
================
需要管理员权限的接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta

from app.models import db
from app.models.user import User
from app.models.chat import Conversation, ChatMessage
from app.services.user_service import UserService
from app.core.auth import require_admin

# 创建 router 实例
router = APIRouter()


# ========== Pydantic 模型 ==========
# 用于数据验证和设置管理的工具
class UserStatusUpdate(BaseModel):
    """用户状态更新请求"""
    is_active: bool

class UserAdminUpdate(BaseModel):
    """管理员权限更新请求"""
    is_admin: bool

class UserListItem(BaseModel):
    """用户列表项响应"""
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    login_count: int
    last_login: Optional[str] = None
    created_at: Optional[str] = None

class SystemStatistics(BaseModel):
    """系统统计响应"""
    total_users: int
    active_users: int
    admin_users: int
    total_conversations: int
    total_messages: int
    total_knowledge_segments: int
    today_conversations: int
    today_messages: int

class ConversationListItem(BaseModel):
    """对话列表项响应"""
    id: int
    user_id: int
    username: str
    role_id: int
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ========== 用户管理 API ==========
@router.get("/users", response_model=List[UserListItem])
def list_users(
        skip: int = Query(0, ge=0, description="跳过数量"),
        limit: int = Query(20, ge=1, le=100, description="返回数量"),
        current_user: User = Depends(require_admin)
):
    """
    获取用户列表（需要管理员权限）
    """
    try:
        user_service = UserService()
        users = user_service.get_all_users(skip=skip, limit=limit)

        return [
            UserListItem(
                id=user.id,
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                is_admin=user.is_admin,
                login_count=user.login_count,
                last_login=user.last_login.isoformat() if user.last_login else None,
                created_at=user.created_at.isoformat() if user.created_at else None
            )
            for user in users
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")

@router.put("/users/{user_id}/status")
def update_user_status(
        user_id: int,
        request: UserStatusUpdate,
        current_user: User = Depends(require_admin)
):
    """
    启用/禁用用户账号（需要管理员权限）
    """
    try:
        # 不能禁用自己
        if user_id == current_user.id:
            raise HTTPException(status_code=400, detail="不能禁用或启用自己的账号")

        user_service = UserService()
        user = user_service.set_user_active(user_id, request.is_active)

        return {
            "success": True,
            "message": f"用户已{'启用' if request.is_active else '禁用'}",
            "user_id": user.id,
            "is_active": user.is_active
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新用户状态失败: {str(e)}")


@router.put("/users/{user_id}/admin")
def set_user_admin(
        user_id: int,
        request: UserAdminUpdate,
        current_user: User = Depends(require_admin)
):
    """
    设置/取消用户管理员权限（需要管理员权限）
    """
    try:
        # 不能修改自己的管理员权限
        if user_id == current_user.id:
            raise HTTPException(status_code=400, detail="不能修改自己的管理员权限")

        user_service = UserService()
        user = user_service.set_user_admin(user_id, request.is_admin)

        return {
            "success": True,
            "message": f"用户已{'设为管理员' if request.is_admin else '取消管理员'}",
            "user_id": user.id,
            "is_admin": user.is_admin
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新管理员权限失败: {str(e)}")

@router.get("/users/{user_id}")
def get_user_detail(
        user_id: int,
        current_user: User = Depends(require_admin)
):
    """
    获取用户详细信息（需要管理员权限）
    """
    try:
        user_service = UserService()
        user = user_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "login_count": user.login_count,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户详情失败: {str(e)}")


# ========== 统计 API ==========

@router.get("/statistics", response_model=SystemStatistics)
def get_statistics(current_user: User = Depends(require_admin)):
    """
    获取系统统计信息（需要管理员权限）
    """
    try:
        from app.core.rag import rag_system

        # 用户统计
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        admin_users = db.query(User).filter(User.is_admin == True).count()

        # 对话统计
        total_conversations = db.query(Conversation).count()

        # 消息统计
        total_messages = db.query(ChatMessage).count()

        # 今日统计
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_conversations = db.query(Conversation).filter(
            Conversation.created_at >= today_start
        ).count()
        today_messages = db.query(ChatMessage).filter(
            ChatMessage.created_at >= today_start
        ).count()

        # 知识库统计
        total_knowledge = len(rag_system.knowledge_cache) if hasattr(rag_system, 'knowledge_cache') else 0

        return SystemStatistics(
            total_users=total_users,
            active_users=active_users,
            admin_users=admin_users,
            total_conversations=total_conversations,
            total_messages=total_messages,
            total_knowledge_segments=total_knowledge,
            today_conversations=today_conversations,
            today_messages=today_messages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# ========== 对话管理 API ==========
@router.get("/conversations")
def list_all_conversations(
        skip: int = Query(0, ge=0, description="跳过数量"),
        limit: int = Query(20, ge=1, le=100, description="返回数量"),
        current_user: User = Depends(require_admin)
):
    """
    查看所有用户对话（需要管理员权限）
    """
    try:
        conversations = db.query(Conversation).order_by(
            Conversation.updated_at.desc()
        ).offset(skip).limit(limit).all()

        total = db.query(Conversation).count()

        result = []
        for conv in conversations:
            user = db.get(User, conv.user_id)
            result.append({
                "id": conv.id,
                "user_id": conv.user_id,
                "username": user.username if user else "未知",
                "role_id": conv.role_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None
            })

        return {
            "total": total,
            "items": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {str(e)}")

@router.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
        conversation_id: int,
        current_user: User = Depends(require_admin)
):
    """
    查看指定对话的所有消息（需要管理员权限）
    """
    try:
        conversation = db.get(Conversation, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        ).order_by(ChatMessage.id.asc()).all()

        user = db.get(User, conversation.user_id)

        return {
            "conversation": {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "username": user.username if user else "未知",
                "role_id": conversation.role_id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None
            },
            "messages": [
                {
                    "id": msg.id,
                    "sender": msg.sender,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None
                }
                for msg in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {str(e)}")

@router.delete("/conversations/{conversation_id}")
def delete_conversation(
        conversation_id: int,
        current_user: User = Depends(require_admin)
):
    """
    删除指定对话（需要管理员权限）
    """
    try:
        conversation = db.get(Conversation, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        # 删除对话的所有消息
        db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation_id).delete()
        # 删除对话
        db.delete(conversation)
        db.commit()

        return {
            "success": True,
            "message": f"对话 {conversation_id} 已删除"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除对话失败: {str(e)}")


# ========== 知识库管理 API ==========
@router.get("/knowledge/statistics")
def get_knowledge_statistics(current_user: User = Depends(require_admin)):
    """
    获取知识库统计信息（需要管理员权限）
    """
    try:
        from app.core.rag import rag_system

        # 按角色统计
        role_counts = {}
        for item in rag_system.knowledge_items:
            for role_id in item.get("role_ids", []):
                role_counts[role_id] = role_counts.get(role_id, 0) + 1

        # 按来源文件统计
        source_counts = {}
        for item in rag_system.knowledge_items:
            source = item.get("source_file", "未知")
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "total_segments": len(rag_system.knowledge_cache),
            "total_files": len(source_counts),
            "role_counts": {str(k): v for k, v in role_counts.items()},
            "source_counts": source_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识库统计失败: {str(e)}")

@router.post("/knowledge/refresh")
def refresh_knowledge_base(current_user: User = Depends(require_admin)):
    """
    刷新知识库（需要管理员权限）
    """
    try:
        from app.core.rag import rag_system
        from app.services.knowledge_service import KnowledgeService
        import os

        # 构建知识库目录路径
        current_file = os.path.abspath(__file__)
        api_dir = os.path.dirname(current_file)
        app_dir = os.path.dirname(api_dir)
        knowledge_dir = os.path.join(app_dir, 'knowledge', 'data')

        knowledge_service = KnowledgeService()
        result = knowledge_service.refresh_knowledge(knowledge_dir, rag_system)

        return {
            "success": True,
            "message": "知识库刷新完成",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新知识库失败: {str(e)}")