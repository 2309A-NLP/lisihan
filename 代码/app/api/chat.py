# -*- coding: utf-8 -*-
"""
聊天相关 API 路由模块
======================
提供聊天机器人的核心功能接口：

功能列表：
- POST /api/chat/          - 普通聊天接口（一次性返回完整回复）
- POST /api/chat/stream    - 流式聊天接口（SSE实时返回，打字机效果）
- POST /api/chat/conversations - 创建新对话会话
- GET  /api/chat/conversations - 查询用户的对话列表
- GET  /api/chat/conversations/{id}/messages - 获取指定对话的消息记录
- POST /api/chat/history   - 获取指定角色的最近对话历史

技术特点：
- 支持同步和异步处理
- 使用线程池处理阻塞操作
- SSE（Server-Sent Events）实现流式响应
- 自动创建数据库表
"""

# ========== 导入标准库 ==========
import logging  # 日志记录模块，用于记录程序运行状态和错误信息
from typing import List, Optional  # 类型提示：List用于列表类型，Optional用于可选参数

# ========== 导入FastAPI相关组件 ==========
from fastapi import APIRouter, HTTPException, Query  # APIRouter: 路由管理器, HTTPException: 错误响应, Query: 查询参数
from fastapi.concurrency import run_in_threadpool  # 线程池执行器，将同步函数转为异步执行
from fastapi.responses import StreamingResponse  # 流式响应，用于SSE实时推送数据

# ========== 导入数据库相关 ==========
from app.models import SessionLocal, engine  # SessionLocal: 数据库会话, engine: 数据库引擎
from app.models.chat import ChatMessage, Conversation  # ChatMessage: 聊天消息模型, Conversation: 对话会话模型

# ========== 导入Pydantic数据模型（Schemas） ==========
from app.schemas.chat import (
    ChatRequest,  # 聊天请求体：包含用户消息、角色ID等
    ChatResponse,  # 聊天响应体：包含AI回复、会话ID
    ConversationCreateRequest,  # 创建会话请求：角色ID、标题等
    ConversationResponse,  # 会话响应：会话信息列表
    HistoryRequest,  # 历史记录请求：用户ID、角色ID
    HistoryResponse,  # 历史记录响应：消息历史列表
    MessageResponse,  # 消息响应：单条消息格式
)

# ========== 导入业务服务层 ==========
from app.services.chat_service import run_chat_sync  # 同步聊天服务：核心聊天逻辑（RAG+LLM）
from app.services.role_service import RoleService  # 角色服务：角色查询和管理

# ========== 导入工具函数 ==========
from app.utils.chat_utils import (
    _latest_user_question,  # 获取会话中最新的一条用户问题
    _message_timestamp,  # 获取消息的时间戳（格式化）
    _serialize_conversation,  # 序列化会话对象为字典格式
    _sse,  # 生成SSE格式的数据包
    _split_text,  # 将长文本分割成小块（用于流式输出）
)

# ========== 初始化路由器和日志 ==========
router = APIRouter()  # 创建APIRouter实例，用于在主应用中注册
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
_run_chat_sync = run_chat_sync  # 创建本地引用（便于在线程池中调用）


# ========== 辅助函数：确保数据库表存在 ==========
def _ensure_chat_tables():
    """确保聊天相关的数据库表已经创建。

    这个函数会在每次数据库操作前调用，
    确保表结构存在（适用于开发环境，生产环境建议用迁移工具）。
    """
    from app.models import Base  # 导入所有模型的基类
    Base.metadata.create_all(bind=engine)  # 创建所有未创建的表


# ========== 辅助函数：SSE流式事件生成器 ==========
async def _stream_chat_events(request: ChatRequest):
    """流式执行聊天：复用完整的聊天流程，再按片段返回结果。

    这是一个异步生成器函数，通过SSE协议逐步返回AI的回复内容。

    工作流程：
    1. 发送开始事件
    2. 在线程池中执行同步聊天函数
    3. 发送元数据（会话ID）
    4. 将AI回复分割成小块，逐个发送
    5. 发送完成事件
    6. 发生错误时发送错误事件

    参数:
        request: ChatRequest - 聊天请求对象（包含用户消息、角色ID等）

    产出:
        SSE格式的字符串: data: {json}\n\n
    """
    # 1. 发送开始事件，告知客户端流式传输开始
    yield _sse("start", {"message": "started"})

    try:
        # 2. 在线程池中执行同步聊天函数（避免阻塞事件循环）
        # run_in_threadpool 将同步函数包装成异步调用
        result = await run_in_threadpool(_run_chat_sync, request)

        # 3. 发送元数据：包含会话ID（客户端可以用这个ID管理会话）
        yield _sse("meta", {"conversation_id": result.conversation_id})

        # 4. 将AI回复分割成小块，逐个发送（实现打字机效果）
        for chunk in _split_text(result.response):
            yield _sse("token", {"content": chunk})

        # 5. 发送完成事件，包含完整的回复内容和会话ID
        yield _sse("done", {
            "conversation_id": result.conversation_id,
            "response": result.response
        })

    except HTTPException as exc:
        # 6. 处理HTTP异常（如404、400等）
        yield _sse("error", {"status_code": exc.status_code, "detail": exc.detail})
    except Exception as exc:
        # 7. 处理其他未预期的异常
        yield _sse("error", {"status_code": 500, "detail": str(exc)})


# ========== API端点1: 创建新对话 ==========
@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(request: ConversationCreateRequest):
    """创建新对话会话。

    为指定用户和角色创建一个新的对话会话。
    会话用于组织和存储一组连续的消息。

    请求体示例:
    {
        "user_id": 1,           # 用户ID
        "role_id": 2,           # 角色ID（如医生、律师等）
        "title": "健康咨询"      # 可选，会话标题
    }

    返回:
        ConversationResponse: 创建的会话信息（ID、标题、创建时间等）
    """
    # 确保数据库表存在
    _ensure_chat_tables()

    # 创建数据库会话（每个请求独立的会话）
    session = SessionLocal()

    try:
        # 1. 验证角色是否存在
        role = RoleService().get_role_by_id(request.role_id)
        if not role:
            # 角色不存在，返回404错误
            raise HTTPException(status_code=404, detail="Role not found")

        # 2. 创建会话对象
        conversation = Conversation(
            user_id=request.user_id,  # 关联的用户ID
            role_id=request.role_id,  # 关联的角色ID
            title=request.title or f"{role.name} new chat",  # 标题（默认格式）
        )

        # 3. 添加到数据库并提交
        session.add(conversation)  # 添加到会话
        session.commit()  # 提交事务
        session.refresh(conversation)  # 刷新获取自增ID等字段

        # 4. 序列化并返回
        return _serialize_conversation(conversation)

    except HTTPException:
        # 重新抛出HTTP异常（不记录堆栈）
        raise
    except Exception as exc:
        # 发生其他异常，回滚事务
        session.rollback()
        # 记录异常日志（包含完整堆栈）
        logger.exception("创建会话异常", exc_info=exc)
        # 返回500错误
        raise HTTPException(status_code=500, detail=f"Create conversation failed: {exc}")
    finally:
        # 确保数据库会话被关闭（释放连接）
        session.close()


# ========== API端点2: 获取用户会话列表 ==========
@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(
        user_id: int = Query(..., description="用户ID（必填）"),
        role_id: Optional[int] = Query(None, description="角色ID（可选，用于过滤）")
):
    """列出用户的所有会话，可按角色过滤。

    查询参数:
        - user_id: 用户的ID（必填）
        - role_id: 角色的ID（可选，只返回该角色的会话）

    返回:
        List[ConversationResponse]: 会话列表，按更新时间倒序排列
    """
    # 确保数据库表存在
    _ensure_chat_tables()

    # 创建数据库会话
    session = SessionLocal()

    try:
        # 1. 构建基础查询：筛选指定用户的会话
        query = session.query(Conversation).filter(Conversation.user_id == user_id)

        # 2. 如果提供了role_id，添加角色过滤条件
        if role_id is not None:
            query = query.filter(Conversation.role_id == role_id)

        # 3. 执行查询：按更新时间倒序，然后按ID倒序
        conversations = query.order_by(
            Conversation.updated_at.desc(),  # 最近更新的在前
            Conversation.id.desc()  # 相同时间ID大的在前
        ).all()

        # 4. 序列化每个会话，并包含最新的用户问题
        return [
            _serialize_conversation(
                item,
                last_question=_latest_user_question(session, item.id)  # 获取最新问题
            )
            for item in conversations
        ]

    except Exception as exc:
        # 记录异常日志
        logger.exception("查询会话列表异常", exc_info=exc)
        raise HTTPException(status_code=500, detail=f"List conversations failed: {exc}")
    finally:
        # 关闭数据库会话
        session.close()


# ========== API端点3: 获取指定会话的消息记录 ==========
@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
def list_messages(
        conversation_id: int,  # 路径参数：会话ID
        user_id: int = Query(..., description="用户ID（用于权限验证）")
):
    """获取指定会话的所有消息记录。

    路径参数:
        - conversation_id: 会话ID

    查询参数:
        - user_id: 用户ID（验证用户是否有权访问该会话）

    返回:
        List[MessageResponse]: 消息列表，按发送顺序排列
    """
    # 确保数据库表存在
    _ensure_chat_tables()

    # 创建数据库会话
    session = SessionLocal()

    try:
        # 1. 获取会话并验证所有权
        conversation = session.get(Conversation, conversation_id)

        # 2. 验证：会话不存在或不属于该用户
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # 3. 查询该会话的所有消息（按ID升序，即时间顺序）
        messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.id.asc())  # 升序：最早的在前
            .all()
        )

        # 4. 转换为响应格式
        return [
            MessageResponse(
                id=item.id,  # 消息ID
                conversation_id=item.conversation_id,  # 所属会话ID
                sender=item.sender,  # 发送者（user/assistant）
                content=item.content,  # 消息内容
                timestamp=_message_timestamp(item),  # 时间戳（格式化）
            )
            for item in messages
        ]

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as exc:
        # 记录其他异常
        logger.exception("查询会话消息异常", exc_info=exc)
        raise HTTPException(status_code=500, detail=f"List messages failed: {exc}")
    finally:
        # 关闭数据库会话
        session.close()


# ========== API端点4: 普通聊天接口 ==========
@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """普通聊天接口：在线程池中执行同步RAG/数据库逻辑。

    这个接口会等待完整的AI回复生成后，一次性返回结果。
    适合不需要实时显示打字效果的场景。

    请求体示例:
    {
        "user_id": 1,
        "role_id": 2,
        "message": "我最近头疼",
        "conversation_id": null  # 可选，不提供则自动创建
    }

    返回:
        ChatResponse: 包含AI回复和会话ID
    """
    # 在线程池中执行同步聊天函数
    # 为什么要用run_in_threadpool？
    # 因为run_chat_sync内部包含：
    # - 数据库查询（同步）
    # - RAG检索（可能耗时）
    # - LLM调用（可能耗时）
    # 这些操作会阻塞事件循环，所以要放到线程池中执行
    result = await run_in_threadpool(_run_chat_sync, request)

    # 返回响应
    return ChatResponse(
        response=result.response,  # AI的回复内容
        conversation_id=result.conversation_id  # 会话ID（新建或已有的）
    )


# ========== API端点5: 流式聊天接口 ==========
@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口：使用SSE按token分块返回回复。

    实现打字机效果，AI实时生成内容并逐字/逐词推送给客户端。

    优势：
    - 用户体验好，不用等待完整回复
    - 降低首字延迟
    - 适合长文本生成

    响应格式：Server-Sent Events (SSE)
    事件类型：
        - start: 开始生成
        - meta: 元数据（会话ID）
        - token: 内容片段
        - done: 完成
        - error: 错误

    客户端示例:
        const eventSource = new EventSource('/api/chat/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'token') {
                appendText(data.content);
            }
        };
    """
    # 返回流式响应
    return StreamingResponse(
        _stream_chat_events(request),  # 事件生成器
        media_type="text/event-stream",  # SSE的MIME类型
        headers={
            "Cache-Control": "no-cache",  # 禁用缓存，确保实时性
            "X-Accel-Buffering": "no",  # 禁用nginx缓冲（用于反向代理）
        },
    )


# ========== API端点6: 获取指定角色的最近会话历史 ==========
@router.post("/history", response_model=HistoryResponse)
def get_history(request: HistoryRequest):
    """获取指定角色最近会话的历史记录。

    用于加载对话历史，让AI了解之前的对话上下文。

    请求体示例:
    {
        "user_id": 1,
        "role_id": 2,
        "limit": 10  # 可选，限制消息数量
    }

    返回:
        HistoryResponse: 包含消息历史列表
    """
    # 确保数据库表存在
    _ensure_chat_tables()

    # 创建数据库会话
    session = SessionLocal()

    try:
        # 1. 查询用户在该角色下的最新会话
        conversation = (
            session.query(Conversation)
            .filter(
                Conversation.user_id == request.user_id,
                Conversation.role_id == request.role_id
            )
            .order_by(
                Conversation.updated_at.desc(),  # 最新更新的在前
                Conversation.id.desc()  # ID大的在前
            )
            .first()  # 取第一条，即最新的会话
        )

        # 2. 如果没有找到会话，返回空历史
        if not conversation:
            return HistoryResponse(history=[])

        # 3. 查询该会话的所有消息
        rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation.id)
            .order_by(ChatMessage.id.asc())  # 按时间顺序
            .all()
        )

        # 4. 转换为历史记录格式
        history = [
            {
                "sender": item.sender,  # 发送者
                "content": item.content,  # 消息内容
                "timestamp": _message_timestamp(item),  # 时间戳
                "conversation_id": item.conversation_id,  # 会话ID
            }
            for item in rows
        ]

        # 5. 返回历史记录
        return HistoryResponse(history=history)

    except Exception as exc:
        # 记录异常
        logger.exception("查询历史记录异常", exc_info=exc)
        raise HTTPException(status_code=500, detail=f"Get history failed: {exc}")
    finally:
        # 关闭数据库会话
        session.close()