# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""聊天相关的请求/响应模型定义。"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ========== 默认配置常量 ==========
DEFAULT_HISTORY_LIMIT = 8
DEFAULT_CONTEXT_BUDGET = 6000
DEFAULT_KNOWLEDGE_TOP_K = 2
DEFAULT_MEMORY_TOP_K = 4


# ========== 请求模型 ==========
class ChatRequest(BaseModel):
    """聊天请求数据。

    Attributes:
        user_id: 用户ID
        role_id: 角色ID（AI助手身份）
        message: 用户消息内容
        conversation_id: 会话ID（可选，不传则创建新会话）
        stream: 是否使用流式输出
        history_limit: 历史消息数量限制（0-30条）
        context_budget: 上下文预算（字符数，1000-20000）
        knowledge_top_k: 知识库检索数量（0-10条，0表示禁用）
        memory_top_k: 长期记忆检索数量（0-10条，0表示禁用）
        use_embedding_memory: 是否使用 embedding 检索长期记忆
    """

    user_id: int
    role_id: int
    message: str
    conversation_id: Optional[int] = None
    stream: bool = False
    history_limit: int = Field(DEFAULT_HISTORY_LIMIT, ge=0, le=30)
    context_budget: int = Field(DEFAULT_CONTEXT_BUDGET, ge=1000, le=20000)
    knowledge_top_k: int = Field(DEFAULT_KNOWLEDGE_TOP_K, ge=0, le=10)
    memory_top_k: int = Field(DEFAULT_MEMORY_TOP_K, ge=0, le=10)
    use_embedding_memory: bool = True


class ConversationCreateRequest(BaseModel):
    """创建新对话请求数据。"""

    user_id: int
    role_id: int
    title: Optional[str] = None


class HistoryRequest(BaseModel):
    """历史记录查询请求数据。"""

    user_id: int
    role_id: int


# ========== 响应模型 ==========
class ChatResponse(BaseModel):
    """普通聊天接口返回数据。"""

    response: str
    conversation_id: int


class ConversationResponse(BaseModel):
    """对话列表和详情返回数据。"""

    id: int
    user_id: int
    role_id: int
    title: str
    updated_at: Optional[str] = None
    last_question: Optional[str] = None


class MessageResponse(BaseModel):
    """单条聊天消息返回数据。"""

    id: int
    conversation_id: int
    sender: str
    content: str
    timestamp: int


class HistoryResponse(BaseModel):
    """历史记录返回数据。"""

    history: List[Dict[str, Any]]


# ========== 内部数据结构 ==========
class ChatResult(BaseModel):
    """内部聊天执行结果。"""

    response: str
    conversation_id: int