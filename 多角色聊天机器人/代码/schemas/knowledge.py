# -*- coding: utf-8 -*-
"""知识库 API 数据模型。"""

from pydantic import BaseModel


class KnowledgeAddRequest(BaseModel):
    """添加知识库内容。"""

    knowledge_base_id: int
    content: str


class KnowledgeUpdateRequest(BaseModel):
    """更新知识库内容。"""

    knowledge_base_id: int
    content: str


class KnowledgeResponse(BaseModel):
    """知识库操作通用响应。"""

    status: str


class KnowledgeUpdateResponse(BaseModel):
    """知识库更新操作详细响应。"""

    success: bool
    data: dict


class EvaluationRequest(BaseModel):
    """RAG 评测请求。"""

    test_cases: list


class EvaluationResponse(BaseModel):
    """RAG 评测响应。"""

    success: bool
    data: dict
