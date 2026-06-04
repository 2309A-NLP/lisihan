# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""聊天上下文构建核心逻辑。

负责：
- 消息长度估算
- 预算裁剪
- 去重
- 三层记忆合并
- 格式转换
"""

import logging
from typing import Any, Dict, List

from app.schemas.chat import DEFAULT_CONTEXT_BUDGET, ChatRequest

logger = logging.getLogger(__name__)


def _message_len(item: Dict[str, Any]) -> int:
    """估算一条上下文消息的长度，用于控制提示词预算。

    计算公式：content长度 + sender长度 + 12（格式开销）
    """
    return len(str(item.get("content", "") or "")) + len(str(item.get("sender", "") or "")) + 12


def _trim_by_budget(items: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """按字符预算从后往前保留上下文，避免无限拼接历史。

    策略：
    1. 从最新的消息开始遍历（倒序）
    2. 尽可能多地保留消息，直到预算用尽
    3. 单条消息超预算时，截断其内容
    """
    kept: List[Dict[str, Any]] = []
    used = 0
    for item in reversed(items):
        cost = _message_len(item)
        if kept and used + cost > budget:
            break
        if cost > budget:
            trimmed = dict(item)
            trimmed["content"] = str(trimmed.get("content", ""))[-budget:]
            kept.append(trimmed)
            break
        kept.append(item)
        used += cost
    return list(reversed(kept))


def _dedupe_context(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按发送方和内容去重，避免多层存储重复注入同一条消息。"""
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("sender"), item.get("content"))
        if not item.get("content") or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _memory_for_conversation(items: List[Dict[str, Any]], conversation_id: int) -> List[Dict[str, Any]]:
    """从Redis中筛选出属于指定会话的对话记录，并按时间排序。"""
    filtered: List[Dict[str, Any]] = []
    for item in items or []:
        try:
            item_conversation_id = int(item.get("conversation_id"))
        except (TypeError, ValueError):
            continue
        if item_conversation_id == int(conversation_id):
            filtered.append(item)
    return sorted(filtered, key=lambda item: int(item.get("timestamp") or 0))


def _build_context(
    history: List[Dict[str, Any]],
    short_memory: List[Dict[str, Any]],
    long_memory: List[Dict[str, Any]],
    budget: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """合并历史、短期记忆和长期记忆，并按预算裁剪。

    预算分配策略：
    - 长期记忆：预算的 1/4，最少 200 字符
    - 短期记忆：预算的 1/4，最少 200 字符
    - 历史对话：剩余预算，最少 400 字符
    """
    budget = max(1000, int(budget or DEFAULT_CONTEXT_BUDGET))
    long_budget = max(200, budget // 4)
    short_budget = max(200, budget // 4)
    history_budget = max(400, budget - long_budget - short_budget)

    long_items = _trim_by_budget(_dedupe_context(long_memory), long_budget)
    short_items = _trim_by_budget(_dedupe_context(short_memory), short_budget)
    history_items = _trim_by_budget(_dedupe_context(history), history_budget)
    combined = _trim_by_budget(_dedupe_context(long_items + short_items + history_items), budget)

    return {
        "history": history_items,
        "short_memory": short_items,
        "long_memory": long_items,
        "combined": combined,
    }


def _context_as_rag_items(context_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把会话上下文转换成 RAG 可消费的 context 条目。"""
    rag_items = []
    for item in context_items:
        sender = item.get("sender") or "unknown"
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        rag_items.append({
            "content": f"历史上下文({sender}): {content}",
            "source_file": "chat_context",
            "score": 0,
        })
    return rag_items


def _build_response_context(
    request: ChatRequest,
    context_bundle: Dict[str, List[Dict[str, Any]]],
    rag_system,  # 依赖注入
) -> List[Dict[str, Any]]:
    """组合知识库检索结果和会话上下文，知识库始终排在历史记忆前面。

    重要改进：
    1. 知识库检索结果优先（放在列表前面）
    2. 当知识库为空时，添加系统提示，避免 LLM 依赖不相关的历史对话
    """
    context_items = _context_as_rag_items(context_bundle["combined"])

    if request.knowledge_top_k <= 0:
        return context_items

    knowledge_items = rag_system.search_knowledge(
        request.message,
        top_k=request.knowledge_top_k,
        role_id=request.role_id,
    )

    # 处理知识库为空的情况
    if not knowledge_items:
        logger.warning(
            f"知识库检索无结果 | user_id={request.user_id} | "
            f"role_id={request.role_id} | query={request.message[:50]}"
        )

        if context_items:
            knowledge_items = [{
                "content": (
                    "【提示】知识库中没有找到与当前问题直接相关的信息。"
                    "请基于你的通用知识回答，不要重复或引用历史对话中的内容，"
                    "除非用户明确在追问之前的话题。"
                ),
                "source_file": "system_hint",
                "score": 0,
            }]
        else:
            knowledge_items = [{
                "content": "【提示】没有找到相关的参考资料，请基于你的知识回答。",
                "source_file": "system_hint",
                "score": 0,
            }]

    return knowledge_items + context_items