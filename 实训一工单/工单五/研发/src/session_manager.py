# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件用于工单五的多轮对话历史管理。历史只作为指代消解和主题公司识别的
参考信息，不作为最终答案来源，答案仍由 RAG 检索、多模态解析和校验链路生成。
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

try:
    import redis
except Exception:  # pragma: no cover - redis is optional at runtime
    redis = None

from src.config import Config
from utils.logger import get_logger


logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    """保存最近 N 轮问答，优先使用 Redis，Redis 不可用时退回进程内存。"""

    def __init__(
        self,
        redis_url: str = None,
        max_rounds: int = None,
        key_prefix: str = None,
    ):
        self.redis_url = redis_url or Config.REDIS_URL
        self.max_rounds = max_rounds or int(getattr(Config, "SESSION_HISTORY_LIMIT", 5))
        self.key_prefix = key_prefix or getattr(Config, "SESSION_MEMORY_KEY_PREFIX", "rag:session")
        self._client: Optional["redis.Redis"] = None
        self._available: Optional[bool] = None
        self._memory_store: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=self.max_rounds))

    @property
    def client(self):
        if redis is None:
            return None
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if self.client is None:
            self._available = False
            return False
        try:
            self.client.ping()
            self._available = True
        except Exception as exc:
            logger.warning("redis session memory unavailable | url=%s | error=%s", self.redis_url, exc)
            self._available = False
        return self._available

    def _key(self, session_id: str = "default") -> str:
        safe_session_id = session_id or "default"
        return f"{self.key_prefix}:{safe_session_id}"

    def add_turn(
        self,
        *,
        session_id: str = "default",
        question: str,
        resolved_question: str,
        answer: str,
        mentioned_companies: List[str] = None,
        current_company: str = None,
        metadata: Dict = None,
    ) -> bool:
        if not question and not resolved_question:
            return False

        record = {
            "question": question or "",
            "resolved_question": resolved_question or question or "",
            "answer": answer or "",
            "mentioned_companies": mentioned_companies or [],
            "current_company": current_company or "",
            "metadata": metadata or {},
            "timestamp": _utc_now(),
        }

        if self.is_available():
            try:
                key = self._key(session_id)
                self.client.rpush(key, json.dumps(record, ensure_ascii=False))
                self.client.ltrim(key, -self.max_rounds, -1)
                self.client.expire(key, Config.REDIS_SESSION_TTL)
                return True
            except Exception as exc:
                logger.warning("redis session write failed | session_id=%s | error=%s", session_id, exc)
                self._available = False

        self._memory_store[session_id or "default"].append(record)
        return True

    def get_history(self, session_id: str = "default", limit: int = None) -> List[Dict]:
        limit = limit or self.max_rounds
        if self.is_available():
            try:
                raw_items = self.client.lrange(self._key(session_id), -limit, -1)
                history: List[Dict] = []
                for raw in raw_items:
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("invalid session memory record skipped | raw=%s", raw)
                        continue
                    if isinstance(item, dict):
                        history.append(item)
                return history
            except Exception as exc:
                logger.warning("redis session read failed | session_id=%s | error=%s", session_id, exc)
                self._available = False

        history = list(self._memory_store.get(session_id or "default", []))
        return history[-limit:]

    def clear(self, session_id: str = "default") -> None:
        if self.is_available():
            try:
                self.client.delete(self._key(session_id))
            except Exception as exc:
                logger.warning("redis session clear failed | session_id=%s | error=%s", session_id, exc)
                self._available = False
        self._memory_store.pop(session_id or "default", None)
