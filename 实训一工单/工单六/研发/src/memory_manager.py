# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import redis

from src.config import Config
from utils.logger import get_logger


logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LongTermMemoryManager:
    def __init__(self, collection_name: str = None, memory_file: str = "memory.json"):
        self.collection_name = collection_name or Config.LONG_TERM_MEMORY_COLLECTION
        self.memory_file = Path(memory_file)
        self.memories: List[Dict] = []
        self._client = None
        self._embedding_model = None
        self._available: Optional[bool] = None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.memory_file.exists():
            self.memories = []
            return
        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            self.memories = data if isinstance(data, list) else data.get("memories", [])
        except Exception as exc:
            logger.warning("long-term memory disk load failed | file=%s | error=%s", self.memory_file, exc)
            self.memories = []

    def _save_to_disk(self) -> bool:
        try:
            self.memory_file.write_text(
                json.dumps(self.memories, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            logger.warning("long-term memory disk save failed | file=%s | error=%s", self.memory_file, exc)
            return False

    def add(self, question: str, answer: str) -> bool:
        """Add a QA pair to long-term memory and persist it locally."""
        if not question or not answer:
            return False

        memory_id = hashlib.sha1(f"{question}\n{answer}".encode("utf-8")).hexdigest()
        existing = next((item for item in self.memories if item.get("id") == memory_id), None)
        if existing:
            existing["helpful_count"] = int(existing.get("helpful_count", 0)) + 1
            existing["timestamp"] = _utc_now()
        else:
            self.memories.append(
                {
                    "id": memory_id,
                    "question": question,
                    "answer": answer,
                    "helpful_count": 1,
                    "timestamp": _utc_now(),
                }
            )
        disk_saved = self._save_to_disk()

        # Best-effort sync to Milvus; disabled by default to keep UI feedback instant.
        if getattr(Config, "ENABLE_MILVUS_LONG_TERM_SYNC", False):
            self._save_qa_to_milvus(question, answer)
        logger.info("long-term memory added | question=%s | disk_saved=%s", question, disk_saved)
        return disk_saved

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=Config.MILVUS_URI, timeout=Config.MILVUS_TIMEOUT)
        return self._client

    def refresh_status(self) -> bool:
        self._available = None
        self._client = None
        return self.is_available()

    def is_available(self) -> bool:
        try:
            self.client.list_collections()
            self._available = True
        except Exception as exc:
            logger.warning("milvus long-term memory unavailable | uri=%s | error=%s", Config.MILVUS_URI, exc)
            self._available = False
            self._client = None
        return self._available

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            return self._embedding_model
        except Exception:
            logger.exception("long-term memory embedding model load failed | model=%s", Config.EMBEDDING_MODEL)
            return None

    def _embed(self, text: str) -> Optional[List[float]]:
        model = self._load_embedding_model()
        if model is None:
            return None
        vector = model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def ensure_collection(self) -> bool:
        if not self.is_available():
            return False
        if self.client.has_collection(self.collection_name):
            return True

        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=8192)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
        schema.add_field("helpful_count", DataType.INT64)
        schema.add_field("timestamp", DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("long-term memory collection created | collection=%s", self.collection_name)
        return True

    def _save_qa_to_milvus(self, question: str, answer: str) -> bool:
        if not question or not answer or not self.ensure_collection():
            return False
        embedding = self._embed(f"问题：{question}\n答案：{answer}")
        if embedding is None:
            return False

        memory_id = hashlib.sha1(f"{question}\n{answer}".encode("utf-8")).hexdigest()
        helpful_count = 1
        try:
            existing = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{memory_id}"',
                output_fields=["helpful_count"],
                limit=1,
            )
            if existing:
                helpful_count = int(existing[0].get("helpful_count", 0)) + 1
        except Exception as exc:
            logger.warning("long-term memory existing lookup failed | id=%s | error=%s", memory_id, exc)

        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "id": memory_id,
                    "question": question[:2048],
                    "answer": answer[:8192],
                    "embedding": embedding,
                    "helpful_count": helpful_count,
                    "timestamp": _utc_now(),
                }
            ],
        )
        logger.info("long-term memory saved | collection=%s | question=%s", self.collection_name, question)
        return True

    def save_qa(self, question: str, answer: str) -> bool:
        return self.add(question, answer)

    def _search_local(self, question: str, threshold: float = None, top_k: int = 1) -> Optional[Dict]:
        if not question or not self.memories:
            return None

        question_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", question or ""))
        ranked = []
        for item in self.memories:
            memory_text = f"{item.get('question', '')} {item.get('answer', '')}"
            memory_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", memory_text))
            if question in item.get("question", "") or item.get("question", "") in question:
                score = 1.0
            elif question_tokens and memory_tokens:
                score = len(question_tokens & memory_tokens) / max(len(question_tokens | memory_tokens), 1)
            else:
                score = 0.0
            ranked.append((item, score))

        ranked.sort(key=lambda pair: (pair[1], int(pair[0].get("helpful_count", 0))), reverse=True)
        if not ranked:
            return None
        item, score = ranked[0]
        threshold = threshold if threshold is not None else min(Config.LONG_TERM_MEMORY_THRESHOLD, 0.35)
        if float(score) < threshold:
            return None
        return {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "helpful_count": item.get("helpful_count", 0),
            "timestamp": item.get("timestamp", ""),
            "score": float(score),
            "source": "memory.json",
        }

    def search(self, question: str, threshold: float = None, top_k: int = 1) -> Optional[Dict]:
        local_match = self._search_local(question, threshold=threshold, top_k=top_k)
        if local_match:
            return local_match
        if not getattr(Config, "ENABLE_MILVUS_LONG_TERM_SEARCH", False):
            return None
        if not question or not self.ensure_collection():
            return None
        embedding = self._embed(question)
        if embedding is None:
            return None

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="embedding",
            limit=top_k,
            output_fields=["question", "answer", "helpful_count", "timestamp"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        if not results or not results[0]:
            return None

        hit = results[0][0]
        score = hit.get("distance", hit.get("score", 0.0)) if isinstance(hit, dict) else getattr(hit, "distance", 0.0)
        threshold = threshold if threshold is not None else Config.LONG_TERM_MEMORY_THRESHOLD
        if float(score) < threshold:
            return None

        entity = hit.get("entity", hit) if isinstance(hit, dict) else getattr(hit, "entity", {})
        return {
            "question": entity.get("question", ""),
            "answer": entity.get("answer", ""),
            "helpful_count": entity.get("helpful_count", 0),
            "timestamp": entity.get("timestamp", ""),
            "score": float(score),
        }

    def list_memories(self, limit: int = 20) -> List[Dict]:
        if self.memories:
            return sorted(
                self.memories,
                key=lambda item: (int(item.get("helpful_count", 0)), item.get("timestamp", "")),
                reverse=True,
            )[:limit]
        if not self.ensure_collection():
            return []
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="helpful_count >= 0",
                output_fields=["question", "answer", "helpful_count", "timestamp"],
                limit=limit,
            )
        except Exception as exc:
            logger.warning("long-term memory list failed | collection=%s | error=%s", self.collection_name, exc)
            return []
        return rows or []


class RedisMemoryManager:
    def __init__(self, redis_url: str = None, max_messages: int = None, key_prefix: str = None):
        self.redis_url = redis_url or Config.REDIS_URL
        self.max_messages = max_messages or Config.SHORT_MEMORY_LIMIT
        self.key_prefix = key_prefix or Config.SHORT_MEMORY_KEY_PREFIX
        self._client: Optional[redis.Redis] = None
        self._available: Optional[bool] = None

    @property
    def client(self) -> Optional[redis.Redis]:
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self.client.ping()
            self._available = True
        except Exception as exc:
            logger.warning("redis memory unavailable | url=%s | error=%s", self.redis_url, exc)
            self._available = False
        return self._available

    def _key(self, session_id: str = "default") -> str:
        return f"{self.key_prefix}:{session_id}"

    def add_message(self, role: str, content: str, session_id: str = "default") -> bool:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not content:
            return False
        if not self.is_available():
            return False

        record = {
            "role": role,
            "content": content,
            "timestamp": _utc_now(),
        }
        key = self._key(session_id)
        self.client.rpush(key, json.dumps(record, ensure_ascii=False))
        self.client.ltrim(key, -self.max_messages, -1)
        self.client.expire(key, Config.REDIS_SESSION_TTL)
        return True

    def get_history(self, session_id: str = "default", limit: int = None) -> List[Dict]:
        if not self.is_available():
            return []

        limit = limit or self.max_messages
        raw_items = self.client.lrange(self._key(session_id), -limit, -1)
        history: List[Dict] = []
        for raw in raw_items:
            try:
                history.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning("invalid redis memory record skipped | raw=%s", raw)
        return history

    def get_context(self, session_id: str = "default", limit: int = None) -> str:
        history = self.get_history(session_id=session_id, limit=limit)
        return "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in history)

    def clear(self, session_id: str = "default") -> None:
        if self.is_available():
            self.client.delete(self._key(session_id))
