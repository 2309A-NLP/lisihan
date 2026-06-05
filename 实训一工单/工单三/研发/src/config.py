# -*- coding: utf-8 -*-
"""Application configuration."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


class Config:
    TASK_ID = "PDF招股说明书智能问答系统"

    EMBEDDING_MODEL_PATH = os.getenv(
        "EMBEDDING_MODEL_PATH",
        r"C:\Users\freedom\Desktop\专业\模型\bge-base-zh-v1.5",
    )
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_PATH)
    EMBEDDING_BACKEND = "local"
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "siliconflow")
    LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.siliconflow.cn/v1")
    LLM_API_KEY = os.getenv("SILICONFLOW_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "2.5"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_CONTEXT_MAX_CHARS = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "3600"))
    OPENAI_API_KEY = LLM_API_KEY

    RETRIEVAL_BACKEND = "hybrid"
    COLLECTION_NAME = os.getenv("BM25_COLLECTION", "pdf_documents")

    # 混合检索配置
    BM25_K = int(os.getenv("BM25_K", "6"))
    VECTOR_K = int(os.getenv("VECTOR_K", "4"))
    RRF_K = int(os.getenv("RRF_K", "30"))
    BM25_RRF_WEIGHT = float(os.getenv("BM25_RRF_WEIGHT", "3.0"))
    VECTOR_RRF_WEIGHT = float(os.getenv("VECTOR_RRF_WEIGHT", "1.0"))
    FINAL_K = int(os.getenv("FINAL_K", "3"))
    SKIP_VECTOR_FOR_EXACT_QUERIES = os.getenv("SKIP_VECTOR_FOR_EXACT_QUERIES", "true").lower() == "true"
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", str(FINAL_K)))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", str(BM25_K)))
    HYBRID_CANDIDATE_TOP_K = int(os.getenv("HYBRID_CANDIDATE_TOP_K", str(BM25_K)))

    # Milvus 配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_LONG_TERM = os.getenv("MILVUS_COLLECTION_LONG_TERM", "long_term_memory")
    MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    MILVUS_TIMEOUT = float(os.getenv("MILVUS_TIMEOUT", "3"))
    MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
    MILVUS_OUTPUT_FIELDS = [
        item.strip()
        for item in os.getenv(
            "MILVUS_OUTPUT_FIELDS",
            "content,metadata,source_file,page,chunk_id,source_path",
        ).split(",")
        if item.strip()
    ]

    # Redis 配置
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "86400"))
    REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    SHORT_MEMORY_LIMIT = int(os.getenv("SHORT_MEMORY_LIMIT", "20"))
    SHORT_MEMORY_KEY_PREFIX = os.getenv("SHORT_MEMORY_KEY_PREFIX", "rag:short_memory")
    LONG_TERM_MEMORY_COLLECTION = os.getenv("LONG_TERM_MEMORY_COLLECTION", MILVUS_COLLECTION_LONG_TERM)
    LONG_TERM_MEMORY_THRESHOLD = float(os.getenv("LONG_TERM_MEMORY_THRESHOLD", str(SIMILARITY_THRESHOLD)))
    LONG_TERM_MEMORY_LIMIT = int(os.getenv("LONG_TERM_MEMORY_LIMIT", "20"))
    ENABLE_ANSWER_CACHE = os.getenv("ENABLE_ANSWER_CACHE", "false").lower() == "true"
    ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "30"))
    ENABLE_LONG_TERM_MEMORY_ANSWER = os.getenv("ENABLE_LONG_TERM_MEMORY_ANSWER", "false").lower() == "true"

    PDF_DIR = os.getenv("PDF_DIR", "./data")
    PDF_PARSE_OUTPUT_DIR = os.getenv("PDF_PARSE_OUTPUT_DIR", "./parsed_pdfs")
    PDF_CHUNK_STRATEGY = os.getenv("PDF_CHUNK_STRATEGY", "paragraph")
    PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "500"))
    PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "50"))
    ENABLE_MINERU_WSL = os.getenv("ENABLE_MINERU_WSL", "true").lower() == "true"
    MINERU_OUTPUT_DIR = os.getenv("MINERU_OUTPUT_DIR", "./mineru_output")
    MINERU_WSL_CONDA_PREFIX = os.getenv("MINERU_WSL_CONDA_PREFIX", "/home/li/miniconda3")
    MINERU_WSL_CONDA_ENV = os.getenv("MINERU_WSL_CONDA_ENV", "base")
    MINERU_WSL_DISTRO = os.getenv("MINERU_WSL_DISTRO", "")
    MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
    MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "3600"))
    MINERU_USE_CACHE = os.getenv("MINERU_USE_CACHE", "true").lower() == "true"
    MAX_RESPONSE_TIME = 3

    EVALUATION_QUESTIONS = [
        {"id": 260, "question": "军用领域收入分别是多少"},
        {"id": 95, "question": "参与制定了哪个技术标准"},
        {"id": 33, "question": "收入占主营业务收入的比重分别是多少"},
        {"id": 34, "question": "上游涉及哪些企业"},
        {"id": 957, "question": "在哪个领域已经成为重要供应商"},
        {"id": 793, "question": "下游主要包括哪些行业"},
        {"id": 795, "question": "哪个工程荣获了国家科技进步一等奖"},
        {"id": 543, "question": "注册资本是多少"},
        {"id": 531, "question": "法定代表人是谁"},
        {"id": 207, "question": "计划使用多少募集资金补充流动资金"},
        {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
        {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
        {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
        {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    ]

