# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件属于 PDF 招股说明书智能问答系统，增加工单五多轮对话配置，并保留
工单一到工单四的文本检索、图片内容解析和 Redis 缓存能力。
"""

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
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "tp-cxit9r7gak3n335w1vewzxjadh7f8d34ahecucld7514moj9"))
    LLM_MODEL = os.getenv("LLM_MODEL", "mimo-v2.5-pro")
    MULTIMODAL_API_BASE_URL = os.getenv("MULTIMODAL_API_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    MULTIMODAL_API_KEY = os.getenv("MULTIMODAL_API_KEY", "tp-cxit9r7gak3n335w1vewzxjadh7f8d34ahecucld7514moj9")
    MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "mimo-v2.5")
    MULTIMODAL_TIMEOUT = float(os.getenv("MULTIMODAL_TIMEOUT", "60"))
    MULTIMODAL_MAX_TOKENS = int(os.getenv("MULTIMODAL_MAX_TOKENS", "2048"))
    ENABLE_MULTIMODAL_REDIS_CACHE = os.getenv("ENABLE_MULTIMODAL_REDIS_CACHE", "true").lower() == "true"
    MULTIMODAL_CACHE_TTL_SECONDS = int(os.getenv("MULTIMODAL_CACHE_TTL_SECONDS", "3600"))
    MULTIMODAL_CACHE_KEY_PREFIX = os.getenv("MULTIMODAL_CACHE_KEY_PREFIX", "rag:multimodal")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "2.0"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_CONTEXT_MAX_CHARS = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "1800"))
    FAST_RESPONSE_MODE = os.getenv("FAST_RESPONSE_MODE", "true").lower() == "true"
    FAST_LOCAL_EXTRACTION = os.getenv("FAST_LOCAL_EXTRACTION", "true").lower() == "true"
    QUALITY_RETRY_MAX_ELAPSED_SECONDS = float(os.getenv("QUALITY_RETRY_MAX_ELAPSED_SECONDS", "2.2"))
    OPENAI_API_KEY = LLM_API_KEY

    RETRIEVAL_BACKEND = "hybrid"
    COLLECTION_NAME = os.getenv("BM25_COLLECTION", "pdf_documents")

    # 检索策略配置（用户可调）
    RETRIEVAL_CONFIG = {
        "mode": os.getenv("RETRIEVAL_MODE", "hybrid"),  # bm25 / vector / hybrid
        "bm25": {"match_type": os.getenv("RETRIEVAL_BM25_MATCH_TYPE", "phrase")},  # boolean / phrase / fuzzy
        "vector": {"reranker": os.getenv("RETRIEVAL_RERANKER", "llm")},  # llm / tfidf / adaptive
        "hybrid": {
            "bm25_weight": float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.5")),
            "vector_weight": float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.5")),
            "fusion": os.getenv("RETRIEVAL_FUSION", "rrf"),  # weighted_average / rrf
        },
    }

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
    BM25_FILTER_CANDIDATES = int(os.getenv("BM25_FILTER_CANDIDATES", "80"))

    # Milvus 配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_LONG_TERM = os.getenv("MILVUS_COLLECTION_LONG_TERM", "long_term_memory")
    MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    MILVUS_TIMEOUT = float(os.getenv("MILVUS_TIMEOUT", "1"))
    MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
    ENABLE_MILVUS_DOCUMENT_STORE = os.getenv("ENABLE_MILVUS_DOCUMENT_STORE", "true").lower() == "true"
    REBUILD_MILVUS_ON_INIT = os.getenv("REBUILD_MILVUS_ON_INIT", "false").lower() == "true"
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
    SESSION_HISTORY_LIMIT = int(os.getenv("SESSION_HISTORY_LIMIT", "5"))
    SESSION_MEMORY_KEY_PREFIX = os.getenv("SESSION_MEMORY_KEY_PREFIX", "rag:session")
    LONG_TERM_MEMORY_COLLECTION = os.getenv("LONG_TERM_MEMORY_COLLECTION", MILVUS_COLLECTION_LONG_TERM)
    LONG_TERM_MEMORY_THRESHOLD = float(os.getenv("LONG_TERM_MEMORY_THRESHOLD", str(SIMILARITY_THRESHOLD)))
    LONG_TERM_MEMORY_LIMIT = int(os.getenv("LONG_TERM_MEMORY_LIMIT", "20"))
    ENABLE_ANSWER_CACHE = os.getenv("ENABLE_ANSWER_CACHE", "true").lower() == "true"
    ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "30"))
    ENABLE_LONG_TERM_MEMORY_ANSWER = os.getenv("ENABLE_LONG_TERM_MEMORY_ANSWER", "true").lower() == "true"
    ENABLE_MILVUS_LONG_TERM_SEARCH = os.getenv("ENABLE_MILVUS_LONG_TERM_SEARCH", "false").lower() == "true"
    ENABLE_MILVUS_LONG_TERM_SYNC = os.getenv("ENABLE_MILVUS_LONG_TERM_SYNC", "false").lower() == "true"

    PDF_DIR = os.getenv("PDF_DIR", "./data")
    PDF_PARSER_BACKEND = os.getenv("PDF_PARSER_BACKEND", "mineru").lower()
    PDF_PARSE_OUTPUT_DIR = os.getenv("PDF_PARSE_OUTPUT_DIR", "./mineru_output")
    MINERU_OUTPUT_DIR = os.getenv("MINERU_OUTPUT_DIR", PDF_PARSE_OUTPUT_DIR)
    MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
    MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
    MINERU_MODEL_SOURCE = os.getenv("MINERU_MODEL_SOURCE", "modelscope")
    MINERU_COMMAND = os.getenv("MINERU_COMMAND", "mineru")
    MINERU_DEVICE = "cpu"
    MINERU_WSL_DISTRO = os.getenv("MINERU_WSL_DISTRO", "")
    MINERU_WSL_CONDA_ENV = os.getenv("MINERU_WSL_CONDA_ENV", "base")
    MINERU_WSL_CONDA_ROOT = os.getenv("MINERU_WSL_CONDA_ROOT", "/home/li/miniconda3")
    MINERU_REUSE_OUTPUT = os.getenv("MINERU_REUSE_OUTPUT", "true").lower() == "true"
    MINERU_PROCESSING_WINDOW_SIZE = int(os.getenv("MINERU_PROCESSING_WINDOW_SIZE", "8"))
    MINERU_API_MAX_CONCURRENT_REQUESTS = int(os.getenv("MINERU_API_MAX_CONCURRENT_REQUESTS", "1"))
    IMAGES_EXTRACT_DIR = os.getenv("IMAGES_EXTRACT_DIR", "mineru_output/images")
    MULTIMODAL_WORK_ORDER_IMAGES = {
        "organization_chart": {
            "path": os.getenv("ORGANIZATION_CHART_IMAGE", "mineru_output/images/招股说明书2_p039_render.png"),
            "page": int(os.getenv("ORGANIZATION_CHART_PAGE", "39")),
            "source_file": os.getenv("ORGANIZATION_CHART_SOURCE_FILE", "招股说明书2.pdf"),
            "title": "组织结构图",
        },
        "ic_market_growth": {
            "path": os.getenv("IC_MARKET_GROWTH_IMAGE", "mineru_output/images/招股说明书2_p072_render.png"),
            "page": int(os.getenv("IC_MARKET_GROWTH_PAGE", "72")),
            "source_file": os.getenv("IC_MARKET_GROWTH_SOURCE_FILE", "招股说明书2.pdf"),
            "title": "2008年中国IC市场应用结构与增长图",
        },
    }
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


# ========== 混合检索动态权重配置 ==========
HYBRID_WEIGHTS = {
    "default": {"bm25": 0.65, "vector": 0.35},
    "proper_noun": {"bm25": 0.80, "vector": 0.20},
    "numeric": {"bm25": 0.70, "vector": 0.30},
    "abstract": {"bm25": 0.40, "vector": 0.60},
}

PROPER_NOUN_KEYWORDS = [
    "武汉兴图新科",
    "兴图新科",
    "武汉力源信息技术",
    "力源信息",
    "程家明",
    "赵马克",
    "吴祖新",
    "吴志安",
    "某视频技术规范",
    "某情报",
    "指挥控制",
    "国家科技进步一等奖",
]

NUMERIC_KEYWORDS = ["多少", "金额", "收入", "比例", "占比", "比重", "注册资本", "发行股数"]

Config.HYBRID_WEIGHTS = HYBRID_WEIGHTS
Config.PROPER_NOUN_KEYWORDS = PROPER_NOUN_KEYWORDS
Config.NUMERIC_KEYWORDS = NUMERIC_KEYWORDS
