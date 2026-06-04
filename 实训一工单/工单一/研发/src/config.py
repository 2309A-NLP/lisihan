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
    OPENAI_API_KEY = LLM_API_KEY

    RETRIEVAL_BACKEND = "bm25"
    COLLECTION_NAME = os.getenv("BM25_COLLECTION", "pdf_documents")

    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))

    PDF_DIR = os.getenv("PDF_DIR", "./data")
    PDF_PARSER_BACKEND = os.getenv("PDF_PARSER_BACKEND", "mineru").lower()
    MINERU_OUTPUT_DIR = os.getenv("MINERU_OUTPUT_DIR", "./mineru_output")
    MINERU_WSL_DISTRO = os.getenv("MINERU_WSL_DISTRO", "")
    MINERU_CONDA_PREFIX = os.getenv("MINERU_CONDA_PREFIX", "/home/li/miniconda3")
    MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
    MINERU_USE_CACHE = os.getenv("MINERU_USE_CACHE", "1").lower() not in {"0", "false", "no"}
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
    ]

