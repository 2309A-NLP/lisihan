# -*- coding: utf-8 -*-
"""向量/混合检索入口。

中文说明：当前项目已实现 BM25 + 向量 + RRF 混合检索。为了保持前七个工单稳定，
这里提供兼容入口，内部复用 HybridRetriever。若需要 Chroma，可在此模块扩展。
"""

from src.retriever import HybridRetriever

VectorStore = HybridRetriever

__all__ = ["HybridRetriever", "VectorStore"]
