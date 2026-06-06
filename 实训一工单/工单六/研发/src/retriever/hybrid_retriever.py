# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""Compatibility entrypoint for hybrid retrieval."""

from __future__ import annotations

from .hybrid import HybridRetriever, get_dynamic_weights


__all__ = ["HybridRetriever", "get_dynamic_weights"]
