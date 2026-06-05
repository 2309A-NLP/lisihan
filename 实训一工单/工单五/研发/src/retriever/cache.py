# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import Config
from src.document import Document
from utils.logger import get_logger

from .models import _RetrievalHit


logger = get_logger(__name__)


class SearchCacheMixin:
    def _get_cached_search(self, cache_key: Tuple[str, int, str, str]) -> Optional[List[Tuple[Document, float]]]:
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            self._search_cache.move_to_end(cache_key)
            logger.info(
                "hybrid search cache hit | query=%s | mode=%s | source_file=%s | hits=%s",
                cache_key[0],
                cache_key[2],
                cache_key[3],
                len(cached),
            )
        return cached

    def _set_cached_search(self, cache_key: Tuple[str, int, str, str], results: List[Tuple[Document, float]]) -> None:
        self._search_cache[cache_key] = results
        self._search_cache.move_to_end(cache_key)
        while len(self._search_cache) > self._search_cache_maxsize:
            self._search_cache.popitem(last=False)
