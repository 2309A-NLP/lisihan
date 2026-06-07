# -*- coding: utf-8 -*-
"""实体关系抽取入口。

中文说明：本模块是工单八要求的顶层入口，内部复用 src.graph_rag.extractor。
"""

from src.graph_rag.extractor import ExtractedEntity, ExtractedRelation, GraphEntityExtractor

__all__ = ["ExtractedEntity", "ExtractedRelation", "GraphEntityExtractor"]
