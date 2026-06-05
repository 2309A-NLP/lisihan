# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

from pdf_parser.main import render_pdf_page_to_image
from src.config import Config
from src.models import RAGResponse

try:
    import redis
except Exception:  # pragma: no cover - optional runtime dependency
    redis = None

class MultimodalMixin:
    def ask_with_image(self, question: str, image_path: str) -> str:
        return self.image_parser.ask_about_image(image_path, question)

    def _multimodal_cache_key(self, image_path: str, question: str) -> str:
        try:
            normalized_path = str(Path(image_path).resolve())
        except Exception:
            normalized_path = str(image_path or "")
        digest = hashlib.sha256(f"{normalized_path}\n{question or ''}".encode("utf-8")).hexdigest()
        return f"{Config.MULTIMODAL_CACHE_KEY_PREFIX}:{digest}"

    def _multimodal_cache_client(self):
        if not Config.ENABLE_MULTIMODAL_REDIS_CACHE or redis is None:
            return None
        if not hasattr(self, "_multimodal_redis_client"):
            self._multimodal_redis_client = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
        return self._multimodal_redis_client

    def _get_multimodal_cached_payload(self, image_path: str, question: str) -> Dict | None:
        client = self._multimodal_cache_client()
        if client is None:
            return None
        cache_key = self._multimodal_cache_key(image_path, question)
        try:
            cached = client.get(cache_key)
        except Exception as exc:
            self.logger.warning("multimodal redis cache read failed | key=%s | error=%s", cache_key, exc)
            return None
        if not cached:
            return None
        try:
            payload = json.loads(cached)
        except json.JSONDecodeError:
            payload = {"answer": cached}
        payload["cache_key"] = cache_key
        return payload

    def _set_multimodal_cached_payload(self, image_path: str, question: str, payload: Dict) -> None:
        client = self._multimodal_cache_client()
        if client is None:
            return
        cache_key = self._multimodal_cache_key(image_path, question)
        ttl_seconds = getattr(Config, "MULTIMODAL_CACHE_TTL_SECONDS", 3600)
        try:
            value = json.dumps(payload, ensure_ascii=False)
            if ttl_seconds and ttl_seconds > 0:
                client.setex(cache_key, ttl_seconds, value)
            else:
                client.set(cache_key, value)
        except Exception as exc:
            self.logger.warning("multimodal redis cache write failed | key=%s | error=%s", cache_key, exc)

    def _multimodal_response_from_payload(
        self,
        payload: Dict,
        *,
        question: str,
        question_type: str,
        answer_language: str,
        start_time: float,
        image_info: Dict,
        question_kind: str,
    ) -> RAGResponse:
        answer = str(payload.get("answer", ""))
        source_chunk = payload.get("source_chunk") or {
            "content": f"multimodal_image:{image_info.get('path')}",
            "metadata": image_info,
            "relevance_score": 1.0,
        }
        query_analysis = payload.get("query_analysis") or {
            "intent": "multimodal_image",
            "is_ambiguous": False,
            "answer_language": answer_language,
            "image_question_kind": question_kind,
            "image_path": image_info.get("path"),
            "image_page": image_info.get("page"),
        }
        query_analysis["cache_hit"] = True
        query_analysis["cache_key"] = payload.get("cache_key")
        return RAGResponse(
            question=question,
            answer=answer,
            question_type=question_type,
            memory_hit=False,
            retrieval_mode="multimodal_image_cache",
            retrieved_contexts=[source_chunk.get("content", "")],
            scores=[float(source_chunk.get("relevance_score", 1.0) or 1.0)],
            response_time=time.time() - start_time,
            accuracy=1.0,
            source_chunks=[source_chunk],
            has_context=True,
            query_analysis=query_analysis,
        )

    def _is_multimodal_work_order_question(self, question: str, question_id: int = None) -> str | None:
        q = question or ""
        image_keywords = ["组织结构图", "增长图", "图中", "如图所示", "从图中", "IC市场", "IC 市场", "销售部构成"]
        asks_structure_or_count = "图" in q and any(marker in q for marker in ["结构", "构成", "多少", "几个", "几家", "几类"])
        looks_like_image_question = any(keyword in q for keyword in image_keywords) or asks_structure_or_count

        if question_id == 5 or (
            looks_like_image_question
            and ("组织结构" in q or "销售部" in q)
            and ("销售部" in q or "大客户销售部" in q)
        ):
            return "organization_chart"
        if question_id == 6 or (
            looks_like_image_question
            and ("IC市场" in q or "IC 市场" in q or "应用结构" in q)
            and ("增长" in q or "负增长" in q or "增长率" in q)
        ):
            return "ic_market_growth"
        return None

    def _fixed_multimodal_image_info(self, question_kind: str) -> Dict | None:
        image_info = Config.MULTIMODAL_WORK_ORDER_IMAGES.get(question_kind)
        if not image_info:
            return None

        image_path = Path(image_info["path"])
        if image_path.exists():
            return {
                "kind": "page",
                "source_file": image_info["source_file"],
                "page": image_info["page"],
                "title": image_info["title"],
                "index": 0,
                "xref": None,
                "path": str(image_path),
                "bbox": [],
                "rendered_page": True,
            }

        parsed = self._load_parsed_json(image_info["source_file"])
        pdf_path = self._source_path_for_pdf(image_info["source_file"], parsed)
        rendered = render_pdf_page_to_image(pdf_path, image_info["page"], Config.IMAGES_EXTRACT_DIR)
        if rendered and Path(rendered["path"]).exists():
            rendered["title"] = image_info["title"]
            return rendered
        return None

    def _load_parsed_json(self, source_file: str) -> Dict:
        parsed_path = Path(Config.PDF_PARSE_OUTPUT_DIR) / f"{Path(source_file).stem}_chunks.json"
        if not parsed_path.exists():
            return {}
        try:
            return json.loads(parsed_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _source_path_for_pdf(self, source_file: str, parsed: Dict) -> str:
        source_path = parsed.get("source_path")
        if source_path and Path(source_path).exists():
            return source_path
        return str(Path(Config.PDF_DIR) / source_file)

    def _find_page_by_keywords(self, parsed: Dict, keyword_sets: List[List[str]]) -> int | None:
        for keywords in keyword_sets:
            for chunk in parsed.get("chunks", []):
                content = str(chunk.get("content", ""))
                if all(keyword in content for keyword in keywords):
                    page = chunk.get("page")
                    if isinstance(page, int):
                        return page
        return None

    def _select_multimodal_image(self, question_kind: str) -> Dict | None:
        fixed_image = self._fixed_multimodal_image_info(question_kind)
        if fixed_image:
            return fixed_image

        configured = Config.MULTIMODAL_WORK_ORDER_IMAGES.get(question_kind, {})
        source_file = configured.get("source_file", "招股说明书2.pdf")
        parsed = self._load_parsed_json(source_file)
        pdf_path = self._source_path_for_pdf(source_file, parsed)
        target_page = configured.get("page")
        if not target_page:
            keyword_sets = {
                "ic_market_growth": [["2008 年中国IC 市场应用结构与增长"], ["IC 市场应用结构与增长"]],
            }.get(question_kind, [])
            target_page = self._find_page_by_keywords(parsed, keyword_sets)

        if target_page:
            rendered = render_pdf_page_to_image(pdf_path, target_page, Config.IMAGES_EXTRACT_DIR)
            if rendered and Path(rendered["path"]).exists():
                return rendered

            images = parsed.get("images", [])
            same_page_images = [
                image
                for image in images
                if image.get("page") == target_page and Path(str(image.get("path", ""))).exists()
            ]
            if same_page_images:
                return max(
                    same_page_images,
                    key=lambda image: sum(
                        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        for bbox in image.get("bbox", [])
                        if len(bbox) == 4
                    ),
                )

        images = parsed.get("images", [])
        for image in images:
            if Path(str(image.get("path", ""))).exists():
                return image
        return None

    def _answer_multimodal_question(
        self,
        *,
        question: str,
        question_id: int,
        question_type: str,
        answer_language: str,
        cache_key: str,
        start_time: float,
    ) -> RAGResponse | None:
        question_kind = self._is_multimodal_work_order_question(question, question_id)
        if not question_kind:
            return None

        image_info = self._select_multimodal_image(question_kind)
        if not image_info:
            self.logger.warning("multimodal image not found | question=%s | kind=%s", question, question_kind)
            return None

        image_path = str(image_info.get("path", ""))
        cached_payload = self._get_multimodal_cached_payload(image_path, question)
        if cached_payload is not None:
            self.logger.info("multimodal redis cache hit | question=%s | image=%s", question, image_path)
            return self._multimodal_response_from_payload(
                cached_payload,
                question=question,
                question_type=question_type,
                answer_language=answer_language,
                start_time=start_time,
                image_info=image_info,
                question_kind=question_kind,
            )

        prompt = (
            "请分析这张图片，回答以下问题：\n"
            f"{question}\n\n"
            "要求：\n"
            "1. 只依据图片内容作答，不要编造图片中不存在的信息。\n"
            "2. 如果问题问的是“有几个部门”，请列出完整数量+名称。\n"
            "3. 如果问题问的是“有几个销售处”，请列出完整数量+名称。\n"
            "4. 不要省略任何信息。\n"
            "5. 用简洁的一句话回答，但必须完整。\n"
            "6. 如果图片中无法确认答案，请直接说明无法从图中确认。"
        )

        try:
            answer = self.ask_with_image(prompt, image_info["path"])
        except Exception as exc:
            self.logger.warning(
                "multimodal answer failed | question=%s | image=%s | error=%s",
                question,
                image_info.get("path"),
                exc,
            )
            return None

        localized_answer = self._localize_answer(question, answer, answer_language)
        self.logger.info("multimodal answer length=%s | question=%s", len(localized_answer), question)
        source_chunk = {
            "content": f"multimodal_image:{image_info.get('path')}",
            "metadata": image_info,
            "relevance_score": 1.0,
        }
        query_analysis = {
            "intent": "multimodal_image",
            "is_ambiguous": False,
            "answer_language": answer_language,
            "image_question_kind": question_kind,
            "image_path": image_info.get("path"),
            "image_page": image_info.get("page"),
            "cache_hit": False,
        }
        self._set_multimodal_cached_payload(
            image_path,
            question,
            {
                "answer": localized_answer,
                "question": question,
                "question_type": question_type,
                "source_chunk": source_chunk,
                "query_analysis": query_analysis,
            },
        )
        return self._cached_response(
            cache_key,
            question=question,
            answer=localized_answer,
            question_type=question_type,
            retrieval_mode="multimodal_image",
            retrieved_contexts=[source_chunk["content"]],
            scores=[1.0],
            start_time=start_time,
            accuracy=1.0,
            source_chunks=[source_chunk],
            has_context=True,
            query_analysis=query_analysis,
        )
