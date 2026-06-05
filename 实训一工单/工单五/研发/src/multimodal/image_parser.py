# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import base64
from typing import Optional

import requests
from requests import HTTPError

from src.config import Config


class MultimodalImageParser:
    def __init__(
        self,
        api_key: str,
        model: str = "mimo-v2.5",
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def ask(
        self,
        image_path: str,
        question: str,
        timeout: Optional[float] = None,
        max_tokens: int | None = None,
    ) -> str:
        base64_image = self.image_to_base64(image_path)
        request_timeout = Config.MULTIMODAL_TIMEOUT if timeout is None else timeout
        token_limit = Config.MULTIMODAL_MAX_TOKENS if max_tokens is None else max_tokens

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                "max_tokens": token_limit,
                "temperature": 0.1,
            },
            timeout=request_timeout,
        )
        try:
            response.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(f"multimodal API failed: {response.status_code} {response.text}") from exc
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def ask_about_image(self, image_path: str, question: str, max_tokens: int | None = None) -> str:
        return self.ask(image_path, question, max_tokens=max_tokens)
