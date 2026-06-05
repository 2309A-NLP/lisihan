# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import requests
from PIL import Image
from requests import HTTPError

from src.config import Config


DEFAULT_BASE_URL = Config.MULTIMODAL_API_BASE_URL
DEFAULT_API_KEY = Config.MULTIMODAL_API_KEY
DEFAULT_MODEL = Config.MULTIMODAL_MODEL


TABLE_DETECTION_PROMPT = """请分析这张PDF页面截图，找出页面中的所有表格区域。

对于每个表格，返回以下JSON格式：
{
  "has_table": true/false,
  "tables": [
    {
      "table_id": 1,
      "bbox_percent": [x0, y0, x1, y1],
      "is_partial": true/false,
      "continues_from_previous": true/false,
      "continues_to_next": true/false
    }
  ]
}

要求：
- bbox_percent 使用百分比数值，范围 0-100。
- 只输出JSON，不要输出解释文字。
- 忽略水印、页眉、页脚、页码等非正文信息。
- 如果没有表格，返回 {"has_table": false, "tables": []}。
"""


TABLE_EXTRACTION_PROMPT = """请分析这张完整的表格图片，忽略水印、页眉、页脚。

要求：
1. 逐行读取所有数据。
2. 识别表头和各列含义。
3. 输出为Markdown表格格式。
4. 如果表格有合计行，请保留。
5. 不要遗漏任何一行数据。
"""


@dataclass
class PageImage:
    page: int
    path: str
    width: int
    height: int


@dataclass
class TableRegion:
    page: int
    table_id: int
    bbox_percent: List[float]
    bbox_pixels: List[int]
    is_partial: bool = False
    continues_from_previous: bool = False
    continues_to_next: bool = False
    page_image_path: str = ""
    crop_path: str = ""


@dataclass
class ExtractedTable:
    table_index: int
    pages: List[int]
    image_path: str
    markdown: str
    regions: List[TableRegion] = field(default_factory=list)


class MultimodalTableExtractor:
    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        output_dir: str | Path = "parsed_output/multimodal_tables",
        timeout: float = Config.MULTIMODAL_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.timeout = timeout

    def image_to_base64(self, image_path: str | Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def ask_image(self, image_path: str | Path, prompt: str, max_tokens: int = 1024) -> str:
        base64_image = self.image_to_base64(image_path)
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
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(f"multimodal API failed: {response.status_code} {response.text}") from exc
        return response.json()["choices"][0]["message"]["content"]

    def pdf_to_images(self, pdf_path: str | Path, zoom: float = 2.0) -> List[PageImage]:
        pdf_file = Path(pdf_path)
        page_dir = self.output_dir / pdf_file.stem / "pages"
        page_dir.mkdir(parents=True, exist_ok=True)

        pages: List[PageImage] = []
        with fitz.open(str(pdf_file)) as doc:
            for page_index, page in enumerate(doc, start=1):
                image_path = page_dir / f"page_{page_index:04d}.png"
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                pix.save(str(image_path))
                pages.append(PageImage(page=page_index, path=str(image_path), width=pix.width, height=pix.height))
        return pages

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise ValueError(f"model did not return JSON: {text}")
            return json.loads(match.group(0))

    def detect_tables_on_page(self, page_image: PageImage) -> List[TableRegion]:
        response = self.ask_image(page_image.path, TABLE_DETECTION_PROMPT, max_tokens=1024)
        payload = self._extract_json_object(response)
        if not payload.get("has_table"):
            return []

        regions: List[TableRegion] = []
        for idx, item in enumerate(payload.get("tables", []), start=1):
            bbox_percent = [self._parse_percent_value(value) for value in item.get("bbox_percent", [])[:4]]
            if len(bbox_percent) != 4:
                continue
            bbox_pixels = self._percent_bbox_to_pixels(bbox_percent, page_image.width, page_image.height)
            regions.append(
                TableRegion(
                    page=page_image.page,
                    table_id=int(item.get("table_id", idx)),
                    bbox_percent=bbox_percent,
                    bbox_pixels=bbox_pixels,
                    is_partial=bool(item.get("is_partial", False)),
                    continues_from_previous=bool(item.get("continues_from_previous", False)),
                    continues_to_next=bool(item.get("continues_to_next", False)),
                    page_image_path=page_image.path,
                )
            )
        return regions

    def _parse_percent_value(self, value: Any) -> float:
        if isinstance(value, str):
            value = value.strip().rstrip("%")
        return float(value)

    def _percent_bbox_to_pixels(self, bbox_percent: List[float], width: int, height: int) -> List[int]:
        x0, y0, x1, y1 = bbox_percent
        x0_px = max(0, min(width, round(width * x0 / 100)))
        y0_px = max(0, min(height, round(height * y0 / 100)))
        x1_px = max(0, min(width, round(width * x1 / 100)))
        y1_px = max(0, min(height, round(height * y1 / 100)))
        if x1_px <= x0_px or y1_px <= y0_px:
            raise ValueError(f"invalid bbox_percent: {bbox_percent}")
        return [x0_px, y0_px, x1_px, y1_px]

    def crop_table_regions(self, pdf_stem: str, regions: List[TableRegion]) -> List[TableRegion]:
        crop_dir = self.output_dir / pdf_stem / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        for region in regions:
            with Image.open(region.page_image_path) as image:
                cropped = image.crop(tuple(region.bbox_pixels))
                crop_path = crop_dir / f"page_{region.page:04d}_table_{region.table_id:02d}.png"
                cropped.save(crop_path)
                region.crop_path = str(crop_path)
        return regions

    def group_cross_page_tables(self, regions: List[TableRegion]) -> List[List[TableRegion]]:
        groups: List[List[TableRegion]] = []
        current: List[TableRegion] = []

        for region in sorted(regions, key=lambda item: (item.page, item.table_id)):
            if region.continues_from_previous and current:
                current.append(region)
            else:
                if current:
                    groups.append(current)
                current = [region]

            if not region.continues_to_next:
                groups.append(current)
                current = []

        if current:
            groups.append(current)
        return groups

    def stitch_table_group(self, pdf_stem: str, table_index: int, regions: List[TableRegion]) -> str:
        stitched_dir = self.output_dir / pdf_stem / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)

        images = [Image.open(region.crop_path).convert("RGB") for region in regions]
        if len(images) == 1:
            output_path = stitched_dir / f"table_{table_index:04d}_p{regions[0].page:04d}.png"
            images[0].save(output_path)
            images[0].close()
            return str(output_path)

        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        stitched = Image.new("RGB", (width, height), "white")
        y = 0
        for image in images:
            stitched.paste(image, (0, y))
            y += image.height
            image.close()

        page_part = f"p{regions[0].page:04d}_to_p{regions[-1].page:04d}"
        output_path = stitched_dir / f"table_{table_index:04d}_{page_part}.png"
        stitched.save(output_path)
        return str(output_path)

    def extract_table_markdown(self, table_image_path: str | Path) -> str:
        return self.ask_image(table_image_path, TABLE_EXTRACTION_PROMPT, max_tokens=2048)

    def extract(self, pdf_path: str | Path, zoom: float = 2.0) -> Dict[str, Any]:
        pdf_file = Path(pdf_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        page_images = self.pdf_to_images(pdf_file, zoom=zoom)
        all_regions: List[TableRegion] = []
        for page_image in page_images:
            page_regions = self.detect_tables_on_page(page_image)
            all_regions.extend(page_regions)

        self.crop_table_regions(pdf_file.stem, all_regions)
        groups = self.group_cross_page_tables(all_regions)

        extracted_tables: List[ExtractedTable] = []
        for table_index, group in enumerate(groups, start=1):
            stitched_path = self.stitch_table_group(pdf_file.stem, table_index, group)
            markdown = self.extract_table_markdown(stitched_path)
            extracted_tables.append(
                ExtractedTable(
                    table_index=table_index,
                    pages=[region.page for region in group],
                    image_path=stitched_path,
                    markdown=markdown,
                    regions=group,
                )
            )

        result = {
            "source_file": pdf_file.name,
            "page_images": [page.__dict__ for page in page_images],
            "regions": [region.__dict__ for region in all_regions],
            "tables": [
                {
                    "table_index": table.table_index,
                    "pages": table.pages,
                    "image_path": table.image_path,
                    "markdown": table.markdown,
                    "regions": [region.__dict__ for region in table.regions],
                }
                for table in extracted_tables
            ],
        }

        result_path = self.output_dir / pdf_file.stem / "tables.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal PDF table extractor")
    parser.add_argument("pdf_path", help="PDF file path")
    parser.add_argument("--output_dir", default="parsed_output/multimodal_tables", help="Output directory")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Multimodal model name")
    parser.add_argument("--zoom", type=float, default=2.0, help="PDF rendering zoom")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    extractor = MultimodalTableExtractor(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        output_dir=args.output_dir,
    )
    result = extractor.extract(args.pdf_path, zoom=args.zoom)
    print(json.dumps({"source_file": result["source_file"], "table_count": len(result["tables"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
