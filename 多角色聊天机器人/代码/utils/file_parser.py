"""
文件解析工具类
支持格式: .docx, .pdf, .txt
"""

import os
import base64
import shutil
import zipfile
import xml.etree.ElementTree as ET

# ========== DOCX 解析依赖 ==========
try:
    from docx import Document
except Exception:
    Document = None

# ========== PDF 解析依赖 ==========
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import fitz
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import requests
except Exception:
    requests = None


class FileParser:
    """文件解析工具类"""

    @staticmethod
    def parse_file(file_path: str) -> str:
        """主入口：根据扩展名调用对应解析器"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.docx':
            return FileParser._parse_docx(file_path)
        elif ext == '.pdf':
            if FileParser._use_multimodal_pdf_parser():
                try:
                    result = FileParser._parse_pdf_with_multimodal(file_path)
                    if FileParser._has_useful_text(result):
                        print(f"PDF 多模态解析成功: {os.path.basename(file_path)}")
                        return result
                except Exception as exc:
                    print(f"PDF 多模态解析失败，回退到原有流程: {exc}")
            return FileParser._parse_pdf(file_path)
        elif ext == '.txt':
            return FileParser._parse_txt(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    # ========== DOCX 解析 ==========

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """解析 DOCX 文件，优先使用 python-docx，失败则使用内置解析"""
        try:
            if Document is not None:
                doc = Document(file_path)
                text = [p.text for p in doc.paragraphs if p.text.strip()]
                return '\n'.join(text)
            return FileParser._parse_docx_with_stdlib(file_path)
        except Exception as e:
            raise Exception(f"解析.docx文件失败: {str(e)}")

    @staticmethod
    def _parse_docx_with_stdlib(file_path: str) -> str:
        """使用标准库解析 DOCX（不依赖 python-docx）"""
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text_parts = []

        with zipfile.ZipFile(file_path) as archive:
            xml_names = [name for name in archive.namelist()
                        if name.startswith("word/") and name.endswith(".xml")
                        and (name == "word/document.xml" or name.startswith("word/header") or name.startswith("word/footer"))]

            for xml_name in xml_names:
                root = ET.fromstring(archive.read(xml_name))
                for paragraph in root.findall(".//w:p", namespace):
                    runs = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text and node.text.strip()]
                    if runs:
                        text_parts.append("".join(runs))

        return "\n".join(text_parts)

    # ========== PDF 解析（核心）==========

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """
        解析 PDF 文件
        流程: 1. PyMuPDF提取文字 → 2. PyPDF2提取文字 → 3. OCR识别
        """
        try:
            # 方法1: PyMuPDF (fitz) 直接提取文字层
            if fitz is not None:
                with fitz.open(file_path) as doc:
                    text = [page.get_text("text", sort=True) for page in doc]
                    result = '\n'.join(text)
                    if FileParser._has_useful_text(result):
                        print(f"PDF 直接提取成功: {os.path.basename(file_path)}")
                        return result

            # 方法2: PyPDF2 直接提取文字层
            if PyPDF2 is not None:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = [page.extract_text() for page in reader.pages if page.extract_text()]
                    result = '\n'.join(text)
                    if FileParser._has_useful_text(result):
                        print(f"PDF 直接提取成功: {os.path.basename(file_path)}")
                        return result

            # 方法3: OCR 识别（用于扫描版 PDF）
            print(f"直接提取无内容，尝试 OCR 识别: {os.path.basename(file_path)}")
            ocr_result = FileParser._parse_pdf_with_ocr(file_path)
            if FileParser._has_useful_text(ocr_result):
                print(f"OCR 识别成功: {os.path.basename(file_path)}")
                return ocr_result

            raise RuntimeError("PDF 无法提取任何文本内容")

        except Exception as e:
            raise Exception(f"解析.pdf文件失败: {str(e)}")

    @staticmethod
    def _use_multimodal_pdf_parser() -> bool:
        """Return whether the optional multimodal PDF parser is enabled."""
        value = os.getenv("PDF_ENABLE_MULTIMODAL")
        if value is not None:
            return value.lower() in {"1", "true", "yes", "on"}
        try:
            from config.config import settings
            return bool(getattr(settings, "PDF_ENABLE_MULTIMODAL", False))
        except Exception:
            return False

    @staticmethod
    def _parse_pdf_with_multimodal(file_path: str) -> str:
        """
        Parse complex PDFs with a vision-capable chat-completions model.

        This path is optional and disabled by default. It keeps the native PDF
        text layer as text, extracts detected tables as Markdown, and only sends
        cropped visual regions such as charts, figures, and image/table areas to
        the configured multimodal model.
        """
        if requests is None:
            raise RuntimeError("requests is not available")
        if fitz is None:
            raise RuntimeError("PyMuPDF is not available")

        try:
            from config.config import settings
        except Exception:
            settings = None

        def _setting(name: str, fallback=""):
            value = os.getenv(name)
            if value not in (None, ""):
                return value
            return getattr(settings, name, fallback) if settings else fallback

        api_key = _setting("PDF_MM_API_KEY") or _setting("API_KEY")
        api_base_url = _setting("PDF_MM_API_BASE_URL") or _setting("API_BASE_URL")
        model_name = _setting("PDF_MM_MODEL") or _setting("MODEL_NAME")
        dpi = int(_setting("PDF_MM_DPI", 180) or 180)
        max_pages = int(_setting("PDF_MM_MAX_PAGES", 0) or 0)
        max_visuals_per_page = int(_setting("PDF_MM_MAX_VISUALS_PER_PAGE", 4) or 4)

        if not api_key:
            raise RuntimeError("PDF_MM_API_KEY is empty")
        if not api_base_url:
            raise RuntimeError("PDF_MM_API_BASE_URL is empty")
        if not model_name:
            raise RuntimeError("PDF_MM_MODEL is empty")

        system_prompt = (
            "You are a PDF document parsing assistant. Convert the supplied "
            "native PDF text, extracted tables, and cropped visual elements into "
            "editable Chinese Markdown. Do not perform generic OCR on the whole "
            "page. Preserve headings, body text, lists, tables, and chart "
            "information. Convert tables to Markdown tables where possible. "
            "For charts, describe the topic, axes or legend, trend, and key "
            "numbers visible in the cropped visual elements. Output only the "
            "parsed content."
        )

        def _response_text(payload: dict) -> str:
            choice = (payload.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if text:
                            parts.append(text)
                return "\n".join(parts).strip()
            return str(content or "").strip()

        def _rect_area(rect) -> float:
            return max(0, rect.width) * max(0, rect.height)

        def _overlap_ratio(rect_a, rect_b) -> float:
            inter = fitz.Rect(rect_a)
            inter.intersect(rect_b)
            inter_area = _rect_area(inter)
            base_area = min(_rect_area(rect_a), _rect_area(rect_b))
            return inter_area / base_area if base_area > 0 else 0

        def _clip_rect(page, rect, margin=4):
            clipped = fitz.Rect(rect)
            clipped.x0 = max(page.rect.x0, clipped.x0 - margin)
            clipped.y0 = max(page.rect.y0, clipped.y0 - margin)
            clipped.x1 = min(page.rect.x1, clipped.x1 + margin)
            clipped.y1 = min(page.rect.y1, clipped.y1 + margin)
            return clipped

        def _render_region(page, rect) -> bytes:
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix, alpha=False, clip=rect)
            return pix.tobytes("png")

        def _extract_tables(page):
            tables = []
            table_rects = []
            try:
                finder = page.find_tables()
            except Exception:
                return tables, table_rects

            page_area = _rect_area(page.rect)
            for table_index, table in enumerate(getattr(finder, "tables", []), start=1):
                rect = fitz.Rect(table.bbox)
                rect_area = _rect_area(rect)
                if rect_area <= 0:
                    continue
                if rect_area / page_area > 0.65:
                    continue
                if getattr(table, "row_count", 0) < 2 or getattr(table, "col_count", 0) < 2:
                    continue

                try:
                    table_md = table.to_markdown(clean=True).strip()
                except Exception:
                    table_md = ""
                if not table_md:
                    continue

                tables.append(f"Table {table_index}:\n{table_md}")
                table_rects.append(rect)

            return tables, table_rects

        def _visual_region_rects(page, table_rects):
            candidates = []
            page_area = _rect_area(page.rect)

            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") == 1 and "bbox" in block:
                    candidates.append(fitz.Rect(block["bbox"]))

            try:
                drawings = page.get_drawings()
            except Exception:
                drawings = []
            for drawing in drawings:
                rect = fitz.Rect(drawing.get("rect"))
                rect_area = _rect_area(rect)
                if rect.width < 80 or rect.height < 35:
                    continue
                if rect_area < 3000 or rect_area / page_area > 0.55:
                    continue
                candidates.append(rect)

            candidates.extend(table_rects)
            candidates = [_clip_rect(page, rect) for rect in candidates]
            candidates.sort(key=_rect_area, reverse=True)

            selected = []
            for rect in candidates:
                if _rect_area(rect) < 3000:
                    continue
                if any(_overlap_ratio(rect, existing) > 0.70 for existing in selected):
                    continue
                selected.append(rect)
                if len(selected) >= max_visuals_per_page:
                    break

            return selected

        def _parse_page(page_no: int, total_pages: int, page_text: str, table_markdowns: list, visual_images: list) -> str:
            prompt = (
                f"Parse page {page_no}/{total_pages} from this PDF.\n"
                "Requirements:\n"
                "1. Use the native text layer as the main source for ordinary text.\n"
                "2. Use extracted Markdown tables as structured table sources.\n"
                "3. Use the cropped visual elements to analyze charts, figures, embedded images, and visually complex tables.\n"
                "4. For each chart, describe its title/topic, axes or legend, trend, and key visible numbers.\n"
                "5. Remove repeated headers, footers, page numbers, watermarks, and decorative noise.\n"
                "6. Output only editable Chinese Markdown.\n"
            )
            if page_text.strip():
                prompt += "\nNative PDF text layer:\n" + page_text[:10000]
            if table_markdowns:
                prompt += "\n\nExtracted tables:\n" + "\n\n".join(table_markdowns)[:6000]
            if visual_images:
                prompt += (
                    f"\n\nThere are {len(visual_images)} cropped visual element(s) "
                    "attached after this text. Analyze them for chart/table/figure "
                    "information and merge the findings into the Markdown."
                )

            content = [{"type": "text", "text": prompt}]
            for image_bytes in visual_images:
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                })

            request_payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
            }
            response = requests.post(
                api_base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=(20, 120),
            )
            if response.status_code != 200:
                raise RuntimeError(f"multimodal API returned {response.status_code}: {response.text[:300]}")
            return _response_text(response.json())

        text_parts = []
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            print(f"开始多模态解析 PDF: {os.path.basename(file_path)}，共 {total_pages} 页")
            for page_index, page in enumerate(doc, start=1):
                if max_pages > 0 and page_index > max_pages:
                    print(f"已达到 PDF_MM_MAX_PAGES={max_pages}，停止多模态解析")
                    break
                page_text = page.get_text("text", sort=True)
                table_markdowns, table_rects = _extract_tables(page)
                visual_images = [
                    _render_region(page, rect)
                    for rect in _visual_region_rects(page, table_rects)
                ]
                page_result = _parse_page(
                    page_index,
                    total_pages,
                    page_text,
                    table_markdowns,
                    visual_images,
                )
                if not page_result.strip() and FileParser._has_useful_text(page_text):
                    page_result = page_text.strip()
                if page_result.strip():
                    text_parts.append(f"【第 {page_index} 页】\n{page_result.strip()}")

        result = "\n\n".join(text_parts).strip()
        if not FileParser._has_useful_text(result):
            raise RuntimeError("multimodal PDF parser returned no useful text")
        return result

    @staticmethod
    def _parse_pdf_with_ocr(file_path: str) -> str:
        """OCR 识别扫描版 PDF"""
        if pytesseract is None or Image is None:
            raise RuntimeError("缺少 OCR 依赖，请安装: pip install pytesseract pdf2image")

        # 设置 tesseract 路径
        tesseract_cmd = os.getenv("TESSERACT_CMD", r"D:\tmp\tesseract.exe")
        if os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        lang = "chi_sim+eng"  # 中文简体 + 英文
        dpi = 200

        text = []
        with fitz.open(file_path) as doc:
            print(f"PDF 共 {len(doc)} 页，开始 OCR 识别...")
            for i, page in enumerate(doc):
                # 将 PDF 页面转为图片
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # OCR 识别
                page_text = pytesseract.image_to_string(img, lang=lang)
                if page_text.strip():
                    text.append(page_text)

                if (i + 1) % 20 == 0:
                    print(f"已识别 {i+1}/{len(doc)} 页")

        return '\n'.join(text)

    @staticmethod
    def _has_useful_text(text: str) -> bool:
        """判断文本是否足够有意义（去除空白后长度 >= 50）"""
        clean = "".join(text.split())
        return len(clean) >= 50

    # ========== TXT 解析 ==========

    @staticmethod
    def _parse_txt(file_path: str) -> str:
        """解析 TXT 文件，自动检测编码"""
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # 降级方案
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ========== 工具方法 ==========

    @staticmethod
    def list_files(directory: str) -> list:
        """列出目录下所有支持的文件 (.docx, .txt, .pdf)"""
        if not os.path.exists(directory):
            return []

        supported = ['.docx', '.txt']
        if PyPDF2 is not None or fitz is not None:
            supported.append('.pdf')

        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in supported:
                files.append(file_path)
        return files
