# PyCharm 调用 WSL MinerU

本项目已支持从 Windows/PyCharm 调用安装在 WSL `/home/li/miniconda3` 中的 MinerU。

## 直接测试 MinerU

在 PyCharm 中右键运行：

```powershell
examples/parse_pdf_with_mineru_wsl.py
```

或在终端运行：

```powershell
.\.venv\Scripts\python.exe examples\parse_pdf_with_mineru_wsl.py data\招股说明书1.pdf
```

默认输出到 Windows 项目目录：

```text
mineru_output/<PDF文件名>/auto/<PDF文件名>.md
```

## 接入问答系统

`src.pdf_parser.PDFParser` 默认使用 `PDF_PARSER_BACKEND=auto`：

- 优先读取或调用 WSL MinerU，使用 MinerU 生成的 Markdown 切片。
- 如果 WSL/MinerU 不可用，自动回退到 PyMuPDF 文本抽取。

如需强制使用 MinerU，可在 `.env` 中设置：

```text
PDF_PARSER_BACKEND=mineru
```

如需只用 PyMuPDF：

```text
PDF_PARSER_BACKEND=pymupdf
```

## 可配置项

```text
MINERU_CONDA_PREFIX=/home/li/miniconda3
MINERU_OUTPUT_DIR=./mineru_output
MINERU_METHOD=auto
MINERU_USE_CACHE=1
MINERU_WSL_DISTRO=Ubuntu
```

如果你的 WSL 有多个发行版，可以把 `MINERU_WSL_DISTRO` 设置为 `Ubuntu` 等发行版名称。
