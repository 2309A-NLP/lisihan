#!/usr/bin/env bash
# @工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
# @作者: [AI生成]
# @功能: Linux/macOS 一键启动 Flask Web 界面

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SCRIPT_DIR"
"$PYTHON_BIN" web_app.py
