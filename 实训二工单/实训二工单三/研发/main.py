"""
文生图智能体 - 主入口
支持命令行和交互式两种使用方式
工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务

使用方式：
D:/an1/python.exe main.py input.jpg
  D:/an1/python.exe main.py input.jpg --no-outpainting
  D:/an1/python.exe main.py input.jpg --interactive
"""

import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_agent import FaceAgent, main

if __name__ == "__main__":
    main()
