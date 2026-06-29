# -*- coding: utf-8 -*-
import sys
import os
import time
sys.path.insert(0, r"C:\Users\freedom\Desktop\agent\06-Agent智能体项目")

from core.mcp_client import McpClient
import logging
logging.basicConfig(level=logging.DEBUG)

print("=" * 60)
print("开始测试 MCP 客户端连接")
print("=" * 60)

client = McpClient(
    sys.executable,
    r"C:\Users\freedom\Desktop\agent\06-Agent智能体项目\mcp_server.py"
)

print("正在连接...")
ok = client.connect(timeout=30)
print(f"连接结果: {ok}")
print(f"工具数量: {len(client.tools)}")

if client.tools:
    for t in client.tools:
        print(f"  - {t.get('name')}: {t.get('description', '')[:50]}")
else:
    print("⚠️ 没有发现任何工具")

client.disconnect()
print("=" * 60)