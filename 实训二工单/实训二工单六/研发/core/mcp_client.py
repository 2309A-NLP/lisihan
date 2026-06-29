#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轻量级 MCP stdio 客户端 —— 无第三方库依赖
通过子进程连接 mcp_server.py，通过 stdio 使用 JSON-RPC 进行通信
兼容 FastMCP 的 Content-Length 帧协议
"""

import json
import subprocess
import time
import logging
import sys
import os
import threading
import queue
from typing import Optional, Dict, List

logger = logging.getLogger("MCP.Client")


class McpClient:
    """
    MCP stdio 客户端，兼容 FastMCP 的 Content-Length 帧协议
    """

    def __init__(self, python_exe: str, server_script: str):
        self.python_exe = python_exe
        self.server_script = server_script
        self.proc: Optional[subprocess.Popen] = None
        self._tools: List[Dict] = []
        self._connected = False
        self._msg_id = 0
        self._lock = threading.Lock()
        self._response_queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()

    # ========== 连接管理 ==========

    def connect(self, timeout: float = 10) -> bool:
        """启动 MCP 服务器并初始化（使用 Content-Length 帧协议）"""
        try:
            logger.info(f"启动 MCP 服务器: {self.python_exe} {self.server_script}")

            # Windows 隐藏窗口
            startupinfo = None
            creationflags = 0
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW

            self.proc = subprocess.Popen(
                [self.python_exe, self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                creationflags=creationflags,
                bufsize=0,
            )
            logger.info(f"子进程已启动，PID: {self.proc.pid}")

            # 启动读取线程
            self._stop_reader.clear()
            self._response_queue = queue.Queue()
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name="MCP-Reader"
            )
            self._reader_thread.start()

            # 发送 initialize 请求（使用帧协议）
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "agent", "version": "1.0"}
                }
            }
            response = self._send_request(init_request, timeout)

            if not response or "error" in response:
                error_msg = response.get("error", {}).get("message", "未知错误") if response else "无响应"
                logger.error(f"初始化失败: {error_msg}")
                self._cleanup()
                return False

            logger.info("MCP 初始化完成")

            # 发送 initialized 通知
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            self._send_notification(notification)

            # 获取工具列表
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            tools_response = self._send_request(tools_request, timeout)

            if not tools_response or "error" in tools_response:
                logger.error("获取工具列表失败")
                self._cleanup()
                return False

            result_data = tools_response.get("result", {})
            if isinstance(result_data, dict):
                self._tools = result_data.get("tools", [])
            elif isinstance(result_data, list):
                self._tools = result_data
            else:
                self._tools = []

            logger.info(f"发现 {len(self._tools)} 个工具")
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"连接失败: {e}")
            self._cleanup()
            return False

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        self._cleanup()
        logger.info("已断开 MCP 连接")

    # ========== 工具调用 ==========

    def call_tool(self, name: str, args: dict, timeout: float = 30) -> str:
        """调用 MCP 工具"""
        if not self._connected or not self.proc:
            return "错误: MCP 客户端未连接"

        try:
            # 自动适配参数名：如果工具需要 question，把 query 转成 question
            mcp_args = args.copy()
            if name in ["fund_query", "prospectus_query"]:
                if "query" in mcp_args and "question" not in mcp_args:
                    mcp_args["question"] = mcp_args.pop("query")

            request = {
                "jsonrpc": "2.0",
                "id": 0,  # id 会在 _send_request 中自动生成
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": mcp_args
                }
            }
            response = self._send_request(request, timeout)

            if not response:
                return "错误: 工具调用超时或无响应"

            if "error" in response:
                return f"错误: {response['error'].get('message', '未知错误')}"

            if "result" in response:
                content = response["result"].get("content", [])
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                if texts:
                    return "\n".join(texts)
                return str(response["result"])

            return "错误: 未知响应格式"

        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            return f"错误: {e}"

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def is_connected(self) -> bool:
        return self._connected and self.proc is not None and self.proc.poll() is None

    # ========== 核心通信方法 ==========

    def _reader_loop(self):
        """后台读取线程：解析 FastMCP 输出的 JSON 行"""
        while not self._stop_reader.is_set():
            try:
                if not self.proc or not self.proc.stdout:
                    break

                if self.proc.poll() is not None:
                    break

                # 读取一行（FastMCP v1.28+ 输出纯 JSON 行，每行一个响应）
                try:
                    line = self.proc.stdout.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    line_str = line.decode("utf-8").strip()
                except Exception:
                    time.sleep(0.05)
                    continue

                if not line_str:
                    continue

                # FastMCP 输出格式：{"jsonrpc":"2.0","id":1,"result":{...}}
                if line_str.startswith("{"):
                    try:
                        response = json.loads(line_str)
                        if "id" in response:
                            self._response_queue.put(response)
                            logger.debug(f"收到响应 ID: {response.get('id')}")
                        elif "method" in response:
                            logger.debug(f"收到通知: {response.get('method')}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}")

            except Exception as e:
                logger.debug(f"读取线程异常: {e}")
                time.sleep(0.05)

        logger.debug("读取线程结束")

    def _send_request(self, request: dict, timeout: float = 10) -> Optional[Dict]:
        """发送请求并等待响应（使用 Content-Length 帧协议）"""
        if not self.proc or not self.proc.stdin:
            logger.error("子进程未启动或 stdin 不可用")
            return None

        # 生成唯一 ID
        with self._lock:
            self._msg_id += 1
            request["id"] = self._msg_id
            msg_id = self._msg_id

        request_json = json.dumps(request)

        # ===== FastMCP v1.28+ 使用纯 JSON 行协议 =====
        # FastMCP 的 stdio_server 逐行读取 stdin (async for line in stdin:)
        # 不支持 Content-Length 帧头，不加 \n 会导致 readline() 永久阻塞
        line = request_json + "\n"

        try:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
            logger.debug(f"发送请求 [{msg_id}]: {request.get('method')}")

            # 等待响应
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break

                    response = self._response_queue.get(timeout=min(remaining, 0.2))
                    if response.get("id") == msg_id:
                        logger.debug(f"收到响应 [{msg_id}]")
                        return response

                except queue.Empty:
                    if self.proc and self.proc.poll() is not None:
                        logger.error("子进程已退出")
                        return None
                    continue

            logger.warning(f"请求超时 [{msg_id}]: {request.get('method')}")
            return None

        except BrokenPipeError:
            logger.error("管道已断开，子进程可能已退出")
            return None
        except Exception as e:
            logger.error(f"发送请求失败: {e}")
            return None

    def _send_notification(self, notification: dict):
        """发送通知（无 ID，不等待响应），使用 Content-Length 帧协议"""
        if not self.proc or not self.proc.stdin:
            return

        notification_json = json.dumps(notification)
        # FastMCP v1.28+ 使用纯 JSON 行协议，不加 \n 会导致阻塞
        line = notification_json + "\n"

        try:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
            logger.debug(f"发送通知: {notification.get('method')}")
        except Exception as e:
            logger.warning(f"发送通知失败: {e}")

    # ========== 资源清理 ==========

    def _cleanup(self):
        """清理子进程"""
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)

        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3)
                    except:
                        self.proc.kill()
            except:
                pass
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except:
                pass
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
            except:
                pass
            try:
                if self.proc.stderr:
                    self.proc.stderr.close()
            except:
                pass
            self.proc = None

        self._connected = False
        self._tools = []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ===================== 测试 =====================

def main():
    logging.basicConfig(level=logging.INFO)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "..", "mcp_server.py")

    if not os.path.exists(server_script):
        print(f"错误: 找不到 mcp_server.py: {server_script}")
        return

    client = McpClient(sys.executable, server_script)

    print("=" * 50)
    print("MCP 客户端测试 (Content-Length 帧协议)")
    print("=" * 50)

    if client.connect(timeout=30):
        print(f"✅ 连接成功，发现 {len(client.tools)} 个工具")
        for tool in client.tools:
            print(f"  - {tool.get('name')}: {tool.get('description', '')[:50]}")

        # 测试工具调用
        if client.tools:
            test_tool = client.tools[0]["name"]
            print(f"\n测试调用工具: {test_tool}")
            result = client.call_tool(test_tool, {"query": "测试查询"})
            print(f"结果: {result}")

        client.disconnect()
        print("\n✅ 已断开连接")
    else:
        print("❌ 连接失败")

    print("=" * 50)


if __name__ == "__main__":
    main()