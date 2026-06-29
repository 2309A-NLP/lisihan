#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Agent Launcher — 启动所有后端服务和 Agent UI
工单编号：人工智能NLP-Agent数字人项目-智能体任务

功能：
1. 启动 5 个后端服务（记账、日程、文生图、基金、招股书）
2. 启动 Agent Web UI
3. 健康检查和服务监控

使用方法：
    python launcher.py                    # 启动所有服务并打开浏览器
    python launcher.py --no-browser      # 不打开浏览器
    python launcher.py --port 6002       # 自定义端口
"""

import os
import sys
import subprocess
import time
import webbrowser
import atexit
import socket
import signal
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ============================================================
# 1. 路径配置（使用相对路径，自动适配）
# ============================================================

# 获取当前脚本所在目录（项目根目录）
BASE = os.path.dirname(os.path.abspath(__file__))

# Python 解释器（使用当前环境）
PYTHON = sys.executable

# 日志目录
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 2. 服务配置（使用相对路径，相对于项目根目录）
# ============================================================

# 获取 agent 目录（launcher.py 的父目录）
AGENT_DIR = os.path.dirname(BASE)  # 即 C:\Users\freedom\Desktop\agent

SERVICES = [
    # 格式: (显示名称, 端口, 绝对路径, 额外等待时间)
    ("Ledger", 8081, r"C:\Users\freedom\Desktop\agent\实训二工单一\family_accounting_agent\app.py", 8),
    ("Schedule", 5000, r"C:\Users\freedom\Desktop\agent\实训二工单二\daily_scheduler_agent\web_app.py", 5),
    ("ImageGen", 7860, r"C:\Users\freedom\Desktop\agent\实训二工单三\webui.py", 60),
    ("FundQA", 5002, r"C:\Users\freedom\Desktop\agent\基金问答智能体\code\app.py", 8),
    ("Prospectus", 5003, r"C:\Users\freedom\Desktop\agent\招股书问答智能体\code\app.py", 8),
]
# 服务启动超时（秒）
STARTUP_TIMEOUT = 60
# 端口检查间隔（秒）
CHECK_INTERVAL = 1

# ============================================================
# 3. 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"launcher_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("Launcher")

# ============================================================
# 4. 进程管理
# ============================================================

class ProcessManager:
    """管理所有子进程"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.service_info: Dict[subprocess.Popen, dict] = {}
    
    def start_service(
        self,
        name: str,
        script_path: str,
        port: int,
        wait_seconds: int = 10,
        env: Optional[dict] = None
    ) -> Optional[subprocess.Popen]:
        """
        启动一个服务
        
        Args:
            name: 服务名称
            script_path: 脚本路径（绝对路径）
            port: 端口号
            wait_seconds: 启动后额外等待时间
            env: 环境变量
            
        Returns:
            进程对象，失败返回 None
        """
        # 检查脚本是否存在
        if not os.path.exists(script_path):
            logger.error(f"[{name}] 脚本不存在: {script_path}")
            return None
        
        # 准备日志文件
        log_file = os.path.join(LOG_DIR, f"{name.lower()}_{datetime.now().strftime('%Y%m%d')}.log")
        log_fd = open(log_file, "a", encoding="utf-8")
        
        # 写入启动标记
        log_fd.write(f"\n{'='*60}\n")
        log_fd.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fd.write(f"脚本: {script_path}\n")
        log_fd.write(f"端口: {port}\n")
        log_fd.write(f"{'='*60}\n\n")
        log_fd.flush()
        
        # 准备环境变量
        if env is None:
            env = os.environ.copy()
        env["BROWSER"] = "none"
        env["PYTHONUNBUFFERED"] = "1"
        
        # 准备启动命令
        cmd = [PYTHON, script_path]
        cwd = os.path.dirname(script_path) or BASE
        
        # Windows 特殊处理（隐藏窗口）
        startupinfo = None
        creationflags = 0
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NO_WINDOW
        
        try:
            logger.info(f"[{name}] 启动中 (端口 {port})...")
            
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                env=env,
                creationflags=creationflags,
            )
            
            # 保存进程信息
            self.processes.append(p)
            self.service_info[p] = {
                "name": name,
                "port": port,
                "script": script_path,
                "log_file": log_file,
                "start_time": datetime.now(),
            }
            
            # 额外等待时间（用于模型加载等）
            if wait_seconds > 0:
                logger.info(f"[{name}] 等待 {wait_seconds}s 让服务初始化...")
                time.sleep(wait_seconds)
            
            return p
            
        except Exception as e:
            logger.error(f"[{name}] 启动失败: {e}")
            log_fd.write(f"启动失败: {e}\n")
            log_fd.close()
            return None
    
    def wait_for_port(self, port: int, timeout: int = 30) -> bool:
        """
        等待端口变为可用
        
        Args:
            port: 端口号
            timeout: 超时时间（秒）
            
        Returns:
            是否成功
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                if result == 0:
                    return True
            except Exception:
                pass
            time.sleep(CHECK_INTERVAL)
        return False
    
    def check_services(self) -> Dict[str, bool]:
        """
        检查所有服务状态
        
        Returns:
            服务名称 -> 是否运行中
        """
        status = {}
        for p, info in self.service_info.items():
            name = info["name"]
            port = info["port"]
            # 检查进程是否存活
            if p.poll() is not None:
                status[name] = False
                logger.warning(f"[{name}] 进程已退出")
            else:
                # 检查端口是否可连接
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(("127.0.0.1", port))
                    sock.close()
                    status[name] = (result == 0)
                except Exception:
                    status[name] = False
        return status
    
    def stop_all(self):
        """停止所有服务"""
        logger.info("停止所有服务...")
        for p in self.processes:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
                if hasattr(p, 'stdout') and p.stdout:
                    p.stdout.close()
            except Exception as e:
                logger.debug(f"停止进程失败: {e}")
        
        self.processes.clear()
        self.service_info.clear()
        logger.info("所有服务已停止")
    
    def get_status_report(self) -> str:
        """获取状态报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("  服务状态报告")
        lines.append("=" * 60)
        
        status = self.check_services()
        for name, is_running in status.items():
            info = None
            for p, inf in self.service_info.items():
                if inf["name"] == name:
                    info = inf
                    break
            
            port = info["port"] if info else "?"
            uptime = ""
            if info and info.get("start_time"):
                elapsed = datetime.now() - info["start_time"]
                uptime = f" (运行: {str(elapsed).split('.')[0]})"
            
            status_icon = "✅" if is_running else "❌"
            lines.append(f"  {status_icon} {name:15s} 端口 {port:5d}  {uptime}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 5. 主启动器
# ============================================================

class Launcher:
    """主启动器"""
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self.start_time = datetime.now()
        self.ui_process = None
    
    def start_services(self) -> bool:
        """启动所有服务"""
        logger.info("=" * 60)
        logger.info("启动服务...")
        logger.info("=" * 60)
        
        # 启动每个服务
        for name, port, rel_path, wait in SERVICES:
            # 构建绝对路径
            script_path = os.path.join(BASE, rel_path)
            
            # 检查脚本是否存在
            if not os.path.exists(script_path):
                logger.warning(f"[{name}] 脚本不存在: {rel_path}")
                continue
            
            # 检查端口是否已被占用
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            
            if result == 0:
                logger.info(f"[{name}] 端口 {port} 已被占用，跳过启动")
                continue
            
            p = self.process_manager.start_service(
                name=name,
                script_path=script_path,
                port=port,
                wait_seconds=wait
            )
            
            if p is None:
                logger.error(f"[{name}] 启动失败")
                continue
        
        # 显示状态
        time.sleep(2)
        logger.info("\n" + self.process_manager.get_status_report())
        
        return True
    
    def start_ui(self, port: int = 6001, open_browser: bool = True):
        """启动 Web UI"""
        logger.info("=" * 60)
        logger.info("启动 Agent Web UI...")
        logger.info("=" * 60)
        
        # UI 脚本路径
        ui_script = os.path.join(BASE, "frontend", "web_app.py")
        
        if not os.path.exists(ui_script):
            logger.error(f"Web UI 脚本未找到: {ui_script}")
            return
        
        # 设置环境变量
        env = os.environ.copy()
        env["BROWSER"] = "none"
        env["PYTHONUNBUFFERED"] = "1"
        
        # Windows 特殊处理
        startupinfo = None
        creationflags = 0
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NO_WINDOW
        
        log_file = os.path.join(LOG_DIR, f"ui_{datetime.now().strftime('%Y%m%d')}.log")
        
        try:
            self.ui_process = subprocess.Popen(
                [PYTHON, ui_script],
                cwd=BASE,
                stdout=open(log_file, "a", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                env=env,
                creationflags=creationflags,
            )
            
            # 等待端口就绪
            logger.info(f"等待 UI 端口 {port} 就绪...")
            if self.process_manager.wait_for_port(port, timeout=30):
                logger.info(f"✅ UI 端口 {port} 已就绪")
                
                if open_browser:
                    url = f"http://127.0.0.1:{port}"
                    logger.info(f"打开浏览器: {url}")
                    webbrowser.open(url)
            else:
                logger.warning(f"⚠️ UI 端口 {port} 未在 30s 内就绪")
                
        except Exception as e:
            logger.error(f"UI 启动失败: {e}")
    
    def run(self, port: int = 6001, open_browser: bool = True):
        """运行启动器"""
        try:
            # 1. 启动服务
            if not self.start_services():
                logger.error("服务启动失败")
                return
            
            # 2. 启动 UI
            self.start_ui(port=port, open_browser=open_browser)
            
            # 3. 保持运行
            logger.info("=" * 60)
            logger.info(f"✅ 所有服务已启动")
            logger.info(f"   Web UI: http://127.0.0.1:{port}")
            logger.info(f"   日志目录: {LOG_DIR}")
            logger.info("   按 Ctrl+C 停止所有服务")
            logger.info("=" * 60)
            
            # 等待用户中断
            while True:
                time.sleep(5)
                
                # 检查子进程状态
                status = self.process_manager.check_services()
                all_running = all(status.values())
                if not all_running:
                    failed = [name for name, running in status.items() if not running]
                    logger.warning(f"部分服务已停止: {failed}")
                
                if self.ui_process and self.ui_process.poll() is not None:
                    logger.warning("UI 进程已退出")
                    break
                
        except KeyboardInterrupt:
            logger.info("\n收到中断信号，正在停止...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """关闭所有服务"""
        logger.info("正在停止所有服务...")
        self.process_manager.stop_all()
        
        if self.ui_process and self.ui_process.poll() is None:
            try:
                self.ui_process.terminate()
                self.ui_process.wait(timeout=3)
            except:
                try:
                    self.ui_process.kill()
                except:
                    pass
        
        logger.info("✅ 所有服务已停止")


# ============================================================
# 6. 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart Agent Launcher - 启动所有服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python launcher.py                    # 启动所有服务并打开浏览器
  python launcher.py --no-browser      # 不打开浏览器
  python launcher.py --port 6002       # 使用自定义端口
        """
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6001,
        help="Web UI 端口 (默认: 6001)"
    )
    
    args = parser.parse_args()
    
    launcher = Launcher()
    launcher.run(port=args.port, open_browser=not args.no_browser)


# ============================================================
# 7. 信号处理和入口
# ============================================================

def signal_handler(sig, frame):
    logger.info(f"收到信号 {sig}，正在退出...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    main()