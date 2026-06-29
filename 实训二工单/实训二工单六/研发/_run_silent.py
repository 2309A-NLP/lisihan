#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Silent backend runner — suppresses browser popups
工单编号：人工智能NLP-Agent数字人项目-智能体任务

用途：
1. 静默启动后端服务，不弹出浏览器窗口
2. 支持 Windows/Linux/macOS 跨平台
3. 可传递参数给目标脚本

使用方法：
    python _run_silent.py <target_script.py> [args...]
    
示例：
    python _run_silent.py web_app.py
    python _run_silent.py mcp_server.py --port 8080
"""

import os
import sys
import webbrowser
import logging
import importlib.util
from typing import Optional

# ============================================================
# 1. 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SilentRunner")

# ============================================================
# 2. 抑制浏览器弹出（多策略）
# ============================================================

def _patch_webbrowser():
    """
    彻底抑制所有浏览器打开行为
    
    策略：
    1. 直接替换 webbrowser 模块的函数
    2. 注册一个空的浏览器处理器
    3. 设置环境变量禁用自动打开
    """
    # ===== 策略1: 替换模块函数 =====
    webbrowser.open = lambda *args, **kwargs: False
    webbrowser.open_new = lambda *args, **kwargs: False
    webbrowser.open_new_tab = lambda *args, **kwargs: False
    logger.debug("✓ webbrowser 函数已替换")
    
    # ===== 策略2: 注册空浏览器 =====
    try:
        # 注册一个什么都不做的浏览器
        class NullBrowser:
            def open(self, url, new=0, autoraise=True):
                return False
            
            def open_new(self, url):
                return False
            
            def open_new_tab(self, url):
                return False
        
        webbrowser.register('null', NullBrowser, instance=NullBrowser())
        webbrowser._tryorder = ['null']
        logger.debug("✓ 空浏览器已注册")
    except Exception as e:
        logger.debug(f"注册空浏览器失败: {e}")
    
    # ===== 策略3: 设置环境变量 =====
    os.environ['BROWSER'] = 'none'
    os.environ['BROWSER_DISABLED'] = '1'
    logger.debug("✓ 环境变量已设置")


def _patch_os_startfile():
    """抑制 Windows 下的 os.startfile"""
    if hasattr(os, 'startfile') and sys.platform == 'win32':
        # 保留原始函数用于其他用途
        original_startfile = os.startfile
        
        def _null_startfile(path, operation=None):
            # 只抑制 URL 和 .html 文件
            if isinstance(path, str):
                path_lower = path.lower()
                if (path_lower.startswith(('http://', 'https://', 'ftp://')) or
                    path_lower.endswith(('.html', '.htm'))):
                    logger.debug(f"抑制打开: {path}")
                    return
            # 其他文件正常打开
            if operation:
                return original_startfile(path, operation)
            return original_startfile(path)
        
        os.startfile = _null_startfile
        logger.debug("✓ os.startfile 已 patch")
    else:
        logger.debug("✓ 非 Windows 平台，跳过 os.startfile patch")


def _patch_subprocess():
    """抑制 subprocess 调用打开浏览器的命令"""
    import subprocess
    
    # 保存原始 Popen
    original_popen = subprocess.Popen
    
    class _SilentPopen(subprocess.Popen):
        """自定义 Popen，拦截浏览器相关命令"""
        
        def __init__(self, args, *pargs, **kwargs):
            # 检查是否包含浏览器相关命令
            if isinstance(args, list) and args:
                cmd = args[0] if args else ''
                if isinstance(cmd, str):
                    cmd_lower = cmd.lower()
                    # Windows: start, explorer
                    if sys.platform == 'win32':
                        if cmd_lower in ('start', 'explorer'):
                            # 检查是否打开 URL 或 HTML
                            for arg in args[1:]:
                                if isinstance(arg, str):
                                    arg_lower = arg.lower()
                                    if (arg_lower.startswith(('http://', 'https://')) or
                                        arg_lower.endswith(('.html', '.htm'))):
                                        logger.debug(f"抑制 subprocess 调用: {args}")
                                        return
                    # Linux/macOS: xdg-open, open, sensible-browser
                    elif sys.platform in ('linux', 'darwin'):
                        if cmd_lower in ('xdg-open', 'open', 'sensible-browser'):
                            for arg in args[1:]:
                                if isinstance(arg, str):
                                    arg_lower = arg.lower()
                                    if (arg_lower.startswith(('http://', 'https://')) or
                                        arg_lower.endswith(('.html', '.htm'))):
                                        logger.debug(f"抑制 subprocess 调用: {args}")
                                        return
            
            # 正常执行
            super().__init__(args, *pargs, **kwargs)
    
    # 替换 Popen
    subprocess.Popen = _SilentPopen
    logger.debug("✓ subprocess.Popen 已 patch")


# ============================================================
# 3. 执行目标脚本
# ============================================================

def execute_script(script_path: str, args: list = None) -> Optional[int]:
    """
    执行目标脚本
    
    Args:
        script_path: 目标脚本的绝对路径
        args: 传递给目标脚本的参数列表
        
    Returns:
        退出码，如果执行失败返回 None
    """
    # ===== 检查文件是否存在 =====
    if not os.path.exists(script_path):
        logger.error(f"目标脚本不存在: {script_path}")
        return None
    
    if not os.path.isfile(script_path):
        logger.error(f"路径不是文件: {script_path}")
        return None
    
    # ===== 准备执行环境 =====
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    # 设置 sys.argv
    if args is None:
        sys.argv = [script_path]
    else:
        sys.argv = [script_path] + args
    
    # 设置路径
    sys.path.insert(0, script_dir)
    os.chdir(script_dir)
    
    # ===== 读取并编译脚本 =====
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(script_path, 'r', encoding='gbk') as f:
            code = f.read()
    
    try:
        compiled = compile(code, script_path, 'exec')
    except SyntaxError as e:
        logger.error(f"脚本语法错误: {e}")
        return None
    
    # ===== 准备全局命名空间 =====
    globals_dict = globals().copy()
    globals_dict.update({
        '__name__': '__main__',
        '__file__': script_path,
        '__builtins__': __builtins__,
    })
    
    # ===== 执行脚本 =====
    try:
        logger.info(f"执行脚本: {script_path}")
        logger.info(f"参数: {sys.argv[1:] if len(sys.argv) > 1 else '(无)'}")
        exec(compiled, globals_dict)
        logger.info(f"✓ 脚本执行完成: {script_name}")
        return 0
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        return 130
    except SystemExit as e:
        # 脚本调用 sys.exit()
        exit_code = e.code if e.code is not None else 0
        if exit_code == 0:
            logger.info(f"✓ 脚本正常退出: {script_name}")
        else:
            logger.warning(f"脚本退出 (code={exit_code}): {script_name}")
        return exit_code
    except Exception as e:
        logger.error(f"脚本执行失败: {e}", exc_info=True)
        return 1


# ============================================================
# 4. 主入口
# ============================================================

def main():
    """命令行入口"""
    # ===== 解析参数 =====
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python _run_silent.py <target_script.py> [args...]")
        print("\n示例:")
        print("  python _run_silent.py web_app.py")
        print("  python _run_silent.py mcp_server.py --port 8080")
        sys.exit(1)
    
    # 获取目标脚本
    target = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # 转换为绝对路径
    target_abs = os.path.abspath(target)
    
    # ===== 应用补丁 =====
    logger.info("应用静默补丁...")
    _patch_webbrowser()
    _patch_os_startfile()
    _patch_subprocess()
    
    # 也可以 patch 其他可能的浏览器打开方式
    # 1. 抑制 os.system 调用
    original_system = os.system
    def _null_system(command):
        if isinstance(command, str):
            cmd_lower = command.lower()
            browser_cmds = ['start', 'explorer', 'xdg-open', 'open', 'sensible-browser']
            if any(cmd in cmd_lower for cmd in browser_cmds):
                logger.debug(f"抑制 os.system: {command}")
                return 0
        return original_system(command)
    os.system = _null_system
    logger.debug("✓ os.system 已 patch")
    
    # ===== 执行目标脚本 =====
    logger.info("=" * 50)
    logger.info(f"目标: {target_abs}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info("=" * 50)
    
    exit_code = execute_script(target_abs, args)
    
    # ===== 退出 =====
    if exit_code is not None:
        sys.exit(exit_code)
    else:
        sys.exit(1)


# ============================================================
# 5. 模块导入支持
# ============================================================

if __name__ == "__main__":
    main()
else:
    # 作为模块导入时，只应用补丁
    logger.info("作为模块导入，应用静默补丁...")
    _patch_webbrowser()
    _patch_os_startfile()
    _patch_subprocess()
    
    # 抑制 os.system
    import os
    original_system = os.system
    def _null_system(command):
        if isinstance(command, str):
            cmd_lower = command.lower()
            browser_cmds = ['start', 'explorer', 'xdg-open', 'open', 'sensible-browser']
            if any(cmd in cmd_lower for cmd in browser_cmds):
                logger.debug(f"抑制 os.system: {command}")
                return 0
        return original_system(command)
    os.system = _null_system