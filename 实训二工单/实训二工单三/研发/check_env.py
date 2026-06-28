"""检查环境中的 diffusers / IP-Adapter 版本"""
import importlib
import sys

for mod_name in ['diffusers', 'torch', 'transformers', 'accelerate', 'insightface', 'peft', 'open_clip']:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"{mod_name}: {ver}")
    except ImportError:
        print(f"{mod_name}: NOT INSTALLED")
    except Exception as e:
        print(f"{mod_name}: ERROR - {e}")

# 检查 IP-Adapter 相关文件
from pathlib import Path
import os

# 检查 diffusers IP-Adapter 相关模块
try:
    from diffusers import StableDiffusionPipeline
    print(f"\nStableDiffusionPipeline: OK")
    # 检查是否有 load_ip_adapter
    pipe = StableDiffusionPipeline
    if hasattr(pipe, 'load_ip_adapter'):
        print("load_ip_adapter: available as class method")
    else:
        print("load_ip_adapter: NOT available as class method")
except Exception as e:
    print(f"StableDiffusionPipeline: ERROR - {e}")
