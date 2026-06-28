# 文生图智能体

**工单编号**：人工智能 NLP-Agent 数字人项目-文生图智能体任务

## 功能概述

输入一张面部图像，输出三张不同角度的面部图像（右转、左转、端正），并支持扩图（outpainting）。

```
输入: [面部图像] ──→ 文生图智能体 ──→ [右转] [端正] [左转]
                                        └── 可选扩图 ──→ [扩图后的三个角度]
```

## 技术方案

| 模块 | 技术选型 | 说明 |
|------|---------|------|
| 人脸身份保持 | IP-Adapter FaceID Plus v2 | 从输入人脸提取身份特征，生成时保持面部一致性 |
| 图像生成 | Stable Diffusion 1.5 / SDXL | 通过文本提示控制生成角度 |
| 扩图 | Stable Diffusion Inpainting | 局部重绘方式实现图像向外扩展 |
| 人脸检测 | OpenCV Haar Cascade / InsightFace | 自动检测人脸区域 |

## 环境要求

- **操作系统**: Windows 10/11
- **GPU**: NVIDIA RTX 4050 6GB（推荐）或更高
- **Python**: 3.10+（推荐使用 Anaconda）
- **CUDA**: 11.8 或 12.x

## 安装步骤

### 0. 确认 Python 环境

```bash
D:\an1\python.exe --version
```

如果 Python 版本低于 3.10，需要升级。

### 1. 安装依赖

```bash
D:\an1\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

国内用户推荐使用清华镜像加速。

### 2. 首次运行（自动下载模型）

```bash
D:\an1\python.exe main.py input/your_face_image.jpg
```

首次运行会自动下载以下模型（约 5-10GB）：
- Stable Diffusion v1.5 / Inpainting 模型
- IP-Adapter FaceID Plus v2 权重
- CLIP 图像编码器

国内用户如果下载慢，可在 `config.yaml` 中设置：
```yaml
mirror:
  type: "modelscope"   # 使用魔搭 ModelScope 镜像
```

## 使用方式

### 命令行模式

```bash
# 基本使用（生成三个角度 + 扩图）
D:\an1\python.exe main.py input.jpg

# 跳过扩图（只生成三个角度）
D:\an1\python.exe main.py input.jpg --no-outpainting

# 指定配置文件
D:\an1\python.exe main.py input.jpg --config my_config.yaml
```

### Python API 模式

```python
from face_agent import FaceAgent

# 初始化智能体
agent = FaceAgent("config.yaml")

# 执行完整流程
result = agent.run("input.jpg", do_outpainting=True)

# 查看结果路径
print("右转:", result["rotated"]["right"])
print("左转:", result["rotated"]["left"])
print("端正:", result["rotated"]["front"])
if "outpainted" in result:
    print("扩图右转:", result["outpainted"]["right"])
```

### GUI 交互模式（可选）

```bash
# 安装额外依赖
D:\an1\python.exe -m pip install gradio

# 启动 Web UI
D:\an1\python.exe app_gradio.py
```

## 输出结构

```
文生图智能体/
├── input/              ← 放入你的面部图像
├── output/
│   ├── rotated/        ← 三个角度的生成结果
│   │   ├── right_20250206_143022.png
│   │   ├── left_20250206_143022.png
│   │   └── front_20250206_143022.png
│   ├── outpainted/     ← 扩图结果
│   │   ├── right_outpainted_20250206_143022.png
│   │   ├── left_outpainted_20250206_143022.png
│   │   └── front_outpainted_20250206_143022.png
│   └── summary_*.txt   ← 结果汇总文件
└── models/             ← 下载的模型缓存
```

## 配置说明

`config.yaml` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.base` | `sd15` | 基础模型，可选 `sd15`（轻量）或 `sdxl`（高质量） |
| `hardware.enable_model_cpu_offload` | `false` | 6GB显存设置为 `true`，分层加载到显存 |
| `generation.face_rotation.ip_adapter_scale` | `0.7` | 身份保持强度，越高越像原人 |
| `generation.outpainting.expand_pixels` | `128` | 每边扩展像素数 |
| `mirror.type` | `huggingface` | 国内改为 `modelscope` 加速下载 |

## 质量调优建议

1. **面部相似度不够**：增大 `ip_adapter_scale`（0.6→0.9）
2. **角度变化不明显**：修改 `prompts` 中的文本描述，加入更明确的角度词
3. **图像质量低**：增大 `num_inference_steps`（30→50）
4. **扩图不自然**：减小 `expand_pixels`（128→64），或增加 `overlap_pixels`
5. **显存不足**：启用 `enable_model_cpu_offload: true`，或换用 `sd15` 模型

## 验收对照

| 验收项 | 标准 |
|--------|------|
| 面部特征保持 | 右转/左转/端正图像的面部特征与原图一致 |
| 角度准确性 | 右转左转有明显角度变化（±30°以内） |
| 图像清晰度 | 无像素化、模糊、失真 |
| 扩图内容一致性 | 扩图部分与原图内容协调 |
| 扩图过渡自然 | 无拼接痕迹，颜色光影一致 |
| 扩图质量提升 | 无新增噪声或干扰元素 |

## 已知问题

1. **IP-Adapter FaceID 下载慢**：国内用户请使用 ModelScope 镜像
2. **6GB 显存限制**：运行 SDXL 时需启用 CPU offload，速度较慢
3. **极端角度不自然**：超过 ±45° 的角度变化可能失真
