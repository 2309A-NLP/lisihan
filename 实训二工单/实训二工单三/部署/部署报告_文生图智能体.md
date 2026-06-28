# 文生图智能体 — 部署报告

> 项目名称：文生图智能体（Face Rotation + Outpainting Agent）  
> 工单编号：人工智能 NLP-Agent 数字人项目-文生图智能体任务  
> 项目路径：`C:\Users\freedom\Desktop\agent\实训二工单三`  
> 生成日期：2026-06-28

---

## 一、项目概述

文生图智能体是一个**人脸角度旋转 + 图像扩图**的 AI 工具。用户上传一张面部照片，系统自动生成**右转 30°、端正、左转 30°** 三个人脸角度，并可选地执行图像向外扩展（outpainting）。

### 核心能力

| 功能 | 描述 |
|---|---|
| **人脸旋转** | 输入面部照片，生成三个角度的人脸（右转/端正/左转），保留身份特征 |
| **图像扩图** | 对生成结果向外扩展（四边各扩展 128px 以上），使画面更完整 |
| **双模式运行** | 纯 API 云端调用（主方案，无需 GPU）或本地 Stable Diffusion（备用） |

### 技术架构

```
用户输入 → face_agent.py (协调器)
            ├── api_rotation.py   (主方案：火山引擎 Doubao API)
            ├── face_rotation.py  (备用1：本地 SD + IP-Adapter)
            ├── face_3d_rotation.py (备用2：3D 关键点变形)
            └── outpainting.py    (本地 SD Inpainting 扩图)
         → output/ 目录输出
```

---

## 二、环境要求

### 2.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|---|---|---|
| **CPU** | 4 核 x86-64 | 6 核以上 |
| **内存** | 8 GB | 16 GB 以上 |
| **GPU** | 不需要（API 模式） | NVIDIA RTX 4050 6GB（本地模式） |
| **磁盘** | 10 GB 可用 | 20 GB 以上（含模型缓存） |
| **操作系统** | Windows 10/11 | Windows 11 |

> 本项目当前部署在 Windows 11 上，**默认使用 CPU + API 云端调用**，不需要 GPU。  
> 如需本地模型推理（face_rotation.py / outpainting.py），建议使用 NVIDIA GPU + CUDA。

### 2.2 软件环境

| 软件 | 版本 | 说明 |
|---|---|---|
| **Python** | 3.10 | 推荐通过 Conda 管理环境 |
| **Conda** | 任意 | 环境名称 `py310` |
| **CUDA** | 11.8 或 12.x | 仅 GPU 模式下需要 |
| **pip 镜像** | 清华源 | 国内下载依赖加速 |

### 2.3 Python 依赖

文件：`requirements.txt`

```
diffusers[torch]>=0.27.0     # 扩散模型推理
transformers>=4.36.0          # HuggingFace 模型
accelerate>=0.25.0            # 混合精度/设备管理
safetensors>=0.4.0            # 安全张量格式
pillow>=10.0.0                # 图像处理
numpy<2.0                     # 数值计算
pyyaml                        # 配置文件解析
tqdm                          # 进度条
```

**附加依赖**（按需安装）：

| 用途 | 包名 | 说明 |
|---|---|---|
| Gradio WebUI | `gradio` | 现代化 Web 界面 |
| 人脸检测 | `opencv-python` | 用于 `face_utils.py` 中的人脸检测 |
| 3D 旋转 | `insightface` | 用于 `face_3d_rotation.py` 人脸关键点检测 |
| 环境检查 | `peft`, `open_clip` | 仅在 `check_env.py` 中使用 |

---

## 三、安装步骤

### 3.1 创建 Conda 环境

```batch
:: 打开 Anaconda Prompt (或 cmd)
conda create -n py310 python=3.10 -y
conda activate py310
```

### 3.2 安装依赖

```batch
:: 设置清华镜像源（国内推荐）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

:: 安装核心依赖
pip install -r requirements.txt

:: 安装附加依赖（根据使用场景选择）
pip install gradio               # 如果需要 Gradio WebUI
pip install opencv-python        # 如果需要人脸检测/3D旋转
pip install insightface          # 如果需要 3D 关键点旋转
```

### 3.3 验证环境

```batch
python check_env.py
```

> 该脚本会检查 `diffusers`、`torch`、`transformers`、`accelerate`、`insightface`、`peft`、`open_clip` 的版本，并验证 `StableDiffusionPipeline` 是否可用。

### 3.4 模型下载说明

**API 模式**（默认）：不需要下载任何模型，直接调用云端 API。

**本地模式**（备用）：

- `face_rotation.py` 会从 HuggingFace 自动下载 `runwayml/stable-diffusion-v1-5` 和 `h94/IP-Adapter` 模型（约 3-5 GB）
- `outpainting.py` 会下载 `runwayml/stable-diffusion-inpainting` 模型（约 2 GB）
- 国内用户可在 `config.yaml` 中设置 `mirror.type: "modelscope"` 使用 ModelScope 镜像加速

---

## 四、配置详解

配置文件：`config.yaml`

### 4.1 模型配置 (`model`)

```yaml
model:
  base: "sd15"              # 可选 "sd15" 或 "sdxl"
  sd15:
    base_model: "runwayml/stable-diffusion-v1-5"
    ip_adapter: "h94/IP-Adapter"
    ip_adapter_subfolder: "models"
    ip_adapter_weight_name: "ip-adapter-plus-face_sd15.bin"
    controlnet: "lllyasviel/control_v11p_sd15_openpose"
  sdxl:
    base_model: "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter: "h94/IP-Adapter-FaceID"
    ip_adapter_subfolder: ""
    ip_adapter_weight_name: "ip-adapter-plus-face_sdxl_vit-h.bin"
```

> `base` 字段选择基础模型。SDXL 需要 8GB+ VRAM，CPU 模式下推荐保持 `sd15`。

### 4.2 生成参数 (`generation`)

```yaml
generation:
  face_rotation:
    num_inference_steps: 30      # 推理步数，越大质量越高但越慢
    guidance_scale: 5.0          # 提示引导强度
    ip_adapter_scale: 0.7        # IP-Adapter 权重（身份保持强度）
    negative_prompt: "ugly, blurly, low quality..."
    seed: 42                     # 随机种子
  outpainting:
    num_inference_steps: 40      # 扩图推理步数
    guidance_scale: 7.5          # 扩图提示引导强度
    expand_pixels: 128           # 每边扩展像素数
    overlap_pixels: 32           # 与原始图像重叠区域
```

### 4.3 硬件配置 (`hardware`)

```yaml
hardware:
  device: "cpu"                  # "cpu" 或 "cuda"
  dtype: "fp32"                  # CPU 使用 fp32，GPU 可使用 fp16
  enable_attention_slicing: true # CPU 内存优化
  enable_model_cpu_offload: false
  enable_vae_slicing: true
  enable_vae_tiling: true
```

> CPU 模式下 dtype 强制为 fp32，fp16 在 CPU 上无优势。

### 4.4 API 配置 (`api`)

```yaml
api:
  api_key: "sk-joz..."           # 已迁移到环境变量 ARK_API_KEY
  model: "doubao-seedream-4-5-251128"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  image_size: 1024
```

> **重要**：API Key 建议通过环境变量 `ARK_API_KEY` 设置，避免明文写入配置文件。  
> Endpoint ID 通过环境变量 `ARK_ENDPOINT_ID` 设置。  
> 默认连接到火山引擎 Doubao-Seedream-4.5 模型。

### 4.5 镜像设置 (`mirror`)

```yaml
mirror:
  type: "huggingface"   # 国内可改为 "modelscope"
```

---

## 五、启动方式

本项目提供 **4 种启动方式**，适应不同使用场景。

### 5.1 命令行模式（交互式）

**方式 A：双击批处理脚本**

双击 `运行命令行.bat`，按提示拖入人脸图片路径：

```batch
@echo off
chcp 65001 >nul
:: 提示输入图片路径
set /p INPUT=
:: 提示是否启用扩图
set /p OUTPAINT=
C:\Users\freedom\.conda\envs\py310\python.exe main.py "%INPUT%"
pause
```

**方式 B：直接命令行执行**

```batch
:: 激活 Conda 环境
conda activate py310

:: 单次运行（带扩图）
python main.py input.jpg

:: 单次运行（跳过扩图）
python main.py input.jpg --no-outpainting

:: 指定配置文件
python main.py input.jpg --config config.yaml
```

### 5.2 WebUI — 纯 Python 标准库版

使用 Python 标准库 `http.server`，无需额外安装 Gradio。

```batch
conda activate py310
python webui.py
```

- 端口：**8642**
- 访问：浏览器打开 `http://127.0.0.1:8642`
- 特点：零额外依赖，界面简洁，支持图片上传预览、生成结果展示、历史记录

### 5.3 WebUI — Gradio 版

现代化 Web 界面，支持进度条和实时预览。

```batch
:: 确保已安装 gradio
pip install gradio

:: 启动
python app_gradio.py
```

- 端口：**7860**
- 访问：浏览器打开 `http://127.0.0.1:7860`
- 特点：界面美观，支持拖动上传、三角度并列展示、扩图结果拼接显示

### 5.4 Windows 启动脚本

双击 `启动WebUI.bat` 即可启动 WebUI（标准库版）：

```batch
C:\Users\freedom\.conda\envs\py310\python.exe webui.py
```

---

## 六、输出说明

### 6.1 输出目录结构

```
output/
├── input_20260628_120000.png          # 原图副本
├── rotated/
│   ├── right_20260628_120000.png      # 右转 30°
│   ├── front_20260628_120000.png      # 端正
│   └── left_20260628_120000.png       # 左转 30°
├── outpainted/
│   ├── right_outpainted_20260628_120000.png
│   ├── front_outpainted_20260628_120000.png
│   └── left_outpainted_20260628_120000.png
└── summary_20260628_120000.txt        # 结果汇总
```

### 6.2 结果汇总文件

`summary_*.txt` 包含：
- 输入图像路径
- 处理时间
- 各角度输出路径
- 扩图结果路径

---

## 七、工作流详解

### 7.1 主要工作流（API 模式）

```
[步骤 1/4] 加载图像
    → 支持 JPG/PNG，自动调整尺寸为 8 的倍数

[步骤 2/4] 图像准备
    → API 模式下保留全图上下文（不裁剪）

[步骤 3/4] 生成三个角度
    → 调用火山引擎 Doubao API
    → 使用中英文混排 prompt 控制角度
    → 右转: "人物头部向左转，右侧脸颊变小"
    → 左转: "人物头部向右转，左侧脸颊变小"
    → 端正: "人物正对镜头，面部左右对称"

[步骤 4/4] 执行扩图
    → 优先调用 API 扩图
    → API 失败时回退到本地 SD Inpainting
    → 向四个方向各扩展 128px+
```

### 7.2 备用方案

| 方案 | 文件 | 适用场景 |
|---|---|---|
| **API 旋转（主）** | `api_rotation.py` | 有网络，无 GPU，最快 |
| **本地 SD 旋转（备用1）** | `face_rotation.py` | 无网络，有 GPU/CPU |
| **3D 关键点旋转（备用2）** | `face_3d_rotation.py` | 轻量级，纯 CPU，无需下载模型 |
| **本地 SD 扩图** | `outpainting.py` | API 扩图失败时的本地回退 |

### 7.3 人脸旋转角度定义

| 角度 | 模型 | yaw | 说明 |
|---|---|---|---|
| `right` | 文档中右转 | +30° | 头部向左转，看到右侧脸颊更少、左耳更多 |
| `front` | 正面 | 0° | 正对镜头，面部左右对称 |
| `left` | 文档中左转 | -30° | 头部向右转，看到左侧脸颊更少、右耳更多 |

---

## 八、常见问题

### Q1: 启动时提示 "No module named 'xxx'"

**原因**：缺少依赖包。

**解决**：
```batch
conda activate py310
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: API 调用失败，提示 "未设置 API Key"

**原因**：火山引擎 API Key 未配置。

**解决**：
```batch
:: 临时设置（当前会话有效）
set ARK_API_KEY=你的API_KEY
set ARK_ENDPOINT_ID=你的ENDPOINT_ID

:: 或者永久添加到系统环境变量
:: 我的电脑 → 属性 → 高级系统设置 → 环境变量 → 新建
```

### Q3: CPU 模式推理非常慢

**原因**：本地 SD 模型在 CPU 上推理，每张图可能需要数分钟。

**解决**：
1. 使用 API 模式（默认），不加载本地模型
2. 启用 GPU：在 `config.yaml` 中设置 `hardware.device: "cuda"`
3. 降低 `num_inference_steps`（如从 30 降到 15）
4. 启用注意力切片：`enable_attention_slicing: true`

### Q4: 模型下载失败 / 网络超时

**原因**：HuggingFace 在国内访问不稳定。

**解决**：
1. 修改 `config.yaml` 中 `mirror.type: "modelscope"`
2. 手动下载模型后放到 `C:\Users\<用户名>\.cache\huggingface\hub\` 目录
3. 使用代理或 VPN

### Q5: Gradio WebUI 启动报错

**原因**：Gradio 版本不兼容或未安装。

**解决**：
```batch
pip install --upgrade gradio
python upgrade_gradio.py
```

### Q6: 图像生成质量不佳 / 人脸不像

**调整参数**：
- 增加 `ip_adapter_scale`（0.7 → 0.85）：加强身份保持
- 降低 `guidance_scale`（5.0 → 3.5）：减少过度加工
- 增加 `num_inference_steps`（30 → 50）：提高细节质量
- 调整 prompt 中的角度描述措辞

### Q7: 人脸旋转方向不对（左/右反了）

**说明**：
- 代码中的"右转"是**观察者视角**：头部**向左**转动，观察者看到更多左耳
- 如果方向反了，可以通过增加 `mirror: true` 配置来镜像翻转
- 参见 `api_rotation.py` 中 `ANGLES` 字典的 `mirror` 字段

### Q8: WebUI 端口被占用

**解决**：
```batch
:: 标准库版：指定其他端口
python webui.py 8888

:: Gradio 版：修改 app_gradio.py 中 server_port 参数
```

### Q9: 使用 GPU 后报 CUDA 错误

**检查**：
1. 确认 CUDA 版本匹配：运行 `nvidia-smi` 查看驱动版本
2. 确认 PyTorch CUDA 版本：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应返回 True
   ```
3. 如不匹配，重新安装对应版本的 PyTorch：
   ```batch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Q10: 扩图结果有拼接痕迹 / 颜色突变

**解决**：
- 增加 `overlap_pixels`（32 → 64）：扩大重叠区域
- 降低 `guidance_scale`（7.5 → 5.0）：减少创造性填充
- 确保输入图像背景简洁，避免复杂纹理

---

## 九、项目文件清单

```
实训二工单三/
├── main.py                  # 入口文件（命令行 + 交互式）
├── face_agent.py            # 核心智能体（工作流协调）
├── config.yaml              # 全局配置文件
├── requirements.txt         # Python 依赖清单
├── webui.py                 # 纯 Python 标准库 WebUI（端口 8642）
├── app_gradio.py            # Gradio 现代化 WebUI（端口 7860）
├── check_env.py             # 环境检查脚本
├── upgrade_gradio.py        # Gradio 升级脚本
├── 启动WebUI.bat            # Windows 一键启动 WebUI 脚本
├── 运行命令行.bat           # Windows 命令行模式启动脚本
├── pipelines/
│   ├── __init__.py
│   ├── api_rotation.py      # 【主方案】火山引擎 Doubao API 旋转
│   ├── face_rotation.py     # 【备用1】本地 SD + IP-Adapter 旋转
│   ├── face_3d_rotation.py  # 【备用2】3D 关键点变形旋转
│   └── outpainting.py       # 本地 SD Inpainting 扩图
├── utils/
│   └── face_utils.py        # 图像加载/裁剪/尺寸对齐工具函数
└── output/                  # 输出目录（自动创建）
    ├── rotated/             # 旋转结果
    ├── outpainted/          # 扩图结果
    └── summary_*.txt        # 处理汇总
```

---

## 十、附：快速启动卡片

```
┌──────────────────────────────────────────┐
│          文生图智能体 — 快速启动           │
├──────────────────────────────────────────┤
│                                          │
│  1. 激活环境                              │
│     $ conda activate py310               │
│                                          │
│  2. 安装依赖                              │
│     $ pip install -r requirements.txt    │
│                                          │
│  3. 配置 API Key                          │
│     set ARK_API_KEY=your_key             │
│     set ARK_ENDPOINT_ID=your_endpoint    │
│                                          │
│  4. 启动 (选一)                           │
│     ─ 命令行: 双击 运行命令行.bat          │
│     ─ WebUI:   python webui.py           │
│     ─ Gradio:  python app_gradio.py      │
│                                          │
│  5. 访问                                  │
│     ─ WebUI:   http://127.0.0.1:8642     │
│     ─ Gradio:  http://127.0.0.1:7860     │
│                                          │
└──────────────────────────────────────────┘
```

---

## 文档版本

| 版本 | 日期 | 修改内容 |
|---|---|---|
| v1.0 | 2026-06-28 | 初始部署报告，覆盖全部功能和配置 |
