# LiveTalking 实时数字人引擎 — 部署文档

> 项目路径: `C:\Users\freedom\Desktop\agent\实训二工单八\研发\LiveTalking`

---

## 目录

1. [环境准备](#1-环境准备)
2. [获取项目](#2-获取项目)
3. [环境配置](#3-环境配置)
4. [模型下载](#4-模型下载)
5. [配置文件](#5-配置文件)
6. [启动服务](#6-启动服务)
7. [客户端接入](#7-客户端接入)
8. [防火墙与端口](#8-防火墙与端口)
9. [验证](#9-验证)
10. [常见问题](#10-常见问题)

---

## 1. 环境准备

### 1.1 系统要求

| 项目       | 推荐配置                      |
| ---------- | ----------------------------- |
| 操作系统   | Ubuntu 22.04 / Windows + WSL2 |
| Python     | 3.10+（推荐 3.12）            |
| 包管理器   | Conda（Miniconda / Anaconda） |
| Git        | 最新稳定版                    |

### 1.2 硬件要求

| 组件          | 最低要求            | 推荐配置                        |
| ------------- | ------------------- | ------------------------------- |
| GPU           | NVIDIA GPU, CUDA 11.6+ | NVIDIA RTX 3060 / 4090 或更高 |
| 显存          | 4 GB                | 8 GB+                           |
| 内存          | 8 GB                | 16 GB+                          |
| 磁盘空间      | 10 GB               | 20 GB+（含模型文件）            |

### 1.3 软件依赖预检查

在开始前，确认以下工具已安装：

```bash
# 检查 Python 版本
python --version   # 需 >= 3.10

# 检查 CUDA 版本（Windows 下使用 nvidia-smi）
nvidia-smi        # 确认 CUDA Version >= 11.6

# 检查 Conda
conda --version

# 检查 Git
git --version
```

> **Windows WSL2 提示**：如使用 WSL2，请确保 WSL 内已安装 NVIDIA 驱动和 CUDA Toolkit。WSL2 通过 GPU-PV 共享宿主机 GPU 驱动。

---

## 2. 获取项目

### 2.1 从 GitHub 克隆（推荐）

```bash
# 克隆仓库
git clone https://github.com/lipku/LiveTalking.git
cd LiveTalking
```

### 2.2 使用已有项目代码

如已提供项目代码包，直接将 `LiveTalking` 文件夹复制到工作目录：

```bash
# 示例：将代码复制到用户桌面
cp -r /path/to/LiveTalking ~/Desktop/LiveTalking
cd ~/Desktop/LiveTalking
```

> **注意**：本项目代码位于 `C:\Users\freedom\Desktop\agent\实训二工单八\研发\LiveTalking`，可直接在此目录下操作。

---

## 3. 环境配置

### 3.1 创建 Conda 虚拟环境

```bash
# 创建环境（指定 Python 3.12）
conda create -n livetalking python=3.12 -y

# 激活环境
conda activate livetalking
```

### 3.2 安装 PyTorch（GPU 版本）

根据 CUDA 版本选择对应的 PyTorch 安装命令。

**CUDA 12.1+ 用户（推荐）：**

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

**其他 CUDA 版本：**

请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应安装命令。

验证 PyTorch GPU 可用性：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

应输出 `True` 及 CUDA 版本号。

### 3.3 安装项目依赖

```bash
# 在项目根目录下执行
cd LiveTalking

# 安装 requirements.txt 中的依赖
pip install -r requirements.txt

# 安装额外必要工具
pip install pyyaml python-dotenv
```

> **提示**：如果 `requirements.txt` 中缺少某些较新依赖（如 `edge-tts`、`websockets` 等），可手动安装：
>
> ```bash
> pip install edge-tts websockets fastapi uvicorn aiortc aiohttp
> ```

---

## 4. 模型下载

### 4.1 下载 wav2lip 模型

| 模型文件         | 目标路径         | 说明                        |
| ---------------- | ---------------- | --------------------------- |
| `wav2lip.pth`    | `models/`        | Wav2Lip 核心权重文件        |

将下载好的 `wav2lip.pth` 放入项目根目录下的 `models/` 文件夹：

```bash
# 创建 models 目录（如不存在）
mkdir -p models

# 将模型文件放入（示例，实际路径根据下载位置调整）
cp /path/to/downloaded/wav2lip.pth models/
```

### 4.2 下载数字人形象

| 模型包                    | 目标路径                     | 说明                      |
| ------------------------- | ---------------------------- | ------------------------- |
| `wav2lip256_avatar1`      | `data/avatars/wav2lip256_avatar1/` | 默认数字人形象包（含视频、音频参考） |

```bash
# 创建 avatars 目录
mkdir -p data/avatars

# 解压形象包到目标路径
# 假设压缩包为 wav2lip256_avatar1.zip
unzip wav2lip256_avatar1.zip -d data/avatars/

# 解压后目录结构示例：
# data/avatars/wav2lip256_avatar1/
# ├── avatar_video.mp4
# ├── avatar_audio.wav
# └── ...
```

### 4.3 模型下载来源

模型文件可从以下渠道获取：

1. **项目 README 中的网盘链接**（百度网盘 / Google Drive）
2. **GitHub Releases**
3. **Hugging Face 模型库**（搜索 `lipku/LiveTalking`）

> **提醒**：确保模型文件版本与代码兼容。如使用 musetalk 或 ultralight 模型，需下载对应权重。

---

## 5. 配置文件

### 5.1 复制配置文件模板

```bash
# 复制示例配置文件为实际配置文件
cp config.yaml.example config.yaml
```

### 5.2 核心配置参数详解

编辑 `config.yaml`，主要配置项如下：

```yaml
# ============== TTS（文本转语音）配置 ==============
tts: edgetts                           # TTS 引擎：edgetts / gpt-sovits / cosyvoice / fishtts / tencent / doubao / indextts2 / azuretts / qwentts / omnitts
REF_FILE: zh-CN-YunxiaNeural           # Edge TTS 语音角色（仅 edgetts 有效）
# 其他 TTS 引擎需额外配置 API Key、端点等

# ============== ASR（语音识别）配置 ==============
asr_server: local                      # ASR 模式：local（本地 SenseVoice）/ remote（远程 ASR）
# 如使用 local 模式，需确保 SenseVoice 模型已下载
# 如使用 remote 模式，需配置远程 ASR 地址

# ============== LLM（大语言模型）配置 ==============
llm_api_url: "https://api.siliconflow.cn/v1/chat/completions"   # LLM API 地址
llm_api_key: "${LLM_API_KEY}"          # API Key（建议通过 .env 文件管理，避免明文）
llm_model: "Qwen/Qwen2.5-7B-Instruct" # 使用的模型名称

# ============== 传输协议配置 ==============
transport: webrtc                      # 传输方式：webrtc / rtcpush / rtmp / virtualcam
listenport: 8010                       # Web 服务监听端口
max_session: 5                         # 最大并发会话数

# ============== 视频 / 数字人配置 ==============
model: wav2lip                         # 数字人模型：wav2lip / musetalk / ultralight
avatar_id: wav2lip256_avatar1          # 数字人形象 ID（对应 data/avatars/ 下的子目录）
```

### 5.3 环境变量配置（.env）

创建 `.env` 文件，存放敏感信息：

```bash
# 创建 .env 文件
touch .env
```

编辑 `.env` 内容：

```env
# LLM API 密钥
LLM_API_KEY=your_siliconflow_api_key_here

# 其他可选配置
# TENCENT_SECRET_ID=
# TENCENT_SECRET_KEY=
# DOUBAO_API_KEY=
# AZURE_TTS_KEY=
# AZURE_TTS_REGION=
```

> **安全提示**：`.env` 文件已加入 `.gitignore`，不会被提交到版本控制中。

---

## 6. 启动服务

### 6.1 使用命令行参数启动

```bash
# 激活环境
conda activate livetalking

# 基础启动命令
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

**常用命令行参数：**

| 参数               | 说明                              | 默认值               |
| ------------------ | --------------------------------- | -------------------- |
| `--transport`      | 传输方式                          | `webrtc`             |
| `--model`          | 数字人模型                        | `wav2lip`            |
| `--avatar_id`      | 数字人形象 ID                     | `wav2lip256_avatar1` |
| `--port`           | 监听端口                          | `8010`               |
| `--max_session`    | 最大会话数                        | `5`                  |
| `--tts`            | TTS 引擎                          | `edgetts`            |
| `--asr_server`     | ASR 模式                          | `local`              |
| `--llm`            | 是否启用 LLM                      | `True`               |
| `--ref_file`       | TTS 语音角色                      | `zh-CN-YunxiaNeural` |

### 6.2 使用配置文件启动

```bash
# 加载 config.yaml 中的配置启动
python app.py -c config.yaml
```

### 6.3 后台运行（生产环境）

```bash
# 使用 nohup 后台运行，日志输出到文件
nohup python app.py -c config.yaml > livetalking.log 2>&1 &

# 查看启动日志
tail -f livetalking.log
```

### 6.4 启动成功标志

日志中出现以下信息表示启动成功：

```
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8010 (Press CTRL+C to quit)
```

---

## 7. 客户端接入

### 7.1 浏览器 Web UI 访问

启动服务后，在浏览器中访问：

```
http://<服务器IP>:8010/index.html
```

- **本地测试**：`http://127.0.0.1:8010/index.html`
- **局域网访问**：`http://192.168.x.x:8010/index.html`
- **公网访问**：需配置公网 IP 或域名解析

Web UI 提供以下功能：
- 文本输入对话
- 语音输入对话
- 实时数字人视频流
- 会话管理

### 7.2 API 调用

LiveTalking 提供以下 REST API 接口：

#### 7.2.1 文本驱动（/human）

```bash
curl -X POST http://localhost:8010/human \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，今天天气怎么样？",
    "session_id": "session_001"
  }'
```

**参数说明：**

| 参数         | 类型   | 必填 | 说明                 |
| ------------ | ------ | ---- | -------------------- |
| `text`       | string | 是   | 要合成的文本内容     |
| `session_id` | string | 否   | 会话 ID，用于保持上下文 |

#### 7.2.2 语音驱动（/speech_chat）

```bash
curl -X POST http://localhost:8010/speech_chat \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@/path/to/audio.wav" \
  -F "session_id=session_001"
```

**参数说明：**

| 参数         | 类型   | 必填 | 说明                 |
| ------------ | ------ | ---- | -------------------- |
| `audio`      | file   | 是   | 音频文件（WAV 格式） |
| `session_id` | string | 否   | 会话 ID              |

#### 7.2.3 WebRTC 信令（/offer）

```bash
curl -X POST http://localhost:8010/offer \
  -H "Content-Type: application/json" \
  -d '{
    "sdp": "<SDP_OFFER>",
    "session_id": "session_001"
  }'
```

### 7.3 桌面客户端

如需使用桌面客户端接入，请参考项目文档中的客户端工具说明。

---

## 8. 防火墙与端口

### 8.1 端口说明

| 端口范围       | 协议 | 用途                        |
| -------------- | ---- | --------------------------- |
| TCP 8010       | TCP  | Web 管理界面 + WebRTC 信令 |
| UDP 1-65536    | UDP  | WebRTC 媒体流传输           |
| TCP 8010       | TCP  | REST API 接口               |

### 8.2 Linux / WSL2 防火墙配置

```bash
# Ubuntu / Debian 使用 ufw
sudo ufw allow 8010/tcp     # Web 和信令端口
sudo ufw allow 1:65536/udp  # WebRTC 媒体流

# 或者使用 iptables
sudo iptables -A INPUT -p tcp --dport 8010 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 1:65536 -j ACCEPT
```

### 8.3 Windows 防火墙配置

在 Windows 宿主机上（如通过 WSL2 运行）：

1. 打开 **控制面板 → Windows Defender 防火墙 → 高级设置**
2. 新建 **入站规则**：
   - **TCP 8010**：允许连接
   - **UDP 1-65536**：允许连接
3. 或者使用命令行（管理员）：

```powershell
# 允许 TCP 8010
netsh advfirewall firewall add rule name="LiveTalking-TCP" dir=in action=allow protocol=TCP localport=8010

# 允许 UDP 全端口（WebRTC）
netsh advfirewall firewall add rule name="LiveTalking-UDP" dir=in action=allow protocol=UDP localport=1-65536
```

### 8.4 WSL2 端口转发提示

如果使用 WSL2 运行服务，需确保 Windows 宿主机能访问 WSL2 的端口。通常 WSL2 会自动处理端口转发，但如果遇到连接问题，可在 Windows PowerShell（管理员）中执行：

```powershell
# 查看 WSL2 IP
wsl -- ip addr show eth0

# 手动添加端口转发（如需要）
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8010 connectaddress=<WSL2_IP> connectport=8010
```

---

## 9. 验证

### 9.1 检查服务日志

```bash
# 实时查看日志
tail -f livetalking.log

# 检查是否有错误信息
grep -i "error\|exception\|traceback" livetalking.log
```

### 9.2 检查进程运行状态

```bash
# 检查进程是否存在
ps aux | grep app.py

# 检查端口是否在监听
ss -tlnp | grep 8010
# 或
netstat -tlnp | grep 8010
```

### 9.3 Web UI 验证

1. 打开浏览器访问 `http://localhost:8010/index.html`
2. 确认页面正常加载
3. 检查 WebRTC 连接状态（页面应显示"已连接"或类似信息）
4. 输入文本测试对话，观察数字人是否正常响应

### 9.4 API 验证

**测试文本接口：**

```bash
curl -X POST http://localhost:8010/human \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，这是一条测试消息。"}' \
  -w "\nHTTP状态码: %{http_code}\n"
```

预期返回 HTTP 200，并包含数字人响应内容。

**测试语音接口：**

```bash
# 准备测试音频文件（或使用项目自带的示例音频）
curl -X POST http://localhost:8010/speech_chat \
  -F "audio=@data/avatars/wav2lip256_avatar1/avatar_audio.wav" \
  -w "\nHTTP状态码: %{http_code}\n"
```

### 9.5 GPU 使用验证

```bash
# 监控 GPU 使用情况（另一个终端）
watch -n 1 nvidia-smi
```

运行测试后，应能看到 Python 进程占用 GPU 显存。

---

## 10. 常见问题

### 10.1 CUDA 版本不匹配

**现象**：`RuntimeError: CUDA error: no kernel image is available for execution on the device`

**解决方案**：
1. 确认 `nvidia-smi` 显示的 CUDA 版本
2. 安装对应版本的 PyTorch：https://pytorch.org/get-started/locally/
3. 或回退到 CPU 模式测试（安装 `pip install torch --index-url https://download.pytorch.org/whl/cpu`）

### 10.2 端口被占用

**现象**：`Address already in use` 或 `port 8010 is already in use`

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i :8010

# 终止占用进程
kill -9 <PID>

# 或更换启动端口
python app.py --port 8011
```

### 10.3 模型文件缺失

**现象**：`FileNotFoundError: model file not found` 或 `No such file or directory: 'models/wav2lip.pth'`

**解决方案**：
1. 确认模型文件已放置在正确的目录下
2. 检查文件权限：`chmod 644 models/wav2lip.pth`
3. 重新下载模型文件

### 10.4 网络代理问题

**现象**：LLM API 调用超时或连接失败

**解决方案**：
1. 确认网络能访问 LLM API 地址
2. 如使用代理，设置环境变量：
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```
3. 在 Windows 上，可在 `.env` 中配置代理

### 10.5 显存不足（OOM）

**现象**：`CUDA out of memory` 或 `RuntimeError: CUDA error: out of memory`

**解决方案**：
```bash
# 1. 减小 batch size（如支持）
# 2. 关闭不必要的其他 GPU 程序
# 3. 使用更低分辨率的数字人模型（如 ultralight）
# 4. 在配置中限制最大会话数
max_session: 2   # 减少并发会话

# 5. 清空 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 10.6 WebRTC 连接失败

**现象**：浏览器无法显示视频流，WebRTC 状态显示"未连接"

**解决方案**：
1. 检查防火墙是否开放了 UDP 端口
2. 确认浏览器支持 WebRTC（Chrome / Edge 推荐）
3. 检查服务端日志中是否有 WebRTC 相关错误
4. 尝试使用 HTTP（而非 HTTPS），因为本地测试不需要 SSL
5. 如通过公网访问，需配置 TURN/STUN 服务器

### 10.7 WSL2 相关问题

**现象**：在 WSL2 中运行正常，但 Windows 浏览器无法访问

**解决方案**：
```bash
# 1. 获取 WSL2 IP 地址
wsl -- ip addr show eth0 | grep inet

# 2. 使用 WSL2 IP 而非 localhost 访问
# http://<WSL2_IP>:8010/index.html

# 3. 或配置端口转发（在 Windows PowerShell 管理员模式运行）
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8010 connectaddress=<WSL2_IP> connectport=8010
```

### 10.8 TTS 语音合成失败

**现象**：文本输入后数字人不说话，或日志中 TTS 相关报错

**解决方案**：
1. 检查 `config.yaml` 中 TTS 配置是否正确
2. 如使用 Edge TTS，确认网络能访问 `speech.microsoft.com`
3. 尝试切换其他 TTS 引擎
4. 测试 TTS 单独运行：
   ```bash
   python -c "import edge_tts; print('Edge TTS OK')"
   ```

### 10.9 LLM 响应异常

**现象**：LLM 返回空响应或错误信息

**解决方案**：
1. 检查 API Key 是否正确
2. 确认 API 地址可访问
3. 测试 API 连通性：
   ```bash
   curl -X POST "https://api.siliconflow.cn/v1/chat/completions" \
     -H "Authorization: Bearer $LLM_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hi"}]}'
   ```
4. 检查账户余额

---

## 附录

### A. 目录结构参考

```
LiveTalking/
├── app.py                 # 主入口
├── config.yaml            # 配置文件
├── config.yaml.example    # 配置文件模板
├── requirements.txt       # Python 依赖
├── .env                   # 环境变量（敏感信息）
├── models/                # 模型文件目录
│   └── wav2lip.pth        # Wav2Lip 权重
├── data/
│   └── avatars/           # 数字人形象目录
│       └── wav2lip256_avatar1/
│           ├── avatar_video.mp4
│           └── avatar_audio.wav
├── web/                   # Web 前端文件
│   ├── index.html
│   └── ...
├── scripts/               # 辅助脚本
└── logs/                  # 日志目录（运行后自动生成）
```

### B. 常用命令速查

```bash
# 启动虚拟环境
conda activate livetalking

# 启动服务（默认配置）
python app.py

# 启动服务（指定配置）
python app.py -c config.yaml

# 后台启动
nohup python app.py -c config.yaml > livetalking.log 2>&1 &

# 查看日志
tail -f livetalking.log

# 停止服务
pkill -f "python app.py"
# 或
kill $(pgrep -f "python app.py")

# 查看 GPU 状态
nvidia-smi
```

### C. 技术栈概览

| 组件           | 技术选型                                    |
| -------------- | ------------------------------------------- |
| 数字人模型     | Wav2Lip / MuseTalk / UltraLight             |
| TTS            | Edge TTS / GPT-SoVITS / CosyVoice / 等      |
| ASR            | SenseVoice（本地） / 远程 ASR              |
| LLM            | Qwen2.5-7B-Instruct（SiliconFlow API）      |
| 传输协议       | WebRTC / RTCPush / RTMP / VirtualCam        |
| Web 框架       | FastAPI / Uvicorn                           |
| Web 前端       | HTML + JavaScript + WebRTC Client           |
| Python 版本    | 3.10+                                       |

---

> **文档版本**：v1.0  
> **最后更新**：2026 年 6 月 29 日  
> **原始仓库**：https://github.com/lipku/LiveTalking
