# LiveTalking 实时交互数字人 部署文档

## 一、方案选择

| 方案 | GPU | WebRTC | 推荐场景 |
|------|-----|--------|---------|
| **AutoDL 4090D** | 云端 24GB | ❌ 需配 SRS 转发 | 批量生成、API 调用 |
| **本地 Windows 4070** | 本地 8GB | ✅ 直通 | 开发测试、实时交互 |

> 本地 Windows 跑是最省事的方案，WebRTC 直通不需要额外配 SRS。

---

## 二、Windows 本地部署（推荐）

### 2.1 环境要求
- NVIDIA RTX 显卡 + CUDA 驱动
- Anaconda / Miniconda
- Git

### 2.2 安装步骤

一条一条在 **PowerShell** 里跑：

```powershell
# 1. 创建环境
conda create -n livetalking python=3.12 -y

# 2. 激活并装 PyTorch
conda activate livetalking
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. 克隆项目
cd D:\
git clone https://github.com/lipku/LiveTalking.git

# 4. 装依赖
cd D:\LiveTalking
pip install -r requirements.txt

# 5. 放模型文件
copy "C:\Users\26332\Desktop\wav2lip.pth" D:\LiveTalking\models\wav2lip.pth
tar -xf "C:\Users\26332\Desktop\wav2lip256_avatar1.zip" -C D:\LiveTalking\data\avatars\
```

### 2.3 启动

```powershell
cd D:\LiveTalking
conda activate livetalking
python app.py --model wav2lip --avatar_id wav2lip256_avatar1
```

浏览器打开 `http://127.0.0.1:8010/index.html`

### 2.4 使用

| 操作 | 说明 |
|------|------|
| 点 **"开始连接"** | 连接 WebRTC，建立数字人会话 |
| 文本框输入文字 → 点 **发送** | 数字人口型同步说话 |
| **打断说话** | 输入新文字直接发送即可 |
| 更换声音 | 在 `config.yaml` 里配置 TTS 参数 |

---

## 三、AutoDL 云端部署

### 3.1 一键安装脚本

在 AutoDL 终端里跑：

```bash
# 写脚本到工作目录
cat > /root/autodl-tmp/setup_livetalking.sh << 'EOF'
#!/bin/bash
set -e
cd /root/autodl-tmp
git clone https://github.com/lipku/LiveTalking.git
cd /root/autodl-tmp/LiveTalking
conda create -n livetalking python=3.12 -y
source activate livetalking
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
mkdir -p models data/avatars
apt-get update && apt-get install ffmpeg -y
EOF

# 执行脚本
bash /root/autodl-tmp/setup_livetalking.sh
```

### 3.2 模型文件

下载地址：https://pan.quark.cn/s/83a750323ef0

| 文件 | 放置路径 |
|------|---------|
| `wav2lip256.pth` → 改名 `wav2lip.pth` | `models/wav2lip.pth` |
| `wav2lip256_avatar1.tar.gz` → 解压 | `data/avatars/wav2lip256_avatar1/` |

> ⚠ AutoDL 文件管理器不支持拖放文件夹，先上传 tar.gz 再用命令解压：
> ```bash
> cd /root/autodl-tmp/LiveTalking/data/avatars && tar -xzf wav2lip256_avatar1.tar.gz
> ```

> ⚠ AutoDL 网页上传大文件（~200MB）容易损坏，如果报 `PytorchStreamReader failed finding central directory`，改用 gdown 从 Google Drive 下：
> ```bash
> pip install gdown
> rm -f /root/autodl-tmp/LiveTalking/models/wav2lip.pth
> gdown --folder 'https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ' \
>   -O /root/autodl-tmp/LiveTalking/models/ --folder --remaining-ok
> ```

### 3.3 启动

```bash
# 使用 screen 保持后台运行，防止 SSH 断开中断
screen -S livetalking

cd /root/autodl-tmp/LiveTalking
source activate livetalking
python app.py --model wav2lip --avatar_id wav2lip256_avatar1 --listenport 6006

# Ctrl+A D 分离 screen
```

### 3.4 访问

1. AutoDL 控制台 → 实例 → **自定义服务**
2. 添加映射：本地端口 `6006` → 实例端口 `6006`
3. 浏览器打开：`https://<分配的地址>:8443/index.html`

> ⚠ 必须带 `/index.html`，只访问根路径返回 403

### 3.5 AutoDL 的限制

| 问题 | 原因 | 解决 |
|------|------|------|
| WebRTC 连不上 | AutoDL 封闭 UDP 端口 | 配 SRS 转发（见下文）或改用本地部署 |
| API 报 "session not found" | 需要 WebRTC 建 session | 同上 |
| 大文件上传播坏 | 网页上传不稳定 | 用 gdown 命令行下载 |

### 3.6 配 SRS 公网转发（可选）

需要一台有公网 IP 的服务器（如腾讯云/阿里云轻量服务器）。

在公网服务器上部署 SRS：

```bash
export CANDIDATE=<公网服务器IP>
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

在 AutoDL 上启动（rtcpush 模式）：

```bash
cd /root/autodl-tmp/LiveTalking
source activate livetalking
python app.py --listenport 6006 --transport rtcpush --model wav2lip \
  --avatar_id wav2lip256_avatar1 \
  --push_url 'http://<SRS公网IP>:1985/rtc/v1/whip/?app=live&stream=livestream'
```

访问：`http://<AutoDL地址>:6006/rtcpushapi.html`

---

## 四、常见问题

### 4.1 conda 网络连不上

```powershell
conda config --remove-key channels
conda create -n livetalking python=3.12 -y
```

### 4.2 PyTorch 加载模型报错 "PytorchStreamReader failed finding central directory"

模型文件上传播坏了。删掉重传：

```bash
rm -f models/wav2lip.pth
```

然后用 gdown 命令行下载（不要用网页上传）：

```bash
pip install gdown
gdown --folder 'https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ' \
  -O models/ --folder --remaining-ok
```

### 4.3 PowerShell 不能使用 `&&`

Windows PowerShell 不支持 `&&`，用 `;` 代替：
```powershell
cd D:\LiveTalking; conda activate livetalking; python app.py ...
```

### 4.4 ffmpeg 未安装告警

不影响核心功能，但建议安装：

```bash
apt-get update && apt-get install ffmpeg -y   # Linux
```
```powershell
winget install ffmpeg   # Windows
```

---

## 五、接口说明

| 接口 | 方法 | 说明 | 需要 session |
|------|------|------|-------------|
| `/index.html` | GET | WebRTC 前端页面 | ❌ |
| `/human` | POST | 文本驱动数字人 | ✅ |
| `/humanaudio` | POST | 音频驱动数字人 | ✅ |
| `/record` | POST | 开始/停止录制 | ✅ |
| `/api/admin/sessions` | GET | 查看活跃会话 | ❌ |

> 所有需要 session 的接口必须先通过 WebRTC 页面连接成功后使用，否则返回 "session not found"。
