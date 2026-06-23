# LiveTalking 实时交互数字人 — Windows 本地部署

## 环境要求
- Windows 10/11 + NVIDIA 显卡（RTX 3060 及以上）
- CUDA 驱动已安装（`nvidia-smi` 确认版本）
- Anaconda / Miniconda
- Git

---

## 一、安装步骤

### 1. 创建 conda 环境

```powershell
conda create -n livetalking python=3.12 -y
conda activate livetalking
```

> 如果 conda 下载慢或报错，先清源：`conda config --remove-key channels`

### 2. 安装 PyTorch

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. 克隆项目

```powershell
git clone https://github.com/lipku/LiveTalking.git D:\LiveTalking
```

> 如果 GitHub 连不上，用 Gitee 镜像：
> ```powershell
> git clone https://gitee.com/lipku/LiveTalking.git D:\LiveTalking
> ```

### 4. 安装依赖

```powershell
cd D:\LiveTalking
pip install -r requirements.txt
```

### 5. 安装 ffmpeg（录制必需）

```powershell
winget install ffmpeg
```

安装完后关掉 PowerShell 重新打开，验证：
```powershell
ffmpeg -version
```

### 6. 放置模型文件

从夸克网盘下载模型：https://pan.quark.cn/s/83a750323ef0

```powershell
copy "C:\path\to\wav2lip.pth" D:\LiveTalking\models\wav2lip.pth
tar -xf "C:\path\to\wav2lip256_avatar1.zip" -C D:\LiveTalking\data\avatars\
```

---

## 二、启动

```powershell
cd D:\LiveTalking
conda activate livetalking
python app.py --model wav2lip --avatar_id wav2lip256_avatar1
```

看到 `start http server; http://<serverip>:8010/index.html` 即启动成功。

浏览器打开：**http://127.0.0.1:8010/index.html**

---

## 三、使用说明

### 实时交互
1. 点击页面上的 **"开始连接"** 按钮
2. 文本框输入文字
3. 点击 **"发送"** → 数字人实时口型同步说话
4. 新输入文字立即发送即可打断当前说话

### 录制视频
连接成功后，通过 API 控制录制：

```powershell
# 开始录制
curl -X POST "http://127.0.0.1:8010/record" -H "Content-Type: application/json" -d "{\"sessionid\":\"\",\"type\":\"start_record\"}"

# 结束录制
curl -X POST "http://127.0.0.1:8010/record" -H "Content-Type: application/json" -d "{\"sessionid\":\"\",\"type\":\"end_record\"}"
```

录制文件保存在：`D:\LiveTalking\data\record\`

---

## 四、常见问题

### 4.1 conda 网络连接失败
```powershell
conda config --remove-key channels
conda create -n livetalking python=3.12 -y
```

### 4.2 PowerShell 不支持 &&
PowerShell 不支持 `&&` 连接命令，用 `;` 代替：
```powershell
cd D:\LiveTalking; conda activate livetalking; python app.py ...
```

### 4.3 模型加载报错 "PytorchStreamReader failed finding central directory"
模型文件损坏，重新下载并覆盖：
```powershell
copy "C:\path\to\new\wav2lip.pth" D:\LiveTalking\models\wav2lip.pth
```

### 4.4 无法录制 / 没有录像文件
检查 ffmpeg 是否安装：
```powershell
ffmpeg -version
```
未安装则执行：`winget install ffmpeg`，然后重启 LiveTalking。

### 4.5 启动后页面打不开
确认服务在运行（终端有 `start http server` 日志），浏览器访问：
```
http://127.0.0.1:8010/index.html
```
注意必须带 `index.html`。

---

## 五、显卡性能参考

| 显卡 | 模型 | FPS |
|------|------|-----|
| RTX 3060 | wav2lip256 | 60 |
| RTX 3080Ti | wav2lip256 | 120 |
| RTX 4090 | wav2lip256 | 120+ |
| RTX 4070 | wav2lip256 | ~80-100（推测） |
