LiveTalking 实时数字人引擎 思维导图
│
├── 📦 部署流程
│   ├── 环境准备
│   │   ├── 系统: Ubuntu 22.04 / Windows + WSL2
│   │   ├── GPU: NVIDIA 显卡, CUDA 11.6+, 显存≥8GB
│   │   ├── Python 3.10+ (推荐 3.12)
│   │   └── Conda 环境管理
│   │
│   ├── 项目获取
│   │   ├── git clone https://github.com/lipku/LiveTalking.git
│   │   └── 或使用已有代码
│   │
│   ├── 环境配置
│   │   ├── conda create -n livetalking python=3.12
│   │   ├── pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
│   │   └── pip install -r requirements.txt
│   │
│   ├── 模型下载
│   │   ├── wav2lip.pth → models/wav2lip.pth
│   │   └── wav2lip256_avatar1 → data/avatars/
│   │
│   ├── 配置文件
│   │   ├── config.yaml (TTS/ASR/LLM/传输协议)
│   │   └── .env (API Keys)
│   │
│   └── 启动服务
│       ├── python app.py --transport webrtc --model wav2lip
│       └── 访问: http://serverip:8010/index.html
│
├── 🏗️ 系统架构
│   ├── 用户层
│   │   ├── 🌐 浏览器 (Web UI)
│   │   ├── 📱 桌面客户端
│   │   └── 🔗 API 调用
│   │
│   ├── 通信层
│   │   ├── WebRTC (默认, 低延迟)
│   │   ├── RTMP (直播推流)
│   │   ├── RTCPush (SRS)
│   │   └── VirtualCam (虚拟摄像头)
│   │
│   ├── 应用层 API (端口 8010)
│   │   ├── /human ── 文本驱动数字人
│   │   │   ├── type=echo: 直接文本输出
│   │   │   └── type=chat: LLM生成回复再输出
│   │   ├── /speech_chat ── 语音驱动 (ASR→LLM→TTS→Avatar)
│   │   ├── /offer ── WebRTC 信令
│   │   ├── /humanaudio ── 上传音频驱动
│   │   ├── /api/asr_http ── 语音识别
│   │   ├── /interrupt_talk ── 打断说话
│   │   ├── /is_speaking ── 查询说话状态
│   │   ├── /record ── 录制控制
│   │   ├── /set_audiotype ── 动作编排状态切换
│   │   ├── /api/capabilities ── 能力检测
│   │   ├── /api/admin/config ── 管理配置
│   │   ├── /api/admin/sessions ── 管理会话
│   │   └── /api/avatar/task ── 形象生成任务
│   │
│   ├── 引擎层
│   │   ├── 🧠 LLM (硅基流动/SiliconFlow)
│   │   │   ├── 模型: Qwen/Qwen2.5-7B-Instruct
│   │   │   ├── API: https://api.siliconflow.cn/v1
│   │   │   └── 流式输出, 逐句推送给数字人
│   │   │
│   │   ├── 🎤 TTS 引擎 (10种)
│   │   │   ├── Edge TTS (默认, 免费)
│   │   │   ├── GPT-SoVITS / CosyVoice / FishTTS
│   │   │   ├── 腾讯云 / 火山豆包 / Azure
│   │   │   ├── IndexTTS2 / QwenTTS / OmniTTS
│   │   │   └── 参数: voice, emotion 透传
│   │   │
│   │   ├── 🎯 ASR 语音识别
│   │   │   ├── 本地: SenseVoiceSmall (FunASR)
│   │   │   └── 远程: HTTP ASR 端点
│   │   │
│   │   └── 👤 数字人模型
│   │       ├── Wav2Lip (默认)
│   │       ├── MuseTalk
│   │       └── UltraLight-Digital-Human
│   │
│   └── 基础设施层
│       ├── OS: Ubuntu / Windows / macOS
│       ├── GPU + CUDA 推理
│       ├── Python 3.10+
│       └── 端口: TCP 8010, UDP 1-65535
│
├── 🔧 核心工作流
│   ├── 文本模式: 用户文字 → LLM → TTS → 数字人 → WebRTC → 浏览器
│   ├── 语音模式: 用户语音 → ASR → LLM → TTS → 数字人 → WebRTC → 浏览器
│   └── 回声模式: 用户文字/语音 → 直接TTS → 数字人
│
├── ✅ 已做优化 (9项)
│   ├── ① YAML 配置文件支持
│   │   ├── 配置文件: config.yaml + config.py
│   │   ├── 优先级: CLI > YAML > 默认值
│   │   └── 效果: 配置管理标准化
│   │
│   ├── ② 本地 SenseVoice ASR 集成
│   │   ├── 文件: server/asr_server.py, benchmark_asr.py
│   │   ├── 支持离线语音识别 (FunASR + ModelScope)
│   │   └── 效果: 不依赖外部API, 无网络也可用
│   │
│   ├── ③ OmniTTS 集成
│   │   ├── 文件: tts/omnitts.py
│   │   └── 效果: 多一种TTS引擎选择
│   │
│   ├── ④ .env 环境变量支持
│   │   ├── 文件: .env.example, app.py (load_dotenv)
│   │   ├── 管理 API Keys (SILICONFLOW_API_KEY 等)
│   │   └── 效果: 密钥不硬编码, 安全性提升
│   │
│   ├── ⑤ 会话管理优化
│   │   ├── 文件: server/session_manager.py, task_manager.py
│   │   ├── 多会话管理 (max_session=5)
│   │   ├── sessionid: int→str (UUID)
│   │   └── 效果: 并发会话可控, 避免冲突
│   │
│   ├── ⑥ TTS 引擎扩展
│   │   ├── QwenTTS (tts/qwentts.py)
│   │   ├── Azure TTS 修复 (tts/azure.py)
│   │   ├── Edge TTS 修复 (tts/edge.py)
│   │   ├── 腾讯云 TTS 修复 (tts/tencent.py)
│   │   ├── TTS API 服务 (server/tts_api_server.py)
│   │   └── 效果: TTS引擎覆盖主流方案
│   │
│   ├── ⑦ Web 前端优化
│   │   ├── 更新 index.html / index-en.html (多语言)
│   │   ├── TTS 测试页面 (web/tts/)
│   │   ├── ASR 前端组件 (web/asr/main.js)
│   │   └── 效果: 前端体验完整
│   │
│   ├── ⑧ ASR 性能基准测试
│   │   ├── 文件: benchmark_asr.py (264行)
│   │   └── 效果: 可量化ASR准确率和速度
│   │
│   └── ⑨ 插件化架构注册表
│       ├── 文件: registry.py
│       ├── 支持 stt/llm/tts/avatar/output 五类
│       ├── @register 装饰器 + registry.create()
│       └── 效果: 架构解耦, 便于扩展新引擎
│
└── 📋 待做优化 (11项)
    ├── ① 日志轮转 (优先级: 高)
    │   ├── 问题: livetalking.log 无限增长 (已550KB)
    │   ├── 方案: RotatingFileHandler, 10MB, 5备份
    │   └── 预期: 磁盘空间安全
    │
    ├── ② GPU 内存泄漏排查 (优先级: 高)
    │   ├── 问题: 长时间运行显存可能持续增长
    │   ├── 方案: 定期 empty_cache, 显式释放中间张量
    │   └── 预期: 服务稳定运行
    │
    ├── ③ 健康检查端点 (优先级: 高)
    │   ├── 问题: 无 /health 端点
    │   ├── 方案: GET /health 返回状态/GPU/会话数
    │   └── 预期: 便于监控集成
    │
    ├── ④ 配置热重载 (优先级: 中)
    │   ├── 问题: 修改配置需重启服务
    │   ├── 方案: SIGHUP 信号 / API 端点
    │   └── 预期: 零停机更新配置
    │
    ├── ⑤ Docker Compose 现代化 (优先级: 中)
    │   ├── 问题: Dockerfile CUDA 11.6 + Python 3.10
    │   ├── 方案: 升级到 CUDA 12.x + Python 3.12
    │   └── 预期: 部署标准化
    │
    ├── ⑥ 错误处理增强 (优先级: 中)
    │   ├── 问题: API错误暴露Python异常堆栈
    │   ├── 方案: 统一错误格式, 友好中文提示
    │   └── 预期: 用户体验提升
    │
    ├── ⑦ WebRTC 连接稳定性 (优先级: 中)
    │   ├── 问题: 默认STUN可能NAT穿透失败
    │   ├── 方案: 自建TURN服务器 + 重连机制
    │   └── 预期: 连接成功率提升
    │
    ├── ⑧ 模型热加载 (优先级: 中)
    │   ├── 问题: 切换avatar_id需重启服务
    │   ├── 方案: API端点动态加载/卸载模型
    │   └── 预期: 运行时切换数字人形象
    │
    ├── ⑨ API 文档完善 (优先级: 低)
    │   ├── 问题: 文档不完整, 缺请求/响应示例
    │   ├── 方案: OpenAPI/Swagger 文档
    │   └── 预期: 对接开发更方便
    │
    ├── ⑩ 前端多语言国际化 (优先级: 低)
    │   ├── 问题: 部分页面仅中文
    │   ├── 方案: i18n 统一模板
    │   └── 预期: 国际化支持
    │
    └── ⑪ 多并发压力测试 (优先级: 低)
        ├── 问题: max_session=5 未经过压力测试
        ├── 方案: locust/jmeter 并发测试
        └── 预期: 性能瓶颈可见
