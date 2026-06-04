from pydantic_settings import BaseSettings
# 导入配置管理基类，自动从环境变量和 .env 文件读取配置

from typing import Optional
# 导入 Optional 类型，表示该字段可以是 None

from config.mysql_config import get_database_url


# 从 mysql_config 模块导入函数，用于获取 MySQL 连接字符串

class Settings(BaseSettings):
    """
    应用配置类

    所有配置项都在这里定义，支持：
    1. 默认值（直接在代码中赋值）
    2. 从 .env 文件覆盖
    3. 从系统环境变量覆盖
    """

    # ========== 应用基本信息 ==========
    APP_NAME: str = "多角色聊天机器人"
    # 应用名称，显示在界面标题等位置

    APP_VERSION: str = "1.0.0"
    # 应用版本号，用于版本管理

    DEBUG: bool = True
    # 调试模式：True 表示开发模式，会输出详细日志
    # 生产环境应设为 False

    # ========== 数据库配置 ==========
    # MySQL 连接信息请填写 config/mysql_config.py。
    # 也可以在 .env 中直接配置 DATABASE_URL 覆盖这里的值。
    DATABASE_URL: str = get_database_url(required=False)
    # MySQL 数据库连接字符串
    # 格式：mysql+pymysql://用户名:密码@主机:端口/数据库名
    # 示例：mysql+pymysql://root:password@localhost:3306/chatbot

    # ========== Redis 配置 ==========
    REDIS_HOST: str = "localhost"
    # Redis 服务器地址，本地开发为 localhost

    REDIS_PORT: int = 6379
    # Redis 端口，默认 6379

    REDIS_DB: int = 0
    # Redis 数据库编号（0-15），默认使用 0

    # ========== Milvus 配置 ==========
    MILVUS_HOST: str = "192.168.190.128"
    # Milvus 向量数据库服务器地址
    # 注意：这是你项目的实际配置地址

    MILVUS_PORT: int = 19530
    # Milvus 端口，默认 19530

    MILVUS_COLLECTION: str = "chatbot_knowledge"
    # 知识库集合名称，存储所有知识向量
    # 你之前看到的 3722 条数据就在这里

    MILVUS_MEMORY_COLLECTION: str = "chatbot_long_term_memory"
    # 长期记忆集合名称，存储对话历史

    # ========== 大模型配置 ==========
    MODEL_TYPE: str = "online"
    # 模型类型：online（在线API）或 local（本地部署）

    API_KEY: Optional[str] = "sk-b605b3ca6d1845fc9de147f751d298f2"
    # API 密钥（已注释掉另一个备选）
    # 用于调用在线大模型服务（如 DeepSeek）

    API_BASE_URL: str = "https://api.deepseek.com/v1/chat/completions"
    # API 接口地址，使用 DeepSeek 的服务

    MODEL_NAME: str = "doubao-seed-1.8-251228"
    PDF_ENABLE_MULTIMODAL: bool = True
    PDF_MM_API_KEY: Optional[str] = "ark-7419c003-7697-4d54-bbec-f08ad08094e5-25ec6"
    PDF_MM_API_BASE_URL: Optional[str] = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    PDF_MM_MODEL: Optional[str] = "doubao-seed-1-8-251228"
    PDF_MM_DPI: int = 180
    PDF_MM_MAX_PAGES: int = 0
    # 模型名称，DeepSeek 的聊天模型

    # ========== 向量化模型配置 ==========
    EMBEDDING_MODEL: str = r"C:\Users\freedom\Desktop\专业\模型\bge-base-zh-v1.5"
    # 向量化模型路径：BGE（BAAI General Embedding）
    # 将文本转换为 768 维向量
    # 注意：使用了绝对路径

    # ========== 重排序模型配置 ==========
    RERANK_MODEL: str = r"C:\Users\freedom\Desktop\专业\模型\bge-reranker-base"
    # 重排序模型路径：用于优化检索结果排序

    RAG_LOAD_MODELS: bool = True
    # 是否启动时加载 embedding 模型；开启后使用 BGE 生成真实语义向量

    RAG_LOAD_RERANK: bool = False
    # 是否加载 reranker；默认关闭，避免启动过慢，后续需要重排时再开启

    # ========== 对话配置 ==========
    MAX_HISTORY_LENGTH: int = 10
    # 最大历史消息数量，RAG 系统使用的上下文长度

    MAX_TOKENS: int = 1000
    # 大模型生成回复的最大 token 数

    # ========== JWT 配置 ==========
    SECRET_KEY: str
    # JWT 签名密钥（没有默认值，必须在 .env 中配置）
    # 用于用户认证令牌的签名

    ALGORITHM: str = "HS256"
    # JWT 签名算法，HS256 是最常用的对称加密算法

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # JWT 令牌过期时间（分钟）

    class Config:
        # Pydantic 配置
        env_file = ".env"
        # 从项目根目录的 .env 文件读取配置

        case_sensitive = True
        # 字段名大小写敏感


# 创建全局配置实例
# 整个应用共享这个配置对象
settings = Settings()
