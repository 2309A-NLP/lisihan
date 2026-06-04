from urllib.parse import quote_plus

# 导入 URL 编码函数
# 用于对用户名和密码进行百分号编码，防止特殊字符导致连接失败
# 例如：密码中的 @ 符号会被编码为 %40


# ========== MySQL 配置参数 ==========
# 请按你的本机 MySQL 实际信息填写下面几项。

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_DATABASE = "rag"
MYSQL_CHARSET = "utf8"

# Function: Check whether MySQL settings are present.
def is_mysql_configured() -> bool:
    """
    检查 MySQL 配置是否完整

    检查所有必要配置项是否都已填写

    Returns:
        bool: True 表示配置完整，False 表示缺少必要配置
    """
    return bool(
        MYSQL_HOST and  # 主机地址不为空
        MYSQL_PORT and  # 端口不为空
        MYSQL_USER and  # 用户名不为空
        MYSQL_DATABASE  # 数据库名不为空
        # 注意：密码可以为空（允许无密码登录）
    )


# Function: Build or return the SQLAlchemy database URL.
def get_database_url(required: bool = False) -> str:
    """
    生成数据库连接字符串（URL）

    根据配置生成符合 SQLAlchemy 格式的连接字符串

    Args:
        required: 是否必需配置
                 - True: 如果配置不完整，抛出异常
                 - False: 如果配置不完整，返回空字符串

    Returns:
        str: 数据库连接字符串
             格式: mysql+mysqlconnector://user:password@host:port/db?charset=utf8

    Raises:
        RuntimeError: 当 required=True 且配置不完整时抛出
    """
    # ========== 检查配置完整性 ==========
    if not is_mysql_configured():
        if required:
            # 配置不完整且要求必需时，抛出错误
            raise RuntimeError(
                "MySQL 配置未完成：请先填写 config/mysql_config.py 里的 "
                "MYSQL_HOST、MYSQL_PORT、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE。"
            )
        # 配置不完整但不要求必需时，返回空字符串
        return ""

    # ========== URL 编码用户名和密码 ==========
    # quote_plus 会特殊字符转义
    # 例如: "admin@123" → "admin%40123"
    #       "pass word" → "pass+word"
    user = quote_plus(MYSQL_USER)
    password = quote_plus(MYSQL_PASSWORD)

    # ========== 其他参数直接使用 ==========
    host = MYSQL_HOST
    port = int(MYSQL_PORT)  # 确保是整数类型
    database = MYSQL_DATABASE
    charset = MYSQL_CHARSET

    # ========== 构建连接字符串 ==========
    # SQLAlchemy 连接字符串格式：
    # dialect+driver://username:password@host:port/database?params
    #
    # mysql+mysqlconnector: 使用 MySQL 数据库 + mysql-connector-python 驱动
    # charset=utf8: 指定字符集
    return (
        f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        f"?charset={charset}"
    )