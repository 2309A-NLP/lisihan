from pydantic_settings import BaseSettings
# 导入 pydantic_settings 的 BaseSettings 类
# 这个类会自动从环境变量和 .env 文件读取配置

import os

class Settings(BaseSettings):
    """
    配置类

    继承自 BaseSettings，会自动：
    1. 从 .env 文件读取配置
    2. 从系统环境变量读取配置
    3. 进行类型验证和转换

    字段必须和 .env 文件中的变量名一一对应
    """

    # 声明配置字段（必须和 .env 里的变量名完全一致）
    SECRET_KEY: str
    # JWT 签名密钥，用于生成和验证令牌
    # 例如: "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"

    ALGORITHM: str
    # JWT 签名算法
    # 例如: "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # 访问令牌过期时间（分钟）
    # 例如: 30


    class Config:
        """
        配置类的配置（元配置）
        指定 .env 文件的位置和读取方式
        """

        # 关键：指定 .env 文件的路径
        # os.path.dirname(__file__) 获取当前文件所在目录（config/）
        # os.path.join("..", ".env") 向上两级到项目根目录的 .env 文件
        # 完整路径: 项目根目录/.env
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")

        # 指定 .env 文件的编码格式
        env_file_encoding = "utf-8"

        # 字段名称大小写敏感
        # True 表示 SECRET_KEY 和 secret_key 是不同的变量
        case_sensitive = True


# 创建全局配置实例
# 这个实例会在导入时自动读取 .env 文件
settings = Settings()

# 测试代码（仅在直接运行此文件时执行）
if __name__ == "__main__":
    # 打印配置值（密钥只显示前10个字符）
    print("读取到的 SECRET_KEY:", settings.SECRET_KEY[:10] + "******")
    print("读取到的算法:", settings.ALGORITHM)
    print("读取到的过期时间:", settings.ACCESS_TOKEN_EXPIRE_MINUTES)