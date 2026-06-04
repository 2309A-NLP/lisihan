"""
FastAPI主应用模块
================
这是整个聊天机器人系统的入口文件，负责：
1. 初始化FastAPI应用
2. 配置中间件（CORS、异常处理）
3. 注册各个功能模块的路由
4. 提供前端静态文件服务
5. 应用启动时的初始化工作
"""

# ========== 导入基础框架和响应类 ==========
from fastapi import FastAPI  # FastAPI核心类，用于创建Web应用
from fastapi.responses import FileResponse  # 用于返回文件响应（如HTML、CSS、JS文件）
from fastapi.middleware.cors import CORSMiddleware  # CORS中间件，处理跨域请求
from fastapi.staticfiles import StaticFiles  # 静态文件服务，用于托管CSS、JS等静态资源

# ========== 导入自定义模块 ==========
from app.core.logging import log_exception, setup_logging  # 日志系统：记录异常和设置日志配置
import uvicorn  # ASGI服务器，用于运行FastAPI应用
import os  # 操作系统接口，用于处理文件路径
import warnings  # 警告控制模块，用于过滤不需要的警告信息
import logging  # 日志记录模块
from starlette.requests import Request  # Starlette的请求对象，用于获取请求信息

# ========== 导入API路由模块 ==========
from app.api import chat, role, user, knowledge  # 导入各功能模块的路由
from app.api import download, admin  # 导入下载和管理员模块的路由
from app.core.memory import initialize_memory  # 内存管理初始化函数
from app.core.rag import init_rag  # RAG（检索增强生成）系统初始化函数
from app.core.vectorstore import init_milvus  # Milvus向量数据库初始化函数
from config import settings  # 应用配置信息（JWT密钥、数据库配置等）
from app.models import initialize_database  # 数据库初始化函数

# ========== 初始化日志系统 ==========
# 获取当前文件的绝对路径所在的目录（即app目录的父目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建前端文件目录路径（frontend文件夹）
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# 构建前端静态资源目录路径（frontend/assets文件夹）
FRONTEND_ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")

# 设置日志系统（配置日志格式、输出位置等）
setup_logging()
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# ========== 过滤警告信息 ==========
# 过滤掉pkg_resources已弃用的警告（这个警告不影响程序运行）
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
# 过滤掉transformers库的未来警告（避免控制台输出过多警告）
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# ========== 创建FastAPI应用实例 ==========
app = FastAPI(
    title="多角色聊天机器人API",  # API的标题，会显示在/docs文档页面
    description="基于RAG的多角色聊天机器人系统",  # API的详细描述
    version="1.0.0"  # API版本号
)

# ========== 配置CORS中间件 ==========
# CORS用于解决跨域问题，允许前端页面从不同域名访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名访问（生产环境应该限制具体域名）
    allow_credentials=True,  # 允许携带凭证（如Cookie）
    allow_methods=["*"],  # 允许所有HTTP方法（GET、POST、PUT、DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


# ========== 自定义中间件：全局异常捕获 ==========
@app.middleware("http")  # 装饰器，声明这是一个HTTP请求级别的中间件
async def log_unhandled_exceptions(request: Request, call_next):
    """
    记录未捕获异常，日志中包含文件、行号、函数名、原因和完整堆栈。

    这个中间件会捕获所有未被处理的异常，并记录到日志中，
    避免服务器直接崩溃，同时便于调试。
    """
    try:
        # 正常处理请求，调用下一个中间件或路由处理函数
        return await call_next(request)
    except Exception as exc:
        # 发生异常时，调用自定义的异常记录函数
        log_exception(
            logger,  # 日志记录器
            "未捕获请求异常: method=%s path=%s client=%s",  # 日志消息模板
            exc,  # 异常对象
            request.method,  # HTTP方法（GET、POST等）
            request.url.path,  # 请求路径
            request.client.host if request.client else "unknown",  # 客户端IP地址
        )
        # 重新抛出异常，让FastAPI处理返回错误响应
        raise


# ========== 应用启动事件 ==========
@app.on_event("startup")  # 装饰器，声明在应用启动时执行
async def startup_event():
    """
    应用启动初始化，避免 uvicorn reload 父进程重复加载知识库。

    这个函数在服务器启动时执行一次，负责初始化：
    1. 内存管理系统
    2. 向量数据库（Milvus）
    3. RAG检索系统
    4. 数据库表结构
    """
    logger.info("应用启动初始化开始")

    # 导入异步IO模块和线程池模块
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    # 获取当前事件循环
    loop = asyncio.get_event_loop()

    # 创建线程池执行器（因为某些初始化函数是同步的，不能在异步函数中直接调用）
    with ThreadPoolExecutor() as executor:
        # 在线程池中执行同步初始化函数，避免阻塞事件循环
        await loop.run_in_executor(executor, initialize_memory)  # 初始化内存管理
        await loop.run_in_executor(executor, init_milvus)  # 初始化Milvus向量数据库
        await loop.run_in_executor(executor, init_rag)  # 初始化RAG检索系统

    # 初始化数据库（创建表结构）
    try:
        logger.info("数据库初始化开始")
        initialize_database()  # 创建所有定义好的数据库表
        logger.info("数据库表检查完成")
    except Exception as exc:
        # 如果数据库已存在或有其他错误，记录日志但不中断启动
        log_exception(logger, "数据库表初始化跳过", exc)

    logger.info("应用启动初始化完成")


# ========== 注册API路由 ==========
# 将各个功能模块的路由注册到主应用，并指定URL前缀和标签
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])  # 聊天相关API
app.include_router(role.router, prefix="/api/role", tags=["role"])  # 角色管理API
app.include_router(user.router, prefix="/api/user", tags=["user"])  # 用户管理API
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["knowledge"])  # 知识库管理API
app.include_router(download.router, prefix="/api/download", tags=["下载"])  # 文件下载API
app.include_router(admin.router, prefix="/api/admin", tags=["管理员"])  # 管理员功能API

# ========== 挂载静态文件目录 ==========
# 将前端静态资源目录挂载到/assets路径下
app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="frontend-assets")


# ========== 辅助函数：获取前端文件路径 ==========
def frontend_file(filename: str) -> str:
    """根据文件名返回完整的前端文件路径"""
    return os.path.join(FRONTEND_DIR, filename)


# ========== 健康检查API ==========
@app.get("/health")
def health_check():
    """简单的健康检查接口，用于监控系统是否正常运行"""
    return {"status": "healthy"}


# ========== 测试JWT配置API ==========
@app.get("/test-jwt-config")
def test_config():
    """
    暴露JWT配置信息（生产环境应该删除此接口）
    用于调试JWT配置是否正确
    """
    return {
        "算法": settings.ALGORITHM,  # JWT签名算法
        "过期时间": settings.ACCESS_TOKEN_EXPIRE_MINUTES,  # Token过期时间（分钟）
        "密钥前缀": settings.SECRET_KEY[:10] + "******"  # 密钥前缀（隐藏完整密钥）
    }


# ========== 前端页面服务 ==========
# 每个前端页面都通过FileResponse返回HTML文件

@app.get("/")
def frontend():
    """服务首页（聊天界面）"""
    return FileResponse(frontend_file("index.html"))


@app.get("/login")
def login_page():
    """服务登录页面"""
    return FileResponse(frontend_file("login.html"))


@app.get("/admin")
def admin_page():
    """服务管理员界面"""
    return FileResponse(frontend_file("admin.html"))


@app.get("/admin.html")
def admin_html_page():
    """兼容性路由，支持直接访问admin.html"""
    return FileResponse(frontend_file("admin.html"))


@app.get("/download")
def download_page():
    """服务下载页面"""
    return FileResponse(frontend_file("download.html"))


@app.get("/download.html")
def download_html_page():
    """兼容性路由，支持直接访问download.html"""
    return FileResponse(frontend_file("download.html"))


@app.get("/style.css")
def style_file():
    """服务前端样式表文件"""
    return FileResponse(os.path.join(FRONTEND_ASSETS_DIR, "style.css"))


@app.get("/script.js")
def script_file():
    """服务前端JavaScript脚本文件"""
    return FileResponse(os.path.join(FRONTEND_ASSETS_DIR, "script.js"))


# ========== API根路径 ==========
@app.get("/api")
def root():
    """API根路径，返回简单的服务描述信息"""
    return {"message": "多角色聊天机器人API服务"}


# ========== 网站图标服务 ==========
@app.get("/favicon.ico")
async def favicon():
    """
    服务网站图标（favicon）
    如果图标文件存在则返回，否则返回None（浏览器会显示默认图标）
    """
    favicon_path = os.path.join(FRONTEND_ASSETS_DIR, "favicon.ico")
    # 判断文件是否存在，存在才返回，否则返回None
    return FileResponse(favicon_path) if os.path.exists(favicon_path) else None


# ========== 直接运行服务器 ==========
def run_server():
    """直接运行FastAPI服务的函数"""
    print("正在启动多角色聊天机器人API服务...")
    print("前端页面: http://127.0.0.1:8000")
    print("管理员界面: http://127.0.0.1:8000/admin")
    print("下载页面: http://127.0.0.1:8000/download")
    print("健康检查: http://127.0.0.1:8000/health")
    print("API 文档: http://127.0.0.1:8000/docs")
    print("按 Ctrl+C 停止服务")
    print("\n服务已启动，正在运行中...")

    # 使用uvicorn启动FastAPI应用
    uvicorn.run(
        app,  # 直接传入app对象，而不是字符串（避免重新加载问题）
        host="127.0.0.1",  # 监听本地回环地址，只能本地访问
        port=8000,  # 监听端口号
        reload=False,  # 禁用自动重载（生产环境建议关闭）
        log_level="info"  # 日志级别
    )


# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 当直接运行此文件时，启动服务
    run_server()