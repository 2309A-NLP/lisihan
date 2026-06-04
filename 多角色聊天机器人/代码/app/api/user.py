"""
用户管理API模块
================
这个模块提供用户系统的管理接口，包括：
- 用户注册（创建新账号）
- 用户登录（JWT令牌认证）
- 获取用户资料
- 更新用户资料（修改用户名、邮箱、密码）

认证机制：
- 使用 JWT (JSON Web Token) 进行身份认证
- 登录成功后返回 access_token
- 后续请求需要在 Header 中携带 token
- Token 有过期时间（默认30分钟或根据配置）

用户数据类型：
- 存储在 MySQL 数据库的 users 表中
- 密码使用 bcrypt 加密存储（绝对不存明文）
"""

from fastapi import APIRouter, HTTPException
# APIRouter: 创建API路由分组
# HTTPException: 返回标准HTTP错误响应（400, 401, 404等）

from pydantic import BaseModel
# BaseModel: Pydantic数据验证基类，自动验证请求数据格式

from app.services.user_service import UserService
# UserService: 用户业务逻辑服务，处理用户的数据库操作和密码验证

from datetime import timedelta
# timedelta: 时间差对象，用于设置JWT令牌的过期时间

from config.config import settings

# settings: 应用配置对象，包含 JWT_SECRET、JWT_EXPIRE_MINUTES 等配置

# ========== 创建API路由器 ==========
# 所有用户相关的API都会以 /user 为前缀
# 完整路径示例: http://localhost:8000/api/user/register
router = APIRouter()


# ========== Pydantic 请求/响应模型 ==========
class RegisterRequest(BaseModel):
    """
    用户注册请求数据模型

    用户注册时需要提供的信息

    字段说明：
        username: 用户名（唯一），用于登录和显示
        password: 明文密码（后端会加密存储）
        email: 电子邮箱，用于找回密码或通知

    注意：
        - 用户名不能重复
        - 密码长度应该有前端验证（至少6位）
        - 邮箱格式会被Pydantic自动验证
    """
    username: str  # 用户名
    password: str  # 密码（明文传输，后端加密）
    email: str  # 电子邮箱


class LoginRequest(BaseModel):
    """
    用户登录请求数据模型

    登录时需要提供的凭证

    字段说明：
        username: 用户名
        password: 密码（明文传输，后端验证）

    注意：
        - 密码在传输过程中应使用 HTTPS 加密
        - 后端会验证密码是否正确
    """
    username: str  # 用户名
    password: str  # 密码


class UserResponse(BaseModel):
    """
    用户基本信息响应模型

    注册成功后返回的最小用户信息

    字段说明：
        user_id: 用户唯一标识（数据库自增ID）
        username: 用户名
    """
    user_id: int  # 用户ID
    username: str  # 用户名


# Function: Return non-sensitive user profile information.
class UserProfileResponse(BaseModel):
    """
    用户完整资料响应模型

    获取用户资料时返回的完整信息

    字段说明：
        user_id: 用户ID
        username: 用户名
        email: 电子邮箱

    注意：
        - 不返回密码字段（安全考虑）
        - 不返回敏感信息
    """
    user_id: int  # 用户ID
    username: str  # 用户名
    email: str  # 电子邮箱

class UserUpdateRequest(BaseModel):
    """
    更新用户资料请求模型

    用户可以修改自己的信息，所有字段都是可选的

    字段说明：
        username: 新的用户名（可选）
        email: 新的邮箱（可选）
        password: 新的密码（可选）

    使用场景：
        - 只修改用户名：只传 username
        - 只修改密码：只传 password
        - 同时修改多个：传入多个字段
    """
    username: str | None = None  # 新用户名（可选）
    email: str | None = None  # 新邮箱（可选）
    password: str | None = None  # 新密码（可选）

class TokenResponse(BaseModel):
    """
    JWT令牌响应模型

    登录成功后返回的认证令牌

    字段说明：
        access_token: JWT访问令牌（需要在后续请求的Header中携带）
        token_type: 令牌类型，通常为 "bearer"
        user_id: 用户ID（方便前端使用）
        username: 用户名（方便前端显示）

    使用示例：
        登录成功后：
        {
            "access_token": "eyJhbGciOiJIUzI1NiIs...",
            "token_type": "bearer",
            "user_id": 123,
            "username": "张三"
        }

        后续请求：
        Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    """
    access_token: str  # JWT访问令牌
    token_type: str  # 令牌类型（bearer）
    user_id: int  # 用户ID
    username: str  # 用户名


# ========== API 端点 1: 用户注册 ==========
# HTTP POST /api/user/register
@router.post("/register", response_model=UserResponse)
# response_model=UserResponse 表示返回用户基本信息
async def register(request: RegisterRequest):
    """
    用户注册接口

    功能说明：
    - 创建新的用户账号
    - 验证用户名、邮箱是否已存在
    - 密码使用 bcrypt 加密后存储
    - 返回用户基本信息

    注册流程：
    1. 接收用户名、密码、邮箱
    2. 检查用户名是否已被注册
    3. 检查邮箱是否已被使用
    4. 对密码进行哈希加密
    5. 在 users 表中插入新记录
    6. 返回用户ID和用户名

    安全措施：
    - 密码绝不存储明文
    - 使用 bcrypt 加盐哈希（抗彩虹表攻击）
    - 限制用户名和密码长度
    - 防止 SQL 注入（使用ORM）

    请求示例：
    POST /api/user/register
    Content-Type: application/json

    {
        "username": "张三",
        "password": "mypassword123",
        "email": "zhangsan@example.com"
    }

    返回示例（成功）：
    {
        "user_id": 123,
        "username": "张三"
    }

    返回示例（失败）：
    {
        "detail": "注册失败: 用户名已存在"
    }

    参数：
        request: RegisterRequest 对象

    返回：
        UserResponse: 新用户的ID和用户名

    可能的错误：
        - 400: 用户名已存在、邮箱已注册、参数格式错误
        - 500: 数据库错误
    """
    try:
        # 实例化用户服务
        # UserService 封装了所有用户相关的业务逻辑
        user_service = UserService()

        # 调用服务层注册用户
        # register 方法会：
        # 1. 验证用户名、邮箱是否已存在
        # 2. 验证密码复杂度（可选）
        # 3. 加密密码
        # 4. 保存到数据库
        # 5. 返回新用户的ID
        user_id = user_service.register(
            username=request.username,  # 用户名
            password=request.password,  # 明文密码（后端会加密）
            email=request.email  # 电子邮箱
        )

        # 返回注册成功信息
        # 注意：不返回密码、邮箱等敏感信息
        return UserResponse(user_id=user_id, username=request.username)
    except Exception as e:
        # 注册失败返回400错误
        # 常见失败原因：
        # - "用户名已存在"
        # - "邮箱已被注册"
        # - "密码长度不足"
        raise HTTPException(status_code=400, detail=f"注册失败: {str(e)}")


# ========== API 端点 2: 用户登录 ==========
# HTTP POST /api/user/login
@router.post("/login", response_model=TokenResponse)
# response_model=TokenResponse 表示返回JWT令牌
async def login(request: LoginRequest):
    """
    用户登录接口

    功能说明：
    - 验证用户名和密码
    - 生成 JWT 访问令牌
    - 返回令牌和用户信息

    登录流程：
    1. 接收用户名和密码
    2. 在数据库中查找用户
    3. 验证密码是否正确（使用 bcrypt 比较）
    4. 生成 JWT 令牌（包含用户ID等信息）
    5. 设置令牌过期时间
    6. 返回令牌和用户信息

    JWT 令牌说明：
    - 使用 HS256 算法签名
    - 包含 payload: {"sub": user_id, "exp": 过期时间}
    - 有效期由 settings.JWT_EXPIRE_MINUTES 配置
    - 令牌包含数字签名，防止篡改

    安全机制：
    - 密码验证失败会延迟响应（防止暴力破解）
    - 登录失败记录日志
    - 令牌无法伪造（需要 JWT_SECRET）

    请求示例：
    POST /api/user/login
    Content-Type: application/json

    {
        "username": "张三",
        "password": "mypassword123"
    }

    返回示例（成功）：
    {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "bearer",
        "user_id": 123,
        "username": "张三"
    }

    返回示例（失败）：
    {
        "detail": "登录失败: 用户名或密码错误"
    }

    参数：
        request: LoginRequest 对象

    返回：
        TokenResponse: JWT令牌和用户信息

    可能的错误：
        - 401: 用户名或密码错误
        - 500: 服务器错误
    """
    try:
        # 实例化用户服务
        user_service = UserService()

        # 调用服务层登录验证
        # login 方法会：
        # 1. 查找用户
        # 2. 验证密码
        # 3. 生成JWT令牌
        # 4. 返回令牌和用户信息
        login_result = user_service.login(
            username=request.username,
            password=request.password
        )

        # 返回JWT令牌
        # 前端需要保存这个 token 并在后续请求中使用
        return TokenResponse(
            access_token=login_result["access_token"],  # JWT令牌字符串
            token_type="bearer",  # 令牌类型（RFC标准）
            user_id=login_result["user_id"],  # 用户ID
            username=login_result["username"]  # 用户名
        )
    except Exception as e:
        # 登录失败返回401错误（未授权）
        # 注意：使用401而不是400，因为是认证失败
        raise HTTPException(status_code=401, detail=f"登录失败: {str(e)}")


# ========== API 端点 3: 获取用户资料 ==========
# HTTP GET /api/user/{user_id}
@router.get("/{user_id}", response_model=UserProfileResponse)
# response_model=UserProfileResponse 返回用户完整资料
async def get_user_profile(user_id: int):
    """
    获取用户资料接口

    功能说明：
    - 根据用户ID查询用户详细信息
    - 返回用户名、邮箱等资料
    - 不返回密码等敏感信息

    使用场景：
    - 用户查看自己的个人资料
    - 管理员查看用户信息
    - 用户资料页面初始化

    安全考虑：
    - 应该验证请求者是否有权限查看该用户资料
    - 通常用户只能查看自己的资料
    - 建议添加JWT令牌验证中间件

    请求示例：
    GET /api/user/123

    返回示例（成功）：
    {
        "user_id": 123,
        "username": "张三",
        "email": "zhangsan@example.com"
    }

    返回示例（失败）：
    {
        "detail": "用户不存在"
    }

    参数：
        user_id: 路径参数，用户ID

    返回：
        UserProfileResponse: 用户完整资料

    可能的错误：
        - 404: 用户不存在
        - 400: 参数错误或数据库错误
    """
    try:
        # 查询用户信息
        # get_user 方法根据ID查询用户
        user = UserService().get_user(user_id)

        # 检查用户是否存在
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        # 返回用户资料
        # 注意：不返回密码字段
        return UserProfileResponse(
            user_id=user.id,  # 用户ID
            username=user.username,  # 用户名
            email=user.email  # 邮箱
        )
    except HTTPException:
        # 直接抛出HTTP异常（如404）
        raise
    except Exception as e:
        # 其他异常返回400错误
        raise HTTPException(status_code=400, detail=f"获取用户资料失败: {str(e)}")


# ========== API 端点 4: 更新用户资料 =========-
# HTTP PUT /api/user/{user_id}
@router.put("/{user_id}", response_model=UserProfileResponse)
# response_model=UserProfileResponse 返回更新后的用户资料
async def update_user_profile(user_id: int, request: UserUpdateRequest):
    """
    更新用户资料接口

    功能说明：
    - 修改用户名、邮箱或密码
    - 支持部分更新（只传需要修改的字段）
    - 返回更新后的用户信息

    更新流程：
    1. 接收用户ID和需要更新的字段
    2. 如果更新用户名，检查新用户名是否已存在
    3. 如果更新邮箱，检查新邮箱是否已被使用
    4. 如果更新密码，对新密码进行哈希加密
    5. 更新数据库记录
    6. 返回更新后的用户信息

    安全机制：
    - 应该验证请求者的身份（只能修改自己的资料）
    - 密码更新需要重新哈希
    - 邮箱更新需要验证（可选）

    使用场景：
    - 用户修改个人资料
    - 修改用户名
    - 更换邮箱
    - 重置密码

    请求示例1（修改用户名）：
    PUT /api/user/123
    Content-Type: application/json

    {
        "username": "张三新名字"
    }

    请求示例2（修改密码）：
    PUT /api/user/123
    {
        "password": "newpassword456"
    }

    请求示例3（同时修改多个）：
    PUT /api/user/123
    {
        "username": "张三",
        "email": "newemail@example.com",
        "password": "newpass"
    }

    返回示例：
    {
        "user_id": 123,
        "username": "张三新名字",
        "email": "zhangsan@example.com"
    }

    参数：
        user_id: 路径参数，要更新的用户ID
        request: UserUpdateRequest 对象，包含要更新的字段

    返回：
        UserProfileResponse: 更新后的用户资料

    可能的错误：
        - 400: 用户名已存在、邮箱已注册、参数错误
        - 404: 用户不存在
    """
    try:
        # 调用服务层更新用户
        # update_user 方法会：
        # 1. 查找用户是否存在
        # 2. 检查用户名/邮箱唯一性
        # 3. 如果更新密码，进行哈希加密
        # 4. 更新指定字段到数据库
        # 5. 返回更新后的用户对象
        user = UserService().update_user(
            user_id=user_id,  # 要更新的用户ID
            username=request.username,  # 新用户名（可选）
            email=request.email,  # 新邮箱（可选）
            password=request.password,  # 新密码（可选）
        )

        # 返回更新后的用户资料
        return UserProfileResponse(
            user_id=user.id,  # 用户ID
            username=user.username,  # 更新后的用户名
            email=user.email  # 更新后的邮箱
        )
    except Exception as e:
        # 更新失败返回400错误
        raise HTTPException(status_code=400, detail=f"更新用户资料失败: {str(e)}")