# -*- coding: utf-8 -*-

"""
认证工具模块（用于保护需要登录的接口）
"""
import base64
import json

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

try:
    from jose import JWTError, jwt
except Exception:
    # 如果没有安装 jose，就降级使用 fallback token
    JWTError = Exception
    jwt = None

from app.models.user import User
from app.services.user_service import UserService
from config.config import settings

# OAuth2 Bearer Token 方案（用于从请求头中获取 token）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/user/login")


# ========= 工具函数 =========
def _decode_fallback_token(token: str) -> dict:
    """
    解码 fallback token（当 JWT 不可用时使用）

    fallback token 实际是：
    base64 编码的 JSON 字符串
    """
    try:
        # base64 解码
        payload = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        # 转成字典
        return json.loads(payload)
    except Exception as exc:
        # 解码失败 → token 无效
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ========= 核心认证函数 =========
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    从 token 中解析当前登录用户

    流程：
    1. 解析 token（JWT 或 fallback）
    2. 获取用户名（sub）
    3. 查询数据库用户
    4. 检查用户是否存在 & 是否启用
    """
    # 通用认证失败错误
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="登录凭证无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # ===== 1. 解析 token =====
    if jwt:
        # 使用 JWT 解码
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
        except JWTError as exc:
            raise credentials_error from exc
    else:
        # fallback 方式
        payload = _decode_fallback_token(token)

    # ===== 2. 获取用户名 =====
    username = payload.get("sub")
    if not username:
        raise credentials_error

    # ===== 3. 查询用户 =====
    user = UserService().get_user_by_username(username)
    if not user:
        raise credentials_error

    # ===== 4. 检查用户状态 =====
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用"
        )

    return user


# ========= 权限控制 =========
def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    校验当前用户是否为管理员

    用法：
    在接口中依赖该函数，即可限制只有管理员能访问
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )

    return current_user