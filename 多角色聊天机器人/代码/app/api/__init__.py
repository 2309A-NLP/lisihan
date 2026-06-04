# -*- coding: utf-8 -*-
"""API 路由统一注册。"""

from fastapi import APIRouter
from app.api import chat

api_router = APIRouter(prefix="/api")
api_router.include_router(chat.router)