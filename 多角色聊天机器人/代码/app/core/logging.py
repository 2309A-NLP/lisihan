# -*- coding: utf-8 -*-
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any

LOG_DIR = "logs"
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "error.log")
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _file_handler(path: str, level: int, formatter: logging.Formatter) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler

def setup_logging():
    """配置日志：终端和文件同时输出，并显示文件、行号、函数和异常堆栈。"""
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("jieba").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = _file_handler(APP_LOG_FILE, logging.INFO, formatter)
    error_file_handler = _file_handler(ERROR_LOG_FILE, logging.ERROR, formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logging.captureWarnings(True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(console_handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.INFO)

    root_logger.info("日志系统初始化完成，日志文件: %s, 错误日志: %s", APP_LOG_FILE, ERROR_LOG_FILE)


def log_exception(logger: logging.Logger, message: str, exc: BaseException, *message_args: Any, **context: Any) -> None:
    """用 logging 打印完整 traceback，包含具体文件、行号和报错原因。"""
    if message_args:
        try:
            message = message % message_args
        except Exception:
            message = " ".join([message, *(repr(arg) for arg in message_args)])
    context_text = " ".join(f"{key}={value!r}" for key, value in context.items())
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "%s%s\n报错原因: %s: %s\nTraceback:\n%s",
        message,
        f" | {context_text}" if context_text else "",
        type(exc).__name__,
        exc,
        traceback_text,
    )
