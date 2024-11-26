import logging
import os
import sys
import threading
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING


class Logger:
    _lock = threading.Lock()
    _default_handler: Optional[logging.Handler] = None
    _file_handler: Optional[logging.Handler] = None

    @classmethod
    def _color_supported(cls) -> bool:
        """检测是否支持颜色输出"""
        if os.environ.get("NO_COLOR", None):
            return False
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    @classmethod
    def _create_formatter(cls, use_color: bool = False) -> logging.Formatter:
        """创建日志格式器"""
        header = "[%(levelname)1.1s %(asctime)s]"
        message = "%(message)s"
        if use_color and COLORLOG_AVAILABLE:
            return colorlog.ColoredFormatter(
                f"%(log_color)s{header}%(reset)s {message}",
                log_colors={
                    "DEBUG": "blue",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        return logging.Formatter(f"{header} {message}")

    @classmethod
    def _get_root_logger(cls) -> logging.Logger:
        """获取根日志记录器"""
        return logging.getLogger()

    @classmethod
    def _configure(cls) -> None:
        """配置根日志记录器"""
        with cls._lock:
            if cls._default_handler:  # 已配置
                return

            # 配置默认控制台处理器
            cls._default_handler = logging.StreamHandler(sys.stderr)
            cls._default_handler.setFormatter(cls._create_formatter(use_color=cls._color_supported()))
            root_logger = cls._get_root_logger()
            root_logger.addHandler(cls._default_handler)
            root_logger.setLevel(logging.INFO)
            root_logger.propagate = False

    @classmethod
    def set_level(cls, level: int) -> None:
        """设置日志输出级别"""
        cls._configure()
        cls._get_root_logger().setLevel(level)

    @classmethod
    def add_file_handler(cls, file_path: str, level: int = logging.INFO) -> None:
        """添加文件处理器"""
        cls._configure()
        if cls._file_handler:
            # 如果文件处理器已存在，移除旧的
            cls._get_root_logger().removeHandler(cls._file_handler)

        cls._file_handler = logging.FileHandler(file_path)
        cls._file_handler.setLevel(level)
        cls._file_handler.setFormatter(cls._create_formatter())
        cls._get_root_logger().addHandler(cls._file_handler)

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """获取指定名称的日志记录器"""
        cls._configure()
        return logging.getLogger(name)


# 短别名，方便外部使用
set_level = Logger.set_level
add_file_handler = Logger.add_file_handler
get_logger = Logger.get_logger
