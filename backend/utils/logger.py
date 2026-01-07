"""
Logging utilities for the fall detection system.
Provides structured logging with rotation and multiple handlers.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger


def setup_logger(
    name: str = "fall_detection",
    log_file: Path = Path("./logs/fall_detection.log"),
    log_level: str = "INFO",
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to use JSON formatting (better for production)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if json_format:
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
            rename_fields={"levelname": "severity", "pathname": "file"}
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class LoggerMixin:
    """Mixin class to add logger to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for the class."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger


# Performance logging decorator
def log_performance(logger: logging.Logger = None):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_performance()
        def my_function():
            pass
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            
            log = logger or logging.getLogger(func.__module__)
            log.debug(f"{func.__name__} took {elapsed:.2f}ms")
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            
            log = logger or logging.getLogger(func.__module__)
            log.debug(f"{func.__name__} took {elapsed:.2f}ms")
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
