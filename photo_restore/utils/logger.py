"""Logging configuration and utilities."""

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.2f}s: {exc_val}")


def setup_logger(
    name: str = "photo_restore",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_style: str = "detailed",
    max_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_style: Format style (simple, detailed, json)
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format strings
    formats = {
        'simple': '%(levelname)s: %(message)s',
        'detailed': '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        'json': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    }
    
    fmt = formats.get(format_style, formats['detailed'])
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    if sys.stdout.isatty():  # Only use colors for terminal output
        console_formatter = ColoredFormatter(fmt)
    else:
        console_formatter = logging.Formatter(fmt)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_formatter = logging.Formatter(fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "photo_restore") -> logging.Logger:
    """Get or create logger instance."""
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")


def log_model_info(logger: logging.Logger, model_name: str, model_size: int) -> None:
    """Log model information."""
    size_mb = model_size / (1024 * 1024)
    logger.info(f"Loaded model: {model_name} ({size_mb:.1f} MB)")