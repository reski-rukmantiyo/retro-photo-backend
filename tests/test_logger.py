"""Comprehensive tests for logging utilities."""

import pytest
import logging
import time
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

from photo_restore.utils.logger import (
    ColoredFormatter,
    PerformanceLogger,
    setup_logger,
    get_logger,
    log_system_info,
    log_model_info
)


class TestColoredFormatter:
    """Test colored formatter for console output."""
    
    def test_color_codes_defined(self):
        """Test that color codes are properly defined."""
        formatter = ColoredFormatter()
        
        expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for level in expected_levels:
            assert level in formatter.COLORS
            assert formatter.COLORS[level].startswith('\033[')
        
        assert formatter.RESET == '\033[0m'
    
    def test_format_with_colors(self):
        """Test formatting with colors."""
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should contain color codes and reset
        assert '\033[32m' in formatted  # Green for INFO
        assert '\033[0m' in formatted   # Reset
        assert 'Test message' in formatted
    
    def test_format_different_levels(self):
        """Test formatting different log levels."""
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        
        levels_colors = [
            (logging.DEBUG, '\033[36m'),    # Cyan
            (logging.INFO, '\033[32m'),     # Green
            (logging.WARNING, '\033[33m'),  # Yellow
            (logging.ERROR, '\033[31m'),    # Red
            (logging.CRITICAL, '\033[35m')  # Magenta
        ]
        
        for level, expected_color in levels_colors:
            record = logging.LogRecord(
                name='test',
                level=level,
                pathname='',
                lineno=0,
                msg='Test message',
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            assert expected_color in formatted
            assert '\033[0m' in formatted
    
    def test_format_unknown_level(self):
        """Test formatting unknown log level."""
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        
        # Create record with custom level
        record = logging.LogRecord(
            name='test',
            level=55,  # Custom level
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.levelname = 'CUSTOM'
        
        formatted = formatter.format(record)
        
        # Should still format without color
        assert 'CUSTOM' in formatted
        assert 'Test message' in formatted


class TestPerformanceLogger:
    """Test performance logging context manager."""
    
    def test_performance_logger_success(self):
        """Test performance logger with successful operation."""
        mock_logger = MagicMock()
        
        with PerformanceLogger(mock_logger, "test operation"):
            time.sleep(0.1)  # Simulate work
        
        # Should log start and completion
        assert mock_logger.debug.called
        assert mock_logger.info.called
        
        # Check debug call for start
        debug_call = mock_logger.debug.call_args[0][0]
        assert "Starting test operation" in debug_call
        
        # Check info call for completion
        info_call = mock_logger.info.call_args[0][0]
        assert "Completed test operation" in info_call
        assert "in" in info_call and "s" in info_call  # Duration
    
    def test_performance_logger_with_exception(self):
        """Test performance logger with exception."""
        mock_logger = MagicMock()
        
        with pytest.raises(ValueError):
            with PerformanceLogger(mock_logger, "failing operation"):
                time.sleep(0.05)
                raise ValueError("Test error")
        
        # Should log start and failure
        assert mock_logger.debug.called
        assert mock_logger.error.called
        
        # Check error call
        error_call = mock_logger.error.call_args[0][0]
        assert "Failed failing operation" in error_call
        assert "Test error" in error_call
    
    def test_performance_logger_timing_accuracy(self):
        """Test performance logger timing accuracy."""
        mock_logger = MagicMock()
        
        start_time = time.time()
        with PerformanceLogger(mock_logger, "timed operation"):
            time.sleep(0.2)
        end_time = time.time()
        
        # Get the logged duration
        info_call = mock_logger.info.call_args[0][0]
        # Extract duration from message like "Completed operation in 0.20s"
        duration_str = info_call.split(" in ")[1].split("s")[0]
        logged_duration = float(duration_str)
        actual_duration = end_time - start_time
        
        # Should be reasonably close (within 0.1s tolerance)
        assert abs(logged_duration - actual_duration) < 0.1


class TestSetupLogger:
    """Test logger setup functionality."""
    
    def test_default_logger_setup(self):
        """Test default logger configuration."""
        logger = setup_logger("test_default")
        
        assert logger.name == "test_default"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only
        
        # Test logging
        with patch('sys.stdout', new=StringIO()) as fake_out:
            logger.info("Test message")
            output = fake_out.getvalue()
            assert "Test message" in output
    
    @pytest.mark.parametrize("level,expected", [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL)
    ])
    def test_logger_levels(self, level, expected):
        """Test different logging levels."""
        logger = setup_logger("test_level", level=level)
        assert logger.level == expected
    
    def test_logger_with_file_output(self, temp_dir):
        """Test logger with file output."""
        log_file = temp_dir / "test.log"
        logger = setup_logger("test_file", log_file=str(log_file))
        
        assert len(logger.handlers) == 2  # Console + file
        
        # Test logging to file
        logger.info("Test file message")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test file message" in content
    
    def test_logger_file_creates_directory(self, temp_dir):
        """Test logger creates log file directory."""
        log_file = temp_dir / "logs" / "subdir" / "test.log"
        logger = setup_logger("test_dir", log_file=str(log_file))
        
        logger.info("Test message")
        
        assert log_file.exists()
        assert log_file.parent.exists()
    
    @pytest.mark.parametrize("format_style,expected_parts", [
        ("simple", ["%(levelname)s", "%(message)s"]),
        ("detailed", ["%(asctime)s", "%(name)s", "%(levelname)s", "%(message)s"]),
        ("json", ["timestamp", "logger", "level", "message"])
    ])
    def test_logger_formats(self, format_style, expected_parts):
        """Test different logging formats."""
        logger = setup_logger("test_format", format_style=format_style)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            logger.info("Test message")
            output = fake_out.getvalue()
            
            if format_style == "json":
                # For JSON format, parse and check structure
                try:
                    log_data = json.loads(output.strip())
                    for part in expected_parts:
                        assert part in log_data
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON format: {output}")
            else:
                # For other formats, check presence of format parts
                assert "Test message" in output
    
    def test_logger_file_rotation_config(self, temp_dir):
        """Test file rotation configuration."""
        log_file = temp_dir / "rotation.log"
        logger = setup_logger(
            "test_rotation",
            log_file=str(log_file),
            max_size_mb=1,
            backup_count=3
        )
        
        # Find the rotating file handler
        file_handler = None
        for handler in logger.handlers:
            if hasattr(handler, 'maxBytes'):
                file_handler = handler
                break
        
        assert file_handler is not None
        assert file_handler.maxBytes == 1024 * 1024  # 1MB
        assert file_handler.backupCount == 3
    
    def test_logger_clears_existing_handlers(self):
        """Test that setup_logger clears existing handlers."""
        # Create logger with initial handler
        logger = logging.getLogger("test_clear")
        initial_handler = logging.StreamHandler()
        logger.addHandler(initial_handler)
        
        # Setup logger should clear existing handlers
        logger = setup_logger("test_clear")
        
        # Should not contain the initial handler
        assert initial_handler not in logger.handlers
    
    def test_color_detection_tty(self):
        """Test color detection for TTY."""
        with patch('sys.stdout.isatty', return_value=True):
            logger = setup_logger("test_color")
            
            # Should use colored formatter
            console_handler = logger.handlers[0]
            assert isinstance(console_handler.formatter, ColoredFormatter)
    
    def test_color_detection_non_tty(self):
        """Test color detection for non-TTY."""
        with patch('sys.stdout.isatty', return_value=False):
            logger = setup_logger("test_no_color")
            
            # Should use regular formatter
            console_handler = logger.handlers[0]
            assert not isinstance(console_handler.formatter, ColoredFormatter)


class TestGetLogger:
    """Test get_logger function."""
    
    def test_get_logger_default(self):
        """Test getting logger with default name."""
        logger = get_logger()
        assert logger.name == "photo_restore"
    
    def test_get_logger_custom_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("custom_logger")
        assert logger.name == "custom_logger"
    
    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns same instance for same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2


class TestLogSystemInfo:
    """Test system information logging."""
    
    @patch('photo_restore.utils.logger.platform')
    @patch('photo_restore.utils.logger.psutil')
    def test_log_system_info(self, mock_psutil, mock_platform):
        """Test logging system information."""
        # Mock system info
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.4.0"
        mock_platform.python_version.return_value = "3.9.0"
        
        mock_memory = MagicMock()
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_logger = MagicMock()
        
        log_system_info(mock_logger)
        
        # Verify all info calls
        assert mock_logger.info.call_count == 4
        
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("System: Linux 5.4.0" in call for call in calls)
        assert any("Python: 3.9.0" in call for call in calls)
        assert any("CPU cores: 8" in call for call in calls)
        assert any("Memory: 8 GB" in call for call in calls)
    
    @patch('photo_restore.utils.logger.platform')
    @patch('photo_restore.utils.logger.psutil')
    def test_log_system_info_error_handling(self, mock_psutil, mock_platform):
        """Test system info logging with errors."""
        # Mock errors
        mock_platform.system.side_effect = Exception("Platform error")
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = MagicMock(total=4*1024**3)
        
        mock_logger = MagicMock()
        
        # Should not raise exception
        log_system_info(mock_logger)
        
        # Should still call some info methods
        assert mock_logger.info.called


class TestLogModelInfo:
    """Test model information logging."""
    
    def test_log_model_info_basic(self):
        """Test basic model info logging."""
        mock_logger = MagicMock()
        
        log_model_info(mock_logger, "TestModel", 1048576)  # 1MB
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "TestModel" in call_args
        assert "1.0 MB" in call_args
    
    @pytest.mark.parametrize("size,expected_mb", [
        (1024 * 1024, "1.0 MB"),
        (2.5 * 1024 * 1024, "2.5 MB"),
        (512 * 1024, "0.5 MB"),
        (10 * 1024 * 1024, "10.0 MB")
    ])
    def test_log_model_info_sizes(self, size, expected_mb):
        """Test model info with different sizes."""
        mock_logger = MagicMock()
        
        log_model_info(mock_logger, "Model", size)
        
        call_args = mock_logger.info.call_args[0][0]
        assert expected_mb in call_args


class TestLoggerIntegration:
    """Test logger integration scenarios."""
    
    def test_multiple_loggers_isolation(self):
        """Test that multiple loggers are isolated."""
        logger1 = setup_logger("logger1", level="DEBUG")
        logger2 = setup_logger("logger2", level="ERROR")
        
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR
        
        # Should be different instances
        assert logger1 is not logger2
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy behavior."""
        parent_logger = setup_logger("parent", level="INFO")
        child_logger = get_logger("parent.child")
        
        # Child should inherit parent's level if not explicitly set
        assert child_logger.name == "parent.child"
    
    def test_performance_logger_integration(self, temp_dir):
        """Test PerformanceLogger with real logger."""
        log_file = temp_dir / "perf.log"
        logger = setup_logger("perf_test", log_file=str(log_file), level="DEBUG")
        
        with PerformanceLogger(logger, "integration test"):
            time.sleep(0.1)
        
        # Check log file content
        log_content = log_file.read_text()
        assert "Starting integration test" in log_content
        assert "Completed integration test" in log_content
    
    def test_logger_with_exception_info(self, temp_dir):
        """Test logger with exception information."""
        log_file = temp_dir / "exception.log"
        logger = setup_logger("exception_test", log_file=str(log_file))
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")
        
        log_content = log_file.read_text()
        assert "An error occurred" in log_content
        assert "ValueError: Test exception" in log_content
        assert "Traceback" in log_content