"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path

from photo_restore.utils.config import Config, ProcessingConfig, ModelConfig, LoggingConfig


class TestConfig:
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = Config()
        
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.logging, LoggingConfig)
        
        assert config.processing.max_image_size == 4096
        assert config.models.cache_dir == "~/.photo-restore/models"
        assert config.logging.level == "INFO"
    
    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        config_dict = {
            'processing': {
                'max_image_size': 2048,
                'memory_limit_gb': 2.0
            },
            'models': {
                'cache_dir': '/custom/path',
                'download_timeout': 600
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'simple'
            }
        }
        
        config = Config(config_dict)
        
        assert config.processing.max_image_size == 2048
        assert config.processing.memory_limit_gb == 2.0
        assert config.models.cache_dir == '/custom/path'
        assert config.models.download_timeout == 600
        assert config.logging.level == 'DEBUG'
        assert config.logging.format == 'simple'
    
    def test_quality_settings(self):
        """Test quality settings retrieval."""
        config = Config()
        
        fast_settings = config.get_quality_settings('fast')
        assert fast_settings['upscale'] == 2
        assert fast_settings['tile_size'] == 256
        assert fast_settings['memory_usage'] == 'low'
        
        balanced_settings = config.get_quality_settings('balanced')
        assert balanced_settings['upscale'] == 4
        assert balanced_settings['tile_size'] == 512
        assert balanced_settings['memory_usage'] == 'medium'
        
        best_settings = config.get_quality_settings('best')
        assert best_settings['upscale'] == 4
        assert best_settings['tile_size'] == 1024
        assert best_settings['memory_usage'] == 'high'
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.yaml'
            
            # Create and save config
            original_config = Config()
            original_config.processing.max_image_size = 8192
            original_config.models.download_timeout = 120
            original_config.save(str(config_path))
            
            # Load config
            loaded_config = Config.load(str(config_path))
            
            assert loaded_config.processing.max_image_size == 8192
            assert loaded_config.models.download_timeout == 120
    
    def test_invalid_quality_setting(self):
        """Test invalid quality setting returns balanced."""
        config = Config()
        invalid_settings = config.get_quality_settings('invalid')
        balanced_settings = config.get_quality_settings('balanced')
        
        assert invalid_settings == balanced_settings