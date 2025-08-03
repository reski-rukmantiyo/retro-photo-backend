"""Comprehensive tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from photo_restore.utils.config import Config, ProcessingConfig, ModelConfig, LoggingConfig


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""
    
    def test_default_values(self):
        """Test default processing configuration values."""
        config = ProcessingConfig()
        
        assert config.max_image_size == 4096
        assert config.memory_limit_gb == 4.0
        assert config.temp_cleanup is True
        assert config.supported_formats == ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp']
        assert config.tile_size == 512
        assert config.tile_overlap == 32
    
    def test_custom_values(self):
        """Test custom processing configuration values."""
        config = ProcessingConfig(
            max_image_size=2048,
            memory_limit_gb=2.0,
            temp_cleanup=False,
            supported_formats=['jpg', 'png'],
            tile_size=256,
            tile_overlap=16
        )
        
        assert config.max_image_size == 2048
        assert config.memory_limit_gb == 2.0
        assert config.temp_cleanup is False
        assert config.supported_formats == ['jpg', 'png']
        assert config.tile_size == 256
        assert config.tile_overlap == 16


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default model configuration values."""
        config = ModelConfig()
        
        assert config.cache_dir == "~/.photo-restore/models"
        assert config.download_timeout == 300
        assert config.esrgan_model == "RealESRGAN_x4plus"
        assert config.gfpgan_model == "GFPGANv1.3"
        assert config.face_detection_threshold == 0.5
    
    def test_custom_values(self):
        """Test custom model configuration values."""
        config = ModelConfig(
            cache_dir="/custom/models",
            download_timeout=600,
            esrgan_model="CustomESRGAN",
            gfpgan_model="CustomGFPGAN",
            face_detection_threshold=0.8
        )
        
        assert config.cache_dir == "/custom/models"
        assert config.download_timeout == 600
        assert config.esrgan_model == "CustomESRGAN"
        assert config.gfpgan_model == "CustomGFPGAN"
        assert config.face_detection_threshold == 0.8


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""
    
    def test_default_values(self):
        """Test default logging configuration values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "detailed"
        assert config.file is None
        assert config.max_size_mb == 10
    
    def test_custom_values(self):
        """Test custom logging configuration values."""
        config = LoggingConfig(
            level="DEBUG",
            format="simple",
            file="/var/log/app.log",
            max_size_mb=50
        )
        
        assert config.level == "DEBUG"
        assert config.format == "simple"
        assert config.file == "/var/log/app.log"
        assert config.max_size_mb == 50


class TestConfig:
    """Test main Config class."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = Config()
        
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.logging, LoggingConfig)
        
        # Test default values are properly set
        assert config.processing.max_image_size == 4096
        assert config.models.cache_dir == "~/.photo-restore/models"
        assert config.logging.level == "INFO"
    
    def test_config_from_dict_complete(self):
        """Test configuration from complete dictionary."""
        config_dict = {
            'processing': {
                'max_image_size': 2048,
                'memory_limit_gb': 2.0,
                'temp_cleanup': False,
                'supported_formats': ['jpg', 'png'],
                'tile_size': 256,
                'tile_overlap': 16
            },
            'models': {
                'cache_dir': '/custom/path',
                'download_timeout': 600,
                'esrgan_model': 'CustomESRGAN',
                'gfpgan_model': 'CustomGFPGAN',
                'face_detection_threshold': 0.8
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'simple',
                'file': '/tmp/test.log',
                'max_size_mb': 20
            }
        }
        
        config = Config(config_dict)
        
        # Test processing config
        assert config.processing.max_image_size == 2048
        assert config.processing.memory_limit_gb == 2.0
        assert config.processing.temp_cleanup is False
        assert config.processing.supported_formats == ['jpg', 'png']
        assert config.processing.tile_size == 256
        assert config.processing.tile_overlap == 16
        
        # Test models config
        assert config.models.cache_dir == '/custom/path'
        assert config.models.download_timeout == 600
        assert config.models.esrgan_model == 'CustomESRGAN'
        assert config.models.gfpgan_model == 'CustomGFPGAN'
        assert config.models.face_detection_threshold == 0.8
        
        # Test logging config
        assert config.logging.level == 'DEBUG'
        assert config.logging.format == 'simple'
        assert config.logging.file == '/tmp/test.log'
        assert config.logging.max_size_mb == 20
    
    def test_config_from_partial_dict(self):
        """Test configuration from partial dictionary."""
        config_dict = {
            'processing': {
                'max_image_size': 1024
            },
            'models': {
                'download_timeout': 120
            }
            # Note: no logging section
        }
        
        config = Config(config_dict)
        
        # Test overridden values
        assert config.processing.max_image_size == 1024
        assert config.models.download_timeout == 120
        
        # Test default values remain
        assert config.processing.memory_limit_gb == 4.0  # default
        assert config.models.cache_dir == "~/.photo-restore/models"  # default
        assert config.logging.level == "INFO"  # default
    
    def test_config_with_invalid_keys(self):
        """Test configuration ignores invalid keys."""
        config_dict = {
            'processing': {
                'max_image_size': 2048,
                'invalid_key': 'should_be_ignored'
            },
            'models': {
                'cache_dir': '/test',
                'another_invalid_key': 123
            },
            'invalid_section': {
                'key': 'value'
            }
        }
        
        config = Config(config_dict)
        
        # Valid keys should be set
        assert config.processing.max_image_size == 2048
        assert config.models.cache_dir == '/test'
        
        # Invalid keys should not create new attributes
        assert not hasattr(config.processing, 'invalid_key')
        assert not hasattr(config.models, 'another_invalid_key')
        assert not hasattr(config, 'invalid_section')
    
    @pytest.mark.parametrize("quality,expected", [
        ('fast', {'upscale': 2, 'tile_size': 256, 'memory_usage': 'low'}),
        ('balanced', {'upscale': 4, 'tile_size': 512, 'memory_usage': 'medium'}),
        ('best', {'upscale': 4, 'tile_size': 1024, 'memory_usage': 'high'})
    ])
    def test_quality_settings_valid(self, quality, expected):
        """Test valid quality settings retrieval."""
        config = Config()
        settings = config.get_quality_settings(quality)
        
        for key, value in expected.items():
            assert settings[key] == value
    
    def test_quality_settings_invalid(self):
        """Test invalid quality setting returns balanced."""
        config = Config()
        
        invalid_settings = config.get_quality_settings('invalid')
        balanced_settings = config.get_quality_settings('balanced')
        
        assert invalid_settings == balanced_settings
    
    def test_config_save_load_roundtrip(self, temp_dir):
        """Test configuration save and load roundtrip."""
        config_path = temp_dir / 'test_config.yaml'
        
        # Create config with custom values
        original_config = Config()
        original_config.processing.max_image_size = 8192
        original_config.processing.memory_limit_gb = 8.0
        original_config.models.download_timeout = 120
        original_config.models.cache_dir = '/custom/models'
        original_config.logging.level = 'DEBUG'
        original_config.logging.file = '/tmp/debug.log'
        
        # Save config
        original_config.save(str(config_path))
        assert config_path.exists()
        
        # Load config
        loaded_config = Config.load(str(config_path))
        
        # Verify all values match
        assert loaded_config.processing.max_image_size == 8192
        assert loaded_config.processing.memory_limit_gb == 8.0
        assert loaded_config.models.download_timeout == 120
        assert loaded_config.models.cache_dir == '/custom/models'
        assert loaded_config.logging.level == 'DEBUG'
        assert loaded_config.logging.file == '/tmp/debug.log'
    
    def test_config_save_creates_directory(self, temp_dir):
        """Test config save creates parent directories."""
        nested_path = temp_dir / 'nested' / 'config' / 'test.yaml'
        
        config = Config()
        config.save(str(nested_path))
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    def test_config_load_nonexistent_file(self):
        """Test loading nonexistent config returns defaults."""
        config = Config.load('/nonexistent/config.yaml')
        
        # Should return default config
        assert config.processing.max_image_size == 4096
        assert config.models.cache_dir == "~/.photo-restore/models"
        assert config.logging.level == "INFO"
    
    def test_config_load_default_user_config(self, temp_dir):
        """Test loading default user config from home directory."""
        # Mock home directory
        mock_home = temp_dir / 'home'
        config_dir = mock_home / '.photo-restore'
        config_dir.mkdir(parents=True)
        
        config_file = config_dir / 'config.yaml'
        config_data = {
            'processing': {'max_image_size': 1024},
            'models': {'download_timeout': 60}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch('pathlib.Path.home', return_value=mock_home):
            config = Config.load()
            
            assert config.processing.max_image_size == 1024
            assert config.models.download_timeout == 60
    
    def test_config_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML returns defaults."""
        config_path = temp_dir / 'invalid.yaml'
        config_path.write_text('invalid: yaml: content: [unclosed')
        
        # Should not raise exception, should return defaults
        config = Config.load(str(config_path))
        assert config.processing.max_image_size == 4096
    
    def test_config_load_empty_file(self, temp_dir):
        """Test loading empty config file."""
        config_path = temp_dir / 'empty.yaml'
        config_path.write_text('')
        
        config = Config.load(str(config_path))
        
        # Should return defaults for empty file
        assert config.processing.max_image_size == 4096
        assert config.models.cache_dir == "~/.photo-restore/models"
    
    def test_config_save_yaml_format(self, temp_dir):
        """Test saved config has correct YAML format."""
        config_path = temp_dir / 'format_test.yaml'
        
        config = Config()
        config.processing.max_image_size = 2048
        config.models.download_timeout = 300
        config.save(str(config_path))
        
        # Load and verify YAML structure
        with open(config_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert 'processing' in saved_data
        assert 'models' in saved_data
        assert 'logging' in saved_data
        assert saved_data['processing']['max_image_size'] == 2048
        assert saved_data['models']['download_timeout'] == 300
    
    def test_config_update_from_dict_edge_cases(self):
        """Test _update_from_dict with edge cases."""
        config = Config()
        
        # Empty dict
        config._update_from_dict({})
        assert config.processing.max_image_size == 4096  # unchanged
        
        # None values
        config._update_from_dict({
            'processing': None,
            'models': {'cache_dir': None}
        })
        # Should handle gracefully without errors
        assert config.processing.max_image_size == 4096
    
    def test_quality_settings_returns_copy(self):
        """Test quality settings returns independent copy."""
        config = Config()
        
        settings1 = config.get_quality_settings('fast')
        settings2 = config.get_quality_settings('fast')
        
        # Modify one copy
        settings1['custom_key'] = 'custom_value'
        
        # Other copy should be unaffected
        assert 'custom_key' not in settings2