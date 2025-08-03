"""Configuration management for photo restoration."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ProcessingConfig:
    """Image processing configuration."""
    max_image_size: int = 4096
    memory_limit_gb: float = 4.0
    temp_cleanup: bool = True
    supported_formats: list = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'])
    tile_size: int = 512
    tile_overlap: int = 32


@dataclass
class ModelConfig:
    """AI model configuration."""
    cache_dir: str = "~/.photo-restore/models"
    download_timeout: int = 300
    esrgan_model: str = "RealESRGAN_x4plus"
    gfpgan_model: str = "GFPGANv1.3"
    face_detection_threshold: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "detailed"
    file: Optional[str] = None
    max_size_mb: int = 10


class Config:
    """Main configuration class."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration."""
        self.processing = ProcessingConfig()
        self.models = ModelConfig()
        self.logging = LoggingConfig()
        
        if config_dict:
            self._update_from_dict(config_dict)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'processing' in config_dict:
            for key, value in config_dict['processing'].items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
        
        if 'models' in config_dict:
            for key, value in config_dict['models'].items():
                if hasattr(self.models, key):
                    setattr(self.models, key, value)
        
        if 'logging' in config_dict:
            for key, value in config_dict['logging'].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file or defaults."""
        config_dict = {}
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        
        # Load from default user config
        elif not config_path:
            default_config_path = Path.home() / '.photo-restore' / 'config.yaml'
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    config_dict = yaml.safe_load(f) or {}
        
        return cls(config_dict)
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_dict = {
            'processing': {
                'max_image_size': self.processing.max_image_size,
                'memory_limit_gb': self.processing.memory_limit_gb,
                'temp_cleanup': self.processing.temp_cleanup,
                'supported_formats': self.processing.supported_formats,
                'tile_size': self.processing.tile_size,
                'tile_overlap': self.processing.tile_overlap
            },
            'models': {
                'cache_dir': self.models.cache_dir,
                'download_timeout': self.models.download_timeout,
                'esrgan_model': self.models.esrgan_model,
                'gfpgan_model': self.models.gfpgan_model,
                'face_detection_threshold': self.models.face_detection_threshold
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file,
                'max_size_mb': self.logging.max_size_mb
            }
        }
        
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_quality_settings(self, quality: str) -> Dict[str, Any]:
        """Get processing settings for quality level."""
        quality_settings = {
            'fast': {
                'upscale': 2,
                'tile_size': 256,
                'memory_usage': 'low'
            },
            'balanced': {
                'upscale': 4,
                'tile_size': 512,
                'memory_usage': 'medium'
            },
            'best': {
                'upscale': 4,
                'tile_size': 1024,
                'memory_usage': 'high'
            }
        }
        return quality_settings.get(quality, quality_settings['balanced'])