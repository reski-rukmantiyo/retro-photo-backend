"""Pytest configuration and shared fixtures."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from PIL import Image
import cv2

from photo_restore.utils.config import Config
from photo_restore.utils.logger import setup_logger


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create test data directory for the session."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config() -> Config:
    """Create sample configuration for testing."""
    config_dict = {
        'processing': {
            'max_image_size': 2048,
            'memory_limit_gb': 2.0,
            'temp_cleanup': True,
            'supported_formats': ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
            'tile_size': 512,
            'tile_overlap': 32
        },
        'models': {
            'cache_dir': '/tmp/test_models',
            'download_timeout': 60,
            'esrgan_model': 'RealESRGAN_x4plus',
            'gfpgan_model': 'GFPGANv1.3',
            'face_detection_threshold': 0.5
        },
        'logging': {
            'level': 'DEBUG',
            'format': 'detailed',
            'file': None,
            'max_size_mb': 10
        }
    }
    return Config(config_dict)


@pytest.fixture
def test_logger():
    """Create test logger."""
    return setup_logger("test_logger", level="DEBUG")


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """Create sample RGB image array."""
    # Create a 256x256 RGB image with gradient pattern
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add gradient patterns
    for i in range(256):
        for j in range(256):
            image[i, j, 0] = min(255, i)  # Red gradient
            image[i, j, 1] = min(255, j)  # Green gradient
            image[i, j, 2] = min(255, (i + j) // 2)  # Blue gradient
    
    return image


@pytest.fixture
def sample_image_bgr(sample_image_rgb) -> np.ndarray:
    """Create sample BGR image array (OpenCV format)."""
    return cv2.cvtColor(sample_image_rgb, cv2.COLOR_RGB2BGR)


@pytest.fixture
def sample_image_file(temp_dir: Path, sample_image_rgb: np.ndarray) -> Path:
    """Create sample image file on disk."""
    image_path = temp_dir / "test_image.jpg"
    
    # Convert to PIL and save
    pil_image = Image.fromarray(sample_image_rgb)
    pil_image.save(image_path, "JPEG", quality=90)
    
    return image_path


@pytest.fixture
def sample_image_files(temp_dir: Path, sample_image_rgb: np.ndarray) -> Dict[str, Path]:
    """Create multiple sample image files with different formats."""
    files = {}
    
    # Create images in different formats
    pil_image = Image.fromarray(sample_image_rgb)
    
    formats = {
        'jpg': ('JPEG', {'quality': 90}),
        'png': ('PNG', {}),
        'tiff': ('TIFF', {}),
        'bmp': ('BMP', {})
    }
    
    for ext, (format_name, kwargs) in formats.items():
        file_path = temp_dir / f"test_image.{ext}"
        pil_image.save(file_path, format_name, **kwargs)
        files[ext] = file_path
    
    return files


@pytest.fixture
def large_image_file(temp_dir: Path) -> Path:
    """Create large test image file."""
    image_path = temp_dir / "large_image.jpg"
    
    # Create 2048x2048 image
    large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    pil_image = Image.fromarray(large_image)
    pil_image.save(image_path, "JPEG", quality=90)
    
    return image_path


@pytest.fixture
def corrupted_image_file(temp_dir: Path) -> Path:
    """Create corrupted image file for error testing."""
    corrupted_path = temp_dir / "corrupted.jpg"
    
    # Write invalid JPEG data
    with open(corrupted_path, 'wb') as f:
        f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF corrupted data')
    
    return corrupted_path


@pytest.fixture
def test_directory_structure(temp_dir: Path, sample_image_rgb: np.ndarray) -> Path:
    """Create test directory structure with images."""
    root_dir = temp_dir / "images"
    
    # Create directory structure
    subdirs = [
        root_dir,
        root_dir / "subdir1",
        root_dir / "subdir2",
        root_dir / "subdir1" / "nested"
    ]
    
    for subdir in subdirs:
        subdir.mkdir(parents=True, exist_ok=True)
    
    # Create images in each directory
    pil_image = Image.fromarray(sample_image_rgb)
    
    image_paths = [
        root_dir / "image1.jpg",
        root_dir / "image2.png",
        root_dir / "subdir1" / "image3.jpg",
        root_dir / "subdir1" / "nested" / "image4.png",
        root_dir / "subdir2" / "image5.jpg"
    ]
    
    for i, image_path in enumerate(image_paths):
        # Vary image content slightly
        modified_image = sample_image_rgb.copy()
        modified_image = np.roll(modified_image, i * 10, axis=0)  # Shift rows
        
        pil_modified = Image.fromarray(modified_image)
        format_name = "JPEG" if image_path.suffix == ".jpg" else "PNG"
        save_kwargs = {"quality": 90} if format_name == "JPEG" else {}
        pil_modified.save(image_path, format_name, **save_kwargs)
    
    return root_dir


@pytest.fixture
def mock_esrgan_model():
    """Mock Real-ESRGAN model for testing."""
    mock_model = MagicMock()
    
    def mock_enhance(image, outscale=4):
        # Return upscaled image with same content
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w * outscale, h * outscale), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    mock_model.enhance = mock_enhance
    return mock_model


@pytest.fixture
def mock_gfpgan_model():
    """Mock GFPGAN model for testing."""
    mock_model = MagicMock()
    
    def mock_enhance(image):
        # Return slightly modified image to simulate face enhancement
        enhanced = image.copy()
        # Add slight brightness increase to simulate enhancement
        enhanced = np.clip(enhanced.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        return enhanced
    
    mock_model.enhance = mock_enhance
    return mock_model


@pytest.fixture
def mock_model_manager(mock_esrgan_model, mock_gfpgan_model):
    """Mock ModelManager for testing."""
    with patch('photo_restore.models.model_manager.ModelManager') as mock_class:
        mock_instance = mock_class.return_value
        
        # Mock model loading methods
        mock_instance.load_esrgan_model.return_value = True
        mock_instance.load_gfpgan_model.return_value = True
        mock_instance.is_esrgan_loaded.return_value = True
        mock_instance.is_gfpgan_loaded.return_value = True
        mock_instance.get_esrgan_model.return_value = mock_esrgan_model
        mock_instance.get_gfpgan_model.return_value = mock_gfpgan_model
        
        # Mock download method
        mock_instance.download_model.return_value = True
        
        # Mock model info
        mock_instance.get_model_info.return_value = {
            'esrgan_loaded': True,
            'esrgan_scale': 4,
            'gfpgan_loaded': True,
            'cache_dir': '/tmp/test_models',
            'available_models': ['esrgan_x2', 'esrgan_x4', 'gfpgan']
        }
        
        yield mock_instance


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        'max_processing_time': 30.0,  # seconds
        'max_memory_usage': 2.0,      # GB
        'batch_size_limit': 10,       # images
        'image_size_limit': 4096      # pixels
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Auto-cleanup fixture to remove temporary files after each test."""
    yield
    # Cleanup any test files in /tmp that match our patterns
    import glob
    temp_patterns = [
        '/tmp/test_*',
        '/tmp/photo_restore_*'
    ]
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except OSError:
                pass  # Ignore cleanup errors


@pytest.fixture
def network_responses():
    """Mock network responses for model downloads."""
    responses = {
        'success': {
            'status_code': 200,
            'headers': {'content-length': '1048576'},  # 1MB
            'content': b'x' * 1048576  # Mock model data
        },
        'not_found': {
            'status_code': 404,
            'reason': 'Not Found'
        },
        'timeout': {
            'side_effect': TimeoutError("Connection timeout")
        }
    }
    return responses


# Parametrized fixtures for different test scenarios
@pytest.fixture(params=['fast', 'balanced', 'best'])
def quality_setting(request):
    """Parametrized quality settings."""
    return request.param


@pytest.fixture(params=[2, 4])
def upscale_factor(request):
    """Parametrized upscale factors."""
    return request.param


@pytest.fixture(params=['jpg', 'png'])
def output_format(request):
    """Parametrized output formats."""
    return request.param


@pytest.fixture(params=[True, False])
def face_enhance_setting(request):
    """Parametrized face enhancement settings."""
    return request.param