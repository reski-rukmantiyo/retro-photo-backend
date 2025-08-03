"""Comprehensive tests for model manager."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import requests

from photo_restore.models.model_manager import ModelManager
from photo_restore.utils.config import Config
from tests.mocks import MockRealESRGANer, MockGFPGANer, patch_realesrgan_import, patch_gfpgan_import


class TestModelManager:
    """Test ModelManager functionality."""
    
    @pytest.fixture
    def model_manager(self, sample_config, test_logger):
        """Create ModelManager instance for testing."""
        return ModelManager(sample_config, test_logger)
    
    @pytest.fixture
    def mock_requests_response(self):
        """Mock successful HTTP response."""
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.headers = {'content-length': '1048576'}  # 1MB
        response.iter_content.return_value = [b'x' * 8192] * 128  # 1MB of data
        return response
    
    def test_init_creates_cache_directory(self, sample_config, test_logger):
        """Test ModelManager creates cache directory."""
        manager = ModelManager(sample_config, test_logger)
        
        cache_dir = Path(sample_config.models.cache_dir).expanduser()
        assert manager.cache_dir == cache_dir
        # Directory should be created during init
        assert cache_dir.exists()
    
    def test_model_urls_defined(self, model_manager):
        """Test that model URLs are properly defined."""
        expected_models = ['esrgan_x2', 'esrgan_x4', 'gfpgan']
        
        for model_key in expected_models:
            assert model_key in ModelManager.MODEL_URLS
            model_info = ModelManager.MODEL_URLS[model_key]
            assert 'url' in model_info
            assert 'filename' in model_info
            assert 'scale' in model_info
    
    @patch('requests.get')
    def test_download_model_success(self, mock_get, model_manager, mock_requests_response):
        """Test successful model download."""
        mock_get.return_value = mock_requests_response
        
        success = model_manager.download_model('esrgan_x4')
        
        assert success is True
        mock_get.assert_called_once()
        
        # Check file was "created"
        expected_path = model_manager.cache_dir / 'RealESRGAN_x4plus.pth'
        assert expected_path.exists()
    
    @patch('requests.get')
    def test_download_model_already_cached(self, mock_get, model_manager):
        """Test download skips if model already cached."""
        # Pre-create model file
        model_path = model_manager.cache_dir / 'RealESRGAN_x4plus.pth'
        model_path.write_bytes(b'fake model data')
        
        success = model_manager.download_model('esrgan_x4')
        
        assert success is True
        mock_get.assert_not_called()  # Should not download
    
    @patch('requests.get')
    def test_download_model_force_download(self, mock_get, model_manager, mock_requests_response):
        """Test force download overwrites existing file."""
        # Pre-create model file
        model_path = model_manager.cache_dir / 'RealESRGAN_x4plus.pth'
        model_path.write_bytes(b'old model data')
        
        mock_get.return_value = mock_requests_response
        
        success = model_manager.download_model('esrgan_x4', force_download=True)
        
        assert success is True
        mock_get.assert_called_once()
    
    def test_download_model_invalid_key(self, model_manager):
        """Test download with invalid model key."""
        success = model_manager.download_model('invalid_model')
        assert success is False
    
    @patch('requests.get')
    def test_download_model_http_error(self, mock_get, model_manager):
        """Test download with HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        success = model_manager.download_model('esrgan_x4')
        assert success is False
    
    @patch('requests.get')
    def test_download_model_timeout(self, mock_get, model_manager):
        """Test download with timeout."""
        mock_get.side_effect = requests.Timeout("Connection timeout")
        
        success = model_manager.download_model('esrgan_x4')
        assert success is False
    
    @patch('requests.get')
    def test_download_progress_tracking(self, mock_get, model_manager, mock_requests_response):
        """Test download progress is tracked."""
        mock_get.return_value = mock_requests_response
        
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress_bar = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
            
            success = model_manager.download_model('esrgan_x4')
            
            assert success is True
            # Progress bar should be updated
            assert mock_progress_bar.update.called
    
    @patch('requests.get')
    def test_download_partial_file_cleanup(self, mock_get, model_manager):
        """Test partial download file is cleaned up on error."""
        # Mock response that fails during iteration
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.side_effect = Exception("Connection lost")
        mock_get.return_value = mock_response
        
        model_path = model_manager.cache_dir / 'RealESRGAN_x4plus.pth'
        
        success = model_manager.download_model('esrgan_x4')
        
        assert success is False
        assert not model_path.exists()  # Partial file should be removed
    
    def test_load_esrgan_model_success(self, model_manager):
        """Test successful ESRGAN model loading."""
        with patch_realesrgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                success = model_manager.load_esrgan_model(scale=4)
                
                assert success is True
                assert model_manager.is_esrgan_loaded() is True
                assert model_manager._current_esrgan_scale == 4
                assert isinstance(model_manager._esrgan_model, MockRealESRGANer)
    
    def test_load_esrgan_model_different_scales(self, model_manager):
        """Test loading ESRGAN with different scales."""
        with patch_realesrgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                # Load x2 model
                success = model_manager.load_esrgan_model(scale=2)
                assert success is True
                assert model_manager._current_esrgan_scale == 2
                
                # Load x4 model (should replace previous)
                success = model_manager.load_esrgan_model(scale=4)
                assert success is True
                assert model_manager._current_esrgan_scale == 4
    
    def test_load_esrgan_model_already_loaded(self, model_manager):
        """Test loading ESRGAN when already loaded with same scale."""
        with patch_realesrgan_import():
            with patch.object(model_manager, 'download_model', return_value=True) as mock_download:\n                # Load once\n                model_manager.load_esrgan_model(scale=4)\n                mock_download.reset_mock()\n                \n                # Load again with same scale\n                success = model_manager.load_esrgan_model(scale=4)\n                \n                assert success is True\n                mock_download.assert_not_called()  # Should not download again
    
    def test_load_esrgan_model_download_failure(self, model_manager):
        """Test ESRGAN loading with download failure."""
        with patch.object(model_manager, 'download_model', return_value=False):
            success = model_manager.load_esrgan_model(scale=4)
            
            assert success is False
            assert model_manager.is_esrgan_loaded() is False
    
    def test_load_esrgan_model_import_error(self, model_manager):
        """Test ESRGAN loading with import error."""
        with patch.object(model_manager, 'download_model', return_value=True):
            # Don't patch the import, so ImportError occurs
            success = model_manager.load_esrgan_model(scale=4)
            
            assert success is False
            assert model_manager.is_esrgan_loaded() is False
    
    def test_load_esrgan_model_initialization_error(self, model_manager):
        """Test ESRGAN loading with model initialization error."""
        with patch_realesrgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                # Mock RealESRGANer to raise exception
                with patch('tests.mocks.ai_models.MockRealESRGANer.__init__', side_effect=Exception("Init failed")):
                    success = model_manager.load_esrgan_model(scale=4)
                    
                    assert success is False
                    assert model_manager._esrgan_model is None
    
    def test_load_gfpgan_model_success(self, model_manager):
        """Test successful GFPGAN model loading."""
        with patch_gfpgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                success = model_manager.load_gfpgan_model()
                
                assert success is True
                assert model_manager.is_gfpgan_loaded() is True
                assert isinstance(model_manager._gfpgan_model, MockGFPGANer)
    
    def test_load_gfpgan_model_already_loaded(self, model_manager):
        """Test loading GFPGAN when already loaded."""
        with patch_gfpgan_import():
            with patch.object(model_manager, 'download_model', return_value=True) as mock_download:
                # Load once
                model_manager.load_gfpgan_model()
                mock_download.reset_mock()
                
                # Load again
                success = model_manager.load_gfpgan_model()
                
                assert success is True
                mock_download.assert_not_called()  # Should not download again
    
    def test_load_gfpgan_model_download_failure(self, model_manager):
        """Test GFPGAN loading with download failure."""
        with patch.object(model_manager, 'download_model', return_value=False):
            success = model_manager.load_gfpgan_model()
            
            assert success is False
            assert model_manager.is_gfpgan_loaded() is False
    
    def test_load_gfpgan_model_import_error(self, model_manager):
        """Test GFPGAN loading with import error."""
        with patch.object(model_manager, 'download_model', return_value=True):
            # Don't patch the import, so ImportError occurs
            success = model_manager.load_gfpgan_model()
            
            assert success is False
            assert model_manager.is_gfpgan_loaded() is False
    
    def test_get_models(self, model_manager):
        """Test getting model instances."""
        with patch_realesrgan_import(), patch_gfpgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                # Initially no models loaded
                assert model_manager.get_esrgan_model() is None
                assert model_manager.get_gfpgan_model() is None
                
                # Load models
                model_manager.load_esrgan_model()
                model_manager.load_gfpgan_model()
                
                # Should now return model instances
                esrgan_model = model_manager.get_esrgan_model()
                gfpgan_model = model_manager.get_gfpgan_model()
                
                assert isinstance(esrgan_model, MockRealESRGANer)
                assert isinstance(gfpgan_model, MockGFPGANer)
    
    def test_unload_models(self, model_manager):
        """Test unloading all models."""
        with patch_realesrgan_import(), patch_gfpgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                # Load models
                model_manager.load_esrgan_model()
                model_manager.load_gfpgan_model()
                
                assert model_manager.is_esrgan_loaded() is True
                assert model_manager.is_gfpgan_loaded() is True
                
                # Unload models
                with patch('torch.cuda.is_available', return_value=True), \
                     patch('torch.cuda.empty_cache') as mock_empty_cache:
                    model_manager.unload_models()
                    mock_empty_cache.assert_called_once()
                
                assert model_manager.is_esrgan_loaded() is False
                assert model_manager.is_gfpgan_loaded() is False
                assert model_manager._current_esrgan_scale is None
    
    def test_get_model_info(self, model_manager):
        """Test getting model information."""
        with patch_realesrgan_import(), patch_gfpgan_import():
            with patch.object(model_manager, 'download_model', return_value=True):
                # Initially no models loaded
                info = model_manager.get_model_info()
                
                assert info['esrgan_loaded'] is False
                assert info['esrgan_scale'] is None
                assert info['gfpgan_loaded'] is False
                assert 'cache_dir' in info
                assert 'available_models' in info
                
                # Load models
                model_manager.load_esrgan_model(scale=2)
                model_manager.load_gfpgan_model()
                
                info = model_manager.get_model_info()
                assert info['esrgan_loaded'] is True
                assert info['esrgan_scale'] == 2
                assert info['gfpgan_loaded'] is True
    
    def test_cleanup_cache(self, model_manager):
        """Test cache cleanup functionality."""
        cache_dir = model_manager.cache_dir
        
        # Create multiple model files with different timestamps
        import time
        
        files = []
        for i in range(5):
            file_path = cache_dir / f'model_{i}.pth'
            file_path.write_bytes(b'fake model data')
            files.append(file_path)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Cleanup keeping only 2 most recent
        model_manager.cleanup_cache(keep_recent=2)
        
        # Check only 2 most recent files remain
        remaining_files = list(cache_dir.glob('*.pth'))
        assert len(remaining_files) == 2
        
        # Should be the last 2 files created
        remaining_names = {f.name for f in remaining_files}
        assert 'model_3.pth' in remaining_names
        assert 'model_4.pth' in remaining_names
    
    def test_cleanup_cache_few_files(self, model_manager):
        """Test cache cleanup with fewer files than keep_recent."""
        cache_dir = model_manager.cache_dir
        
        # Create only 1 file
        file_path = cache_dir / 'single_model.pth'
        file_path.write_bytes(b'fake model data')
        
        # Cleanup should do nothing
        model_manager.cleanup_cache(keep_recent=3)
        
        assert file_path.exists()
    
    def test_cleanup_cache_error_handling(self, model_manager):
        """Test cache cleanup handles errors gracefully."""
        # Create a file that can't be deleted (simulate permission error)
        cache_dir = model_manager.cache_dir
        file_path = cache_dir / 'protected.pth'
        file_path.write_bytes(b'data')
        
        with patch('pathlib.Path.unlink', side_effect=PermissionError("Access denied")):
            # Should not raise exception
            model_manager.cleanup_cache(keep_recent=0)
    
    def test_destructor_cleanup(self, sample_config, test_logger):
        """Test destructor calls unload_models."""
        with patch_realesrgan_import():
            manager = ModelManager(sample_config, test_logger)
            
            with patch.object(manager, 'unload_models') as mock_unload:
                # Force destructor call
                del manager
                # Note: In Python, __del__ isn't guaranteed to be called immediately
                # This test verifies the method exists and can be called
                mock_unload.assert_called()


class TestModelManagerIntegration:
    """Integration tests for ModelManager."""
    
    def test_full_workflow_esrgan(self, sample_config, test_logger, sample_image_bgr):
        """Test full ESRGAN workflow."""
        with patch_realesrgan_import():
            manager = ModelManager(sample_config, test_logger)
            
            with patch.object(manager, 'download_model', return_value=True):
                # Load model
                success = manager.load_esrgan_model(scale=4)
                assert success is True
                
                # Get model and test enhancement
                model = manager.get_esrgan_model()
                enhanced = model.enhance(sample_image_bgr, outscale=4)
                
                # Check result
                original_h, original_w = sample_image_bgr.shape[:2]
                enhanced_h, enhanced_w = enhanced.shape[:2]
                
                assert enhanced_h == original_h * 4
                assert enhanced_w == original_w * 4
                assert model.get_call_count() == 1
    
    def test_full_workflow_gfpgan(self, sample_config, test_logger, sample_image_bgr):
        """Test full GFPGAN workflow."""
        with patch_gfpgan_import():
            manager = ModelManager(sample_config, test_logger)
            
            with patch.object(manager, 'download_model', return_value=True):
                # Load model
                success = manager.load_gfpgan_model()
                assert success is True
                
                # Get model and test enhancement
                model = manager.get_gfpgan_model()
                cropped, restored, enhanced = model.enhance(sample_image_bgr)
                
                # Check result
                assert enhanced.shape == sample_image_bgr.shape
                assert model.get_call_count() == 1
    
    def test_model_switching(self, sample_config, test_logger):
        """Test switching between different model scales."""
        with patch_realesrgan_import():
            manager = ModelManager(sample_config, test_logger)
            
            with patch.object(manager, 'download_model', return_value=True):
                # Load x2 model
                manager.load_esrgan_model(scale=2)
                assert manager._current_esrgan_scale == 2
                x2_model = manager.get_esrgan_model()
                
                # Load x4 model
                manager.load_esrgan_model(scale=4)
                assert manager._current_esrgan_scale == 4
                x4_model = manager.get_esrgan_model()
                
                # Should be different instances
                assert x2_model is not x4_model
                assert x4_model.scale == 4
    
    def test_memory_management(self, sample_config, test_logger):
        """Test memory management with model loading/unloading."""
        with patch_realesrgan_import(), patch_gfpgan_import():
            manager = ModelManager(sample_config, test_logger)
            
            with patch.object(manager, 'download_model', return_value=True):
                # Load models
                manager.load_esrgan_model()
                manager.load_gfpgan_model()
                
                # Verify loaded
                assert manager.is_esrgan_loaded()
                assert manager.is_gfpgan_loaded()
                
                # Unload and verify
                manager.unload_models()
                assert not manager.is_esrgan_loaded()
                assert not manager.is_gfpgan_loaded()
                
                # Models should be None
                assert manager.get_esrgan_model() is None
                assert manager.get_gfpgan_model() is None
    
    @pytest.mark.parametrize("model_key,expected_filename", [
        ("esrgan_x2", "RealESRGAN_x2plus.pth"),
        ("esrgan_x4", "RealESRGAN_x4plus.pth"),
        ("gfpgan", "GFPGANv1.3.pth")
    ])
    def test_model_file_paths(self, sample_config, test_logger, model_key, expected_filename):
        """Test model file path generation."""
        manager = ModelManager(sample_config, test_logger)
        
        model_info = ModelManager.MODEL_URLS[model_key]
        assert model_info['filename'] == expected_filename
        
        expected_path = manager.cache_dir / expected_filename
        # Path should be properly constructed
        assert expected_path.parent == manager.cache_dir