"""Comprehensive tests for image processor."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from photo_restore.processors.image_processor import ImageProcessor
from photo_restore.utils.config import Config
from tests.mocks import MockRealESRGANer, MockGFPGANer, patch_realesrgan_import, patch_gfpgan_import


class TestImageProcessor:
    """Test ImageProcessor functionality."""
    
    @pytest.fixture
    def image_processor(self, sample_config, test_logger):
        """Create ImageProcessor instance for testing."""
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(sample_config, test_logger)
            processor.model_manager = mock_manager
            return processor
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        mock_manager = MagicMock()
        mock_manager.is_esrgan_loaded.return_value = False
        mock_manager.is_gfpgan_loaded.return_value = False
        mock_manager.load_esrgan_model.return_value = True
        mock_manager.load_gfpgan_model.return_value = True
        
        # Mock models
        mock_esrgan = MockRealESRGANer(scale=4)
        mock_gfpgan = MockGFPGANer()
        mock_manager.get_esrgan_model.return_value = mock_esrgan
        mock_manager.get_gfpgan_model.return_value = mock_gfpgan
        
        return mock_manager
    
    def test_init_with_logger(self, sample_config, test_logger):
        """Test ImageProcessor initialization with logger."""
        with patch('photo_restore.processors.image_processor.ModelManager'):
            processor = ImageProcessor(sample_config, test_logger)
            
            assert processor.config == sample_config
            assert processor.logger == test_logger
            assert processor.stats['processed_images'] == 0
            assert processor.stats['processing_time'] == 0.0
    
    def test_init_without_logger(self, sample_config):
        """Test ImageProcessor initialization without logger."""
        with patch('photo_restore.processors.image_processor.ModelManager'):
            processor = ImageProcessor(sample_config)
            
            assert processor.logger is not None  # Should create default logger
    
    def test_load_image_success(self, image_processor, sample_image_file):
        """Test successful image loading."""
        image = image_processor._load_image(str(sample_image_file))
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Height, width, channels
        assert image.shape[2] == 3  # BGR channels
    
    def test_load_image_nonexistent_file(self, image_processor):
        """Test loading nonexistent image file."""
        image = image_processor._load_image('/nonexistent/image.jpg')
        assert image is None
    
    def test_load_image_unsupported_format(self, image_processor, temp_dir):
        """Test loading unsupported image format."""
        # Create text file with image extension
        fake_image = temp_dir / "fake.jpg"
        fake_image.write_text("not an image")
        
        image = image_processor._load_image(str(fake_image))
        assert image is None
    
    def test_load_image_too_small(self, image_processor, temp_dir):
        """Test loading image that's too small."""
        # Create very small image
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        small_path = temp_dir / "small.jpg"
        cv2.imwrite(str(small_path), small_image)
        
        result = image_processor._load_image(str(small_path))
        assert result is None
    
    def test_load_image_too_large(self, image_processor, temp_dir):
        """Test loading image that's too large gets resized."""
        # Mock image that exceeds max size
        with patch('cv2.imread') as mock_imread:
            # Create mock large image
            large_image = np.zeros((8192, 8192, 3), dtype=np.uint8)  # Larger than config max
            mock_imread.return_value = large_image
            
            with patch('cv2.resize') as mock_resize:
                mock_resize.return_value = np.zeros((4096, 4096, 3), dtype=np.uint8)
                
                result = image_processor._load_image(str(temp_dir / "large.jpg"))
                
                assert result is not None
                mock_resize.assert_called_once()
    
    def test_ensure_models_loaded_esrgan_only(self, image_processor, mock_model_manager):
        """Test ensuring ESRGAN model is loaded."""
        image_processor.model_manager = mock_model_manager
        
        image_processor._ensure_models_loaded(upscale=4, face_enhance=False)
        
        mock_model_manager.load_esrgan_model.assert_called_once_with(4)
        mock_model_manager.load_gfpgan_model.assert_not_called()
    
    def test_ensure_models_loaded_both_models(self, image_processor, mock_model_manager):
        """Test ensuring both models are loaded."""
        image_processor.model_manager = mock_model_manager
        
        image_processor._ensure_models_loaded(upscale=2, face_enhance=True)
        
        mock_model_manager.load_esrgan_model.assert_called_once_with(2)
        mock_model_manager.load_gfpgan_model.assert_called_once()
    
    def test_ensure_models_loaded_already_loaded(self, image_processor, mock_model_manager):
        """Test model loading when already loaded."""
        mock_model_manager.is_esrgan_loaded.return_value = True
        mock_model_manager.is_gfpgan_loaded.return_value = True
        image_processor.model_manager = mock_model_manager
        
        image_processor._ensure_models_loaded(upscale=4, face_enhance=True)
        
        # Should not call load methods if already loaded
        mock_model_manager.load_esrgan_model.assert_not_called()
        mock_model_manager.load_gfpgan_model.assert_not_called()
    
    def test_apply_esrgan_enhancement_success(self, image_processor, sample_image_bgr, mock_model_manager):
        """Test successful ESRGAN enhancement."""
        image_processor.model_manager = mock_model_manager
        
        enhanced = image_processor._apply_esrgan_enhancement(sample_image_bgr, upscale=4, quality='balanced')
        
        assert enhanced is not None
        assert enhanced.shape[0] == sample_image_bgr.shape[0] * 4  # Height upscaled
        assert enhanced.shape[1] == sample_image_bgr.shape[1] * 4  # Width upscaled
    
    def test_apply_esrgan_enhancement_no_model(self, image_processor, sample_image_bgr):
        """Test ESRGAN enhancement when model is not available."""
        mock_manager = MagicMock()
        mock_manager.get_esrgan_model.return_value = None
        image_processor.model_manager = mock_manager
        
        enhanced = image_processor._apply_esrgan_enhancement(sample_image_bgr, upscale=4, quality='balanced')
        assert enhanced is None
    
    def test_apply_esrgan_enhancement_with_tiling(self, image_processor, mock_model_manager):
        """Test ESRGAN enhancement with tiling for large images."""
        # Create large image that requires tiling
        large_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image_processor.model_manager = mock_model_manager
        
        with patch.object(image_processor, '_process_with_tiling') as mock_tiling:
            mock_tiling.return_value = np.zeros((4096, 4096, 3), dtype=np.uint8)
            
            enhanced = image_processor._apply_esrgan_enhancement(large_image, upscale=4, quality='balanced')
            
            mock_tiling.assert_called_once()
            assert enhanced is not None
    
    def test_apply_esrgan_enhancement_error(self, image_processor, sample_image_bgr, mock_model_manager):
        """Test ESRGAN enhancement with error."""
        # Make model raise exception
        mock_esrgan = mock_model_manager.get_esrgan_model.return_value
        mock_esrgan.enhance.side_effect = Exception("Enhancement failed")
        image_processor.model_manager = mock_model_manager
        
        enhanced = image_processor._apply_esrgan_enhancement(sample_image_bgr, upscale=4, quality='balanced')
        assert enhanced is None
    
    def test_apply_face_enhancement_success(self, image_processor, sample_image_bgr, mock_model_manager):
        """Test successful face enhancement."""
        image_processor.model_manager = mock_model_manager
        
        enhanced = image_processor._apply_face_enhancement(sample_image_bgr)
        
        assert enhanced is not None
        assert enhanced.shape == sample_image_bgr.shape
        # Mock GFPGAN should have been called
        mock_gfpgan = mock_model_manager.get_gfpgan_model.return_value
        assert mock_gfpgan.get_call_count() == 1
    
    def test_apply_face_enhancement_no_model(self, image_processor, sample_image_bgr):
        """Test face enhancement when model is not available."""
        mock_manager = MagicMock()
        mock_manager.get_gfpgan_model.return_value = None
        image_processor.model_manager = mock_manager
        
        enhanced = image_processor._apply_face_enhancement(sample_image_bgr)
        
        # Should return original image
        assert np.array_equal(enhanced, sample_image_bgr)
    
    def test_apply_face_enhancement_error(self, image_processor, sample_image_bgr, mock_model_manager):
        """Test face enhancement with error."""
        mock_gfpgan = mock_model_manager.get_gfpgan_model.return_value
        mock_gfpgan.enhance.side_effect = Exception("Face enhancement failed")
        image_processor.model_manager = mock_model_manager
        
        enhanced = image_processor._apply_face_enhancement(sample_image_bgr)
        
        # Should return original image on error
        assert np.array_equal(enhanced, sample_image_bgr)
    
    def test_process_with_tiling(self, image_processor, sample_image_bgr):
        """Test image processing with tiling."""
        # Create mock model
        mock_model = MagicMock()
        
        def mock_enhance(tile):
            # Return upscaled tile
            h, w = tile.shape[:2]
            return np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        mock_model.enhance = mock_enhance
        
        # Test with small tile size to force tiling
        tile_size = 128
        enhanced = image_processor._process_with_tiling(sample_image_bgr, mock_model, tile_size)
        
        assert enhanced is not None
        # Should be upscaled
        assert enhanced.shape[0] == sample_image_bgr.shape[0] * 2
        assert enhanced.shape[1] == sample_image_bgr.shape[1] * 2
    
    def test_save_image_jpg(self, image_processor, sample_image_bgr, temp_dir):
        """Test saving image as JPEG."""
        output_path = temp_dir / "output.jpg"
        
        success = image_processor._save_image(sample_image_bgr, str(output_path), "jpg")
        
        assert success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_save_image_png(self, image_processor, sample_image_bgr, temp_dir):
        """Test saving image as PNG."""
        output_path = temp_dir / "output.png"
        
        success = image_processor._save_image(sample_image_bgr, str(output_path), "png")
        
        assert success is True
        assert output_path.exists()
    
    def test_save_image_creates_directory(self, image_processor, sample_image_bgr, temp_dir):
        """Test saving image creates output directory."""
        output_path = temp_dir / "nested" / "subdir" / "output.jpg"
        
        success = image_processor._save_image(sample_image_bgr, str(output_path), "jpg")
        
        assert success is True
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_save_image_error(self, image_processor, sample_image_bgr):
        """Test save image with invalid path."""
        # Try to save to invalid path
        invalid_path = "/invalid/path/output.jpg"
        
        success = image_processor._save_image(sample_image_bgr, invalid_path, "jpg")
        assert success is False
    
    def test_process_image_success(self, image_processor, sample_image_file, temp_dir, mock_model_manager):
        """Test successful image processing."""
        image_processor.model_manager = mock_model_manager
        output_path = temp_dir / "processed.jpg"
        
        # Mock progress callback
        progress_calls = []
        def progress_callback(percent):
            progress_calls.append(percent)
        
        success = image_processor.process_image(
            input_path=str(sample_image_file),
            output_path=str(output_path),
            quality='balanced',
            upscale=4,
            face_enhance=True,
            output_format='jpg',
            progress_callback=progress_callback
        )
        
        assert success is True
        assert output_path.exists()
        assert len(progress_calls) > 0
        assert progress_calls[-1] == 100  # Should end at 100%
        assert image_processor.stats['processed_images'] == 1
    
    def test_process_image_no_face_enhance(self, image_processor, sample_image_file, temp_dir, mock_model_manager):
        """Test image processing without face enhancement."""
        image_processor.model_manager = mock_model_manager
        output_path = temp_dir / "processed.jpg"
        
        success = image_processor.process_image(
            input_path=str(sample_image_file),
            output_path=str(output_path),
            face_enhance=False
        )
        
        assert success is True
        # GFPGAN should not have been called
        mock_gfpgan = mock_model_manager.get_gfpgan_model.return_value
        assert mock_gfpgan.get_call_count() == 0
    
    def test_process_image_load_failure(self, image_processor):
        """Test image processing with load failure."""
        success = image_processor.process_image(
            input_path="/nonexistent/image.jpg",
            output_path="/tmp/output.jpg"
        )
        
        assert success is False
        assert image_processor.stats['processed_images'] == 0
    
    def test_process_image_esrgan_failure(self, image_processor, sample_image_file, temp_dir):
        """Test image processing with ESRGAN failure."""
        # Mock model manager with failing ESRGAN
        mock_manager = MagicMock()
        mock_manager.is_esrgan_loaded.return_value = False
        mock_manager.is_gfpgan_loaded.return_value = False
        mock_manager.load_esrgan_model.return_value = True
        mock_manager.get_esrgan_model.return_value = None  # No model available
        image_processor.model_manager = mock_manager
        
        output_path = temp_dir / "output.jpg"
        
        success = image_processor.process_image(
            input_path=str(sample_image_file),
            output_path=str(output_path)
        )
        
        assert success is False
    
    def test_process_image_save_failure(self, image_processor, sample_image_file, mock_model_manager):
        """Test image processing with save failure."""
        image_processor.model_manager = mock_model_manager
        
        # Mock save to fail
        with patch.object(image_processor, '_save_image', return_value=False):
            success = image_processor.process_image(
                input_path=str(sample_image_file),
                output_path="/invalid/path/output.jpg"
            )
            
            assert success is False
    
    def test_process_image_exception_handling(self, image_processor, sample_image_file, temp_dir):
        """Test image processing handles exceptions gracefully."""
        # Mock load_image to raise exception
        with patch.object(image_processor, '_load_image', side_effect=Exception("Unexpected error")):
            success = image_processor.process_image(
                input_path=str(sample_image_file),
                output_path=str(temp_dir / "output.jpg")
            )
            
            assert success is False
    
    @pytest.mark.parametrize("quality,expected_tile_size", [
        ('fast', 256),
        ('balanced', 512),
        ('best', 1024)
    ])
    def test_quality_settings_integration(self, image_processor, sample_image_file, temp_dir, 
                                        mock_model_manager, quality, expected_tile_size):
        """Test different quality settings affect processing."""
        image_processor.model_manager = mock_model_manager
        output_path = temp_dir / f"output_{quality}.jpg"
        
        success = image_processor.process_image(
            input_path=str(sample_image_file),
            output_path=str(output_path),
            quality=quality
        )
        
        assert success is True
        # Could verify tile size was used correctly by checking config calls
    
    def test_get_stats(self, image_processor):
        """Test getting processing statistics."""
        stats = image_processor.get_stats()
        
        assert 'processed_images' in stats
        assert 'processing_time' in stats
        assert 'enhancement_success_rate' in stats
        
        # Should be a copy, not reference
        stats['processed_images'] = 999
        assert image_processor.stats['processed_images'] != 999
    
    def test_performance_logging_integration(self, image_processor, sample_image_file, temp_dir, mock_model_manager):
        """Test performance logging during processing."""
        image_processor.model_manager = mock_model_manager
        output_path = temp_dir / "output.jpg"
        
        # Capture logger calls
        with patch.object(image_processor.logger, 'info') as mock_info:
            success = image_processor.process_image(
                input_path=str(sample_image_file),
                output_path=str(output_path)
            )
            
            assert success is True
            # Should have logged completion
            assert any("Successfully processed" in str(call) for call in mock_info.call_args_list)


class TestImageProcessorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_process_corrupted_image(self, sample_config, test_logger, corrupted_image_file, temp_dir):
        """Test processing corrupted image file."""
        with patch('photo_restore.processors.image_processor.ModelManager'):
            processor = ImageProcessor(sample_config, test_logger)
            output_path = temp_dir / "output.jpg"
            
            success = processor.process_image(
                input_path=str(corrupted_image_file),
                output_path=str(output_path)
            )
            
            assert success is False
    
    def test_process_zero_size_image(self, sample_config, test_logger, temp_dir):
        """Test processing zero-size image file."""
        zero_file = temp_dir / "zero.jpg"
        zero_file.write_bytes(b'')  # Empty file
        
        with patch('photo_restore.processors.image_processor.ModelManager'):
            processor = ImageProcessor(sample_config, test_logger)
            output_path = temp_dir / "output.jpg"
            
            success = processor.process_image(
                input_path=str(zero_file),
                output_path=str(output_path)
            )
            
            assert success is False
    
    def test_process_with_disk_full_simulation(self, image_processor, sample_image_file, temp_dir, mock_model_manager):
        """Test processing when disk is full (save fails)."""
        image_processor.model_manager = mock_model_manager
        output_path = temp_dir / "output.jpg"
        
        # Mock cv2.imwrite to simulate disk full
        with patch('cv2.imwrite', return_value=False):
            success = image_processor.process_image(
                input_path=str(sample_image_file),
                output_path=str(output_path)
            )
            
            assert success is False
    
    def test_tiling_with_small_overlap(self, image_processor):
        """Test tiling with minimal overlap."""
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Mock model that doubles size
        mock_model = MagicMock()
        mock_model.enhance.side_effect = lambda tile: np.zeros((tile.shape[0]*2, tile.shape[1]*2, 3), dtype=np.uint8)
        
        # Set small overlap
        image_processor.config.processing.tile_overlap = 1
        
        result = image_processor._process_with_tiling(test_image, mock_model, tile_size=100)
        
        assert result is not None
        assert result.shape[0] == test_image.shape[0] * 2
        assert result.shape[1] == test_image.shape[1] * 2
    
    def test_progress_callback_exception(self, image_processor, sample_image_file, temp_dir, mock_model_manager):
        """Test processing continues even if progress callback raises exception."""
        image_processor.model_manager = mock_model_manager
        output_path = temp_dir / "output.jpg"
        
        def failing_callback(percent):
            raise Exception("Callback failed")
        
        # Should not raise exception despite failing callback
        success = image_processor.process_image(
            input_path=str(sample_image_file),
            output_path=str(output_path),
            progress_callback=failing_callback
        )
        
        assert success is True
    
    def test_model_loading_failure_recovery(self, image_processor, sample_image_file, temp_dir):
        """Test recovery when model loading fails."""
        # Mock model manager with loading failures
        mock_manager = MagicMock()
        mock_manager.is_esrgan_loaded.return_value = False
        mock_manager.load_esrgan_model.return_value = False  # Fail to load
        image_processor.model_manager = mock_manager
        
        output_path = temp_dir / "output.jpg"
        
        success = image_processor.process_image(
            input_path=str(sample_image_file),
            output_path=str(output_path)
        )
        
        assert success is False


class TestImageProcessorPerformance:
    """Test performance aspects of image processing."""
    
    def test_large_image_processing(self, sample_config, test_logger, temp_dir, mock_model_manager):
        """Test processing large image with tiling."""
        # Create large test image
        large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        large_image_path = temp_dir / "large.jpg"
        cv2.imwrite(str(large_image_path), large_image)
        
        with patch('photo_restore.processors.image_processor.ModelManager'):
            processor = ImageProcessor(sample_config, test_logger)
            processor.model_manager = mock_model_manager
            
            output_path = temp_dir / "large_output.jpg"
            
            success = processor.process_image(
                input_path=str(large_image_path),
                output_path=str(output_path),
                quality='fast'  # Use fast quality for quicker test
            )
            
            assert success is True
            assert output_path.exists()
    
    def test_memory_efficient_tiling(self, image_processor):
        """Test that tiling processes tiles individually without keeping all in memory."""
        # Create large image
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        # Mock model that tracks calls
        tile_calls = []
        def mock_enhance(tile):
            tile_calls.append(tile.shape)
            return np.zeros((tile.shape[0]*2, tile.shape[1]*2, 3), dtype=np.uint8)
        
        mock_model = MagicMock()
        mock_model.enhance.side_effect = mock_enhance
        
        result = image_processor._process_with_tiling(large_image, mock_model, tile_size=200)
        
        assert result is not None
        assert len(tile_calls) > 1  # Should have processed multiple tiles
        
        # All tiles should be reasonable size (not whole image)
        for tile_shape in tile_calls:
            assert tile_shape[0] <= 232  # tile_size + overlap
            assert tile_shape[1] <= 232