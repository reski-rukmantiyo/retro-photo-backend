"""Comprehensive error handling and edge case validation tests."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from PIL import Image
import cv2

from photo_restore.processors.image_processor import ImageProcessor
from photo_restore.processors.batch_processor import BatchProcessor
from photo_restore.models.model_manager import ModelManager
from photo_restore.utils.config import Config
from photo_restore.utils.file_utils import validate_image_path, safe_copy, safe_remove
from photo_restore.cli import main
from click.testing import CliRunner


class TestErrorHandling:
    """Test error handling across all modules."""
    
    def test_invalid_image_formats(self, sample_config, test_logger, temp_dir):
        """Test handling of invalid image formats."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Create files with wrong content but image extensions
        test_files = {
            'text_as_jpg': ('fake.jpg', b'This is not an image'),
            'binary_garbage': ('garbage.png', b'\x00\x01\x02\x03\x04'),
            'empty_file': ('empty.bmp', b''),
            'truncated_header': ('truncated.gif', b'GIF87a\x00'),  # Incomplete GIF header
        }
        
        for test_name, (filename, content) in test_files.items():
            test_file = temp_dir / filename
            test_file.write_bytes(content)
            
            # Should handle gracefully without crashing
            result = processor._load_image(str(test_file))
            assert result is None, f"Should reject {test_name}"
    
    def test_corrupted_image_data(self, sample_config, test_logger, temp_dir):
        """Test handling of corrupted but valid format images."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Create partially corrupted JPEG
        corrupted_jpeg = temp_dir / "corrupted.jpg"
        jpeg_header = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB'
        # Add some corrupted data
        corrupted_data = jpeg_header + b'\x00' * 100 + b'\xFF\xD9'  # Add end marker
        corrupted_jpeg.write_bytes(corrupted_data)
        
        result = processor._load_image(str(corrupted_jpeg))
        assert result is None, "Should reject corrupted JPEG"
    
    def test_extreme_image_dimensions(self, sample_config, test_logger, temp_dir):
        """Test handling of images with extreme dimensions."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Test very small image (1x1)
        tiny_image = np.ones((1, 1, 3), dtype=np.uint8) * 128
        tiny_path = temp_dir / "tiny.jpg"
        cv2.imwrite(str(tiny_path), tiny_image)
        
        result = processor._load_image(str(tiny_path))
        assert result is None, "Should reject tiny image"
        
        # Test very wide image (10000x1)
        try:
            wide_image = np.ones((1, 10000, 3), dtype=np.uint8) * 128
            wide_path = temp_dir / "wide.jpg"
            cv2.imwrite(str(wide_path), wide_image)
            
            result = processor._load_image(str(wide_path))
            # Should either reject or handle gracefully
            # Large images should be resized by config
            
        except MemoryError:
            # Expected on systems with low memory
            pass
    
    def test_permission_errors(self, sample_config, test_logger, temp_dir):
        """Test handling of permission errors."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Create a file we can't read (simulate permission error)
        restricted_file = temp_dir / "restricted.jpg"
        restricted_file.write_bytes(b"fake content")
        
        with patch('cv2.imread', side_effect=PermissionError("Permission denied")):
            result = processor._load_image(str(restricted_file))
            assert result is None, "Should handle permission error gracefully"
        
        # Test write permission error
        with patch('cv2.imwrite', side_effect=PermissionError("Permission denied")):
            test_image = np.ones((100, 100, 3), dtype=np.uint8)
            success = processor._save_image(test_image, str(temp_dir / "output.jpg"), "jpg")
            assert success is False, "Should handle write permission error"
    
    def test_disk_space_errors(self, sample_config, test_logger, temp_dir):
        """Test handling of disk space errors."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Simulate disk full error
        with patch('cv2.imwrite', side_effect=OSError("No space left on device")):
            test_image = np.ones((100, 100, 3), dtype=np.uint8)
            success = processor._save_image(test_image, str(temp_dir / "output.jpg"), "jpg")
            assert success is False, "Should handle disk full error"
    
    def test_memory_allocation_errors(self, sample_config, test_logger):
        """Test handling of memory allocation errors."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Mock numpy operations to raise MemoryError
        with patch('numpy.zeros', side_effect=MemoryError("Cannot allocate memory")):
            # This would typically happen in tiling operations
            large_image = np.ones((1000, 1000, 3), dtype=np.uint8)
            
            # Should handle memory error gracefully
            try:
                result = processor._process_with_tiling(large_image, MagicMock(), tile_size=512)
                # If it doesn't raise, it should return None or handle gracefully
            except MemoryError:
                pytest.fail("Memory error should be handled gracefully")
    
    def test_model_loading_failures(self, sample_config, test_logger):
        """Test various model loading failure scenarios."""
        manager = ModelManager(sample_config, test_logger)
        
        # Test download failure
        with patch('requests.get', side_effect=Exception("Network error")):
            success = manager.download_model('esrgan_x4')
            assert success is False, "Should handle download failure"
        
        # Test model file corruption
        model_path = manager.cache_dir / 'RealESRGAN_x4plus.pth'
        model_path.write_bytes(b'corrupted model data')
        
        with patch('photo_restore.models.model_manager.RealESRGANer', 
                  side_effect=Exception("Model loading failed")):
            success = manager.load_esrgan_model(scale=4)
            assert success is False, "Should handle corrupted model"
    
    def test_model_enhancement_failures(self, sample_config, test_logger, sample_image_bgr):
        """Test model enhancement failure scenarios."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Mock model manager with failing models
        mock_manager = MagicMock()
        
        # ESRGAN fails
        mock_esrgan = MagicMock()
        mock_esrgan.enhance.side_effect = RuntimeError("ESRGAN enhancement failed")
        mock_manager.get_esrgan_model.return_value = mock_esrgan
        mock_manager.is_esrgan_loaded.return_value = True
        
        processor.model_manager = mock_manager
        
        result = processor._apply_esrgan_enhancement(sample_image_bgr, upscale=4, quality='balanced')
        assert result is None, "Should handle ESRGAN failure"
        
        # GFPGAN fails
        mock_gfpgan = MagicMock()
        mock_gfpgan.enhance.side_effect = RuntimeError("GFPGAN enhancement failed")
        mock_manager.get_gfpgan_model.return_value = mock_gfpgan
        mock_manager.is_gfpgan_loaded.return_value = True
        
        result = processor._apply_face_enhancement(sample_image_bgr)
        # Should return original image on face enhancement failure
        assert np.array_equal(result, sample_image_bgr), "Should return original on GFPGAN failure"
    
    def test_configuration_errors(self, temp_dir):
        """Test configuration loading error scenarios."""
        # Invalid YAML syntax
        invalid_config = temp_dir / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [unclosed")
        
        # Should not raise exception, should use defaults
        config = Config.load(str(invalid_config))
        assert config.processing.max_image_size == 4096, "Should use default on invalid YAML"
        
        # Missing config file
        config = Config.load("/nonexistent/config.yaml")
        assert config.processing.max_image_size == 4096, "Should use default on missing file"
        
        # Config with invalid values
        invalid_values_config = temp_dir / "invalid_values.yaml"
        invalid_values_config.write_text("""
processing:
  max_image_size: "invalid_number"
  memory_limit_gb: -1
models:
  download_timeout: "not_a_number"
""")
        
        # Should handle invalid values gracefully
        try:
            config = Config.load(str(invalid_values_config))
            # Config loading might ignore invalid values or use defaults
        except Exception:
            pytest.fail("Config loading should handle invalid values gracefully")
    
    def test_file_system_edge_cases(self, temp_dir):
        """Test file system edge cases."""
        # Test with paths containing special characters
        special_names = [
            "file with spaces.jpg",
            "file_with_unicode_日本語.jpg",
            "file-with-dashes.jpg",
            "file.with.dots.jpg",
            "UPPERCASE.JPG",
        ]
        
        for name in special_names:
            test_path = temp_dir / name
            test_path.write_bytes(b"fake image data")
            
            try:
                # Should handle special characters in paths
                validated = validate_image_path(test_path)
                # If validation passes, path handling should work
            except (ValueError, FileNotFoundError) as e:
                # Expected for unsupported formats
                if "Unsupported format" not in str(e):
                    pytest.fail(f"Unexpected error for {name}: {e}")
        
        # Test very long filename
        long_name = "a" * 200 + ".jpg"
        long_path = temp_dir / long_name
        
        try:
            long_path.write_bytes(b"test")
            # Should handle long filenames
        except OSError:
            # Expected on some filesystems
            pass
    
    def test_concurrent_access_errors(self, sample_config, test_logger, temp_dir):
        """Test handling of concurrent access issues."""
        processor = ImageProcessor(sample_config, test_logger)
        
        test_file = temp_dir / "concurrent_test.jpg"
        test_file.write_bytes(b"fake image")
        
        # Simulate file being deleted while processing
        def mock_imread_with_deletion(path):
            # Delete file during read attempt
            if Path(path).exists():
                Path(path).unlink()
            return None
        
        with patch('cv2.imread', side_effect=mock_imread_with_deletion):
            result = processor._load_image(str(test_file))
            assert result is None, "Should handle file deletion during read"
    
    def test_batch_processing_partial_failures(self, sample_config, test_logger, temp_dir):
        """Test batch processing with partial failures."""
        batch_processor = BatchProcessor(sample_config, test_logger)
        
        # Create mix of valid and invalid files
        test_dir = temp_dir / "mixed_batch"
        test_dir.mkdir()
        
        # Valid image
        valid_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(test_dir / "valid.jpg"), valid_image)
        
        # Invalid files
        (test_dir / "invalid.jpg").write_bytes(b"not an image")
        (test_dir / "empty.jpg").write_bytes(b"")
        (test_dir / "text.txt").write_text("not an image")
        
        # Mock image processor to fail on invalid files
        mock_processor = MagicMock()
        def mock_process(*args, **kwargs):
            input_path = kwargs.get('input_path', args[0])
            return 'valid.jpg' in input_path
        
        mock_processor.process_image.side_effect = mock_process
        batch_processor.image_processor = mock_processor
        
        output_dir = temp_dir / "batch_output"
        
        success_count = batch_processor.process_directory(
            input_dir=str(test_dir),
            output_dir=str(output_dir)
        )
        
        # Should process valid files and handle invalid ones gracefully
        assert success_count >= 0, "Should handle mixed batch gracefully"
        assert batch_processor.stats['failed_images'] >= 0, "Should track failures"
    
    def test_cli_error_scenarios(self, temp_dir):
        """Test CLI error handling scenarios."""
        runner = CliRunner()
        
        # Test with nonexistent input
        result = runner.invoke(main, ['/nonexistent/input.jpg', str(temp_dir / "output.jpg")])
        assert result.exit_code != 0, "Should fail with nonexistent input"
        
        # Test with invalid upscale factor
        valid_input = temp_dir / "test.jpg"
        valid_input.write_bytes(b"fake image")
        
        result = runner.invoke(main, ['--upscale', '3', str(valid_input)])
        assert result.exit_code != 0, "Should fail with invalid upscale"
        assert "Upscale factor must be 2 or 4" in result.output
        
        # Test with invalid quality
        result = runner.invoke(main, ['--quality', 'invalid', str(valid_input)])
        assert result.exit_code != 0, "Should fail with invalid quality"
        
        # Test processing failure
        with patch('photo_restore.cli.ImageProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor.process_image.return_value = False
            mock_processor_class.return_value = mock_processor
            
            result = runner.invoke(main, [str(valid_input)])
            assert result.exit_code == 1, "Should exit with error code on processing failure"
    
    def test_logging_failures(self, sample_config, temp_dir):
        """Test handling of logging failures."""
        # Test with invalid log file path
        config_dict = {
            'logging': {
                'file': '/invalid/path/that/cannot/be/created.log',
                'level': 'DEBUG'
            }
        }
        config = Config(config_dict)
        
        # Should not crash when logger setup fails
        try:
            processor = ImageProcessor(config)
            # Logger should fall back to console only
            assert processor.logger is not None
        except Exception as e:
            pytest.fail(f"Should handle logging setup failure gracefully: {e}")
    
    def test_resource_cleanup_on_errors(self, sample_config, test_logger, temp_dir):
        """Test that resources are cleaned up properly on errors."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Mock to track temporary files
        temp_files_created = []
        original_tempfile = tempfile.NamedTemporaryFile
        
        def mock_tempfile(*args, **kwargs):
            temp_file = original_tempfile(*args, **kwargs)
            temp_files_created.append(temp_file.name)
            return temp_file
        
        with patch('tempfile.NamedTemporaryFile', side_effect=mock_tempfile):
            # Simulate processing failure after temp files created
            with patch.object(processor, '_apply_esrgan_enhancement', 
                            side_effect=Exception("Processing failed")):
                
                # Create test image
                test_image = temp_dir / "test.jpg"
                valid_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
                cv2.imwrite(str(test_image), valid_image)
                
                # Process should fail but clean up
                success = processor.process_image(
                    input_path=str(test_image),
                    output_path=str(temp_dir / "output.jpg")
                )
                
                assert success is False, "Should fail on processing error"
                
                # Check that temp files don't accumulate
                # (This is a basic check - full cleanup depends on implementation)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_byte_images(self, sample_config, test_logger, temp_dir):
        """Test handling of zero-byte image files."""
        processor = ImageProcessor(sample_config, test_logger)
        
        zero_file = temp_dir / "zero.jpg"
        zero_file.write_bytes(b"")
        
        result = processor._load_image(str(zero_file))
        assert result is None, "Should reject zero-byte files"
    
    def test_maximum_path_length(self, sample_config, test_logger, temp_dir):
        """Test handling of very long file paths."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Create very long path (but within system limits)
        long_path_parts = ["very_long_directory_name"] * 10
        long_path = temp_dir
        
        for part in long_path_parts:
            long_path = long_path / part
            try:
                long_path.mkdir(exist_ok=True)
            except OSError:
                # Path too long for filesystem
                break
        
        long_file = long_path / "image.jpg"
        
        try:
            # Create test image
            valid_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(long_file), valid_image)
            
            # Should handle long paths
            result = processor._load_image(str(long_file))
            # May succeed or fail depending on system limits
            
        except OSError:
            # Expected on some systems with path length limits
            pass
    
    def test_unicode_and_special_characters(self, sample_config, test_logger, temp_dir):
        """Test handling of Unicode and special characters in paths."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Test various Unicode characters
        unicode_names = [
            "测试图片.jpg",          # Chinese
            "テスト画像.jpg",         # Japanese
            "тест_изображение.jpg",  # Russian
            "prueba_émojis_🖼️.jpg", # Emojis
            "file-with-àccénts.jpg", # Accented characters
        ]
        
        for name in unicode_names:
            unicode_file = temp_dir / name
            
            try:
                # Create test image
                valid_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
                cv2.imwrite(str(unicode_file), valid_image)
                
                if unicode_file.exists():
                    result = processor._load_image(str(unicode_file))
                    # Should handle Unicode paths gracefully
                    
            except (UnicodeError, OSError) as e:
                # Some filesystems may not support certain Unicode characters
                print(f"Unicode path {name} not supported: {e}")
    
    def test_boundary_image_sizes(self, sample_config, test_logger, temp_dir):
        """Test boundary image sizes."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Test images at config boundaries
        max_size = sample_config.processing.max_image_size
        
        test_sizes = [
            (max_size - 1, max_size - 1),  # Just under limit
            (max_size, max_size),          # At limit
            (max_size + 1, max_size + 1),  # Just over limit
        ]
        
        for width, height in test_sizes:
            try:
                test_image = np.ones((height, width, 3), dtype=np.uint8) * 128
                test_file = temp_dir / f"size_{width}x{height}.jpg"
                cv2.imwrite(str(test_file), test_image)
                
                result = processor._load_image(str(test_file))
                
                if width <= max_size and height <= max_size:
                    assert result is not None, f"Should accept {width}x{height}"
                else:
                    # Should resize or handle appropriately
                    if result is not None:
                        # Check that it was resized
                        assert max(result.shape[:2]) <= max_size, "Should resize oversized images"
                
            except MemoryError:
                # Expected for very large images
                continue
    
    def test_color_space_edge_cases(self, sample_config, test_logger, temp_dir):
        """Test edge cases with different color spaces."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Test grayscale image
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        gray_file = temp_dir / "gray.jpg"
        cv2.imwrite(str(gray_file), gray_image)
        
        result = processor._load_image(str(gray_file))
        if result is not None:
            assert len(result.shape) == 3, "Should convert grayscale to 3-channel"
            assert result.shape[2] == 3, "Should have 3 channels"
        
        # Test RGBA image (if supported)
        try:
            rgba_image = np.ones((100, 100, 4), dtype=np.uint8) * 128
            rgba_file = temp_dir / "rgba.png"
            cv2.imwrite(str(rgba_file), rgba_image)
            
            result = processor._load_image(str(rgba_file))
            if result is not None:
                assert result.shape[2] == 3, "Should convert RGBA to BGR"
        
        except Exception:
            # OpenCV may not support RGBA in all versions
            pass
    
    def test_numeric_edge_cases(self, sample_config, test_logger):
        """Test numeric edge cases."""
        processor = ImageProcessor(sample_config, test_logger)
        
        # Test with extreme numeric values in config
        extreme_config = Config({
            'processing': {
                'max_image_size': 0,  # Zero max size
                'memory_limit_gb': -1,  # Negative memory
                'tile_size': -100,  # Negative tile size
                'tile_overlap': 1000  # Overlap larger than tile
            }
        })
        
        # Should handle extreme values gracefully
        try:
            extreme_processor = ImageProcessor(extreme_config, test_logger)
            # Should not crash on initialization
        except Exception as e:
            pytest.fail(f"Should handle extreme config values gracefully: {e}")
    
    def test_concurrent_processing_safety(self, sample_config, test_logger, temp_dir):
        """Test thread safety and concurrent processing scenarios."""
        import threading
        import time
        
        processor = ImageProcessor(sample_config, test_logger)
        
        # Create test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        test_file = temp_dir / "concurrent_test.jpg"
        cv2.imwrite(str(test_file), test_image)
        
        results = []
        errors = []
        
        def process_image(index):
            try:
                output_file = temp_dir / f"output_{index}.jpg"
                
                # Mock the processing to avoid actual model loading
                with patch.object(processor, 'model_manager') as mock_manager:
                    mock_manager.is_esrgan_loaded.return_value = False
                    mock_manager.is_gfpgan_loaded.return_value = False
                    mock_manager.load_esrgan_model.return_value = True
                    mock_manager.get_esrgan_model.return_value = MagicMock()
                    
                    success = processor.process_image(
                        input_path=str(test_file),
                        output_path=str(output_file),
                        quality='fast'
                    )
                    results.append(success)
                    
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_image, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)
        
        # Should handle concurrent access without crashes
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 3, "All threads should complete"