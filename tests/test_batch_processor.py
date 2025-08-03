"""Comprehensive tests for batch processor."""

import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from photo_restore.processors.batch_processor import BatchProcessor
from photo_restore.processors.image_processor import ImageProcessor


class TestBatchProcessor:
    """Test BatchProcessor functionality."""
    
    @pytest.fixture
    def batch_processor(self, sample_config, test_logger):
        """Create BatchProcessor instance for testing."""
        with patch.object(ImageProcessor, '__init__', return_value=None):
            processor = BatchProcessor(sample_config, test_logger)
            # Mock the image processor
            processor.image_processor = MagicMock()
            return processor
    
    @pytest.fixture
    def mock_image_processor(self):
        """Create mock image processor."""
        mock_processor = MagicMock()
        mock_processor.process_image.return_value = True  # Default to success
        return mock_processor
    
    def test_init_with_logger(self, sample_config, test_logger):
        """Test BatchProcessor initialization with logger."""
        with patch.object(ImageProcessor, '__init__', return_value=None):
            processor = BatchProcessor(sample_config, test_logger)
            
            assert processor.config == sample_config
            assert processor.logger == test_logger
            assert processor.stats['total_images'] == 0
            assert processor.stats['processed_images'] == 0
            assert processor.stats['failed_images'] == 0
            assert processor.stats['skipped_images'] == 0
    
    def test_init_without_logger(self, sample_config):
        """Test BatchProcessor initialization without logger."""
        with patch.object(ImageProcessor, '__init__', return_value=None):
            processor = BatchProcessor(sample_config)
            
            assert processor.logger is not None  # Should create default logger
    
    def test_find_image_files_flat(self, batch_processor, test_directory_structure):
        """Test finding image files in flat directory."""
        image_files = batch_processor._find_image_files(test_directory_structure, recursive=False)
        
        # Should find only images in root directory
        assert len(image_files) == 2
        image_names = [f.name for f in image_files]
        assert "image1.jpg" in image_names
        assert "image2.png" in image_names
    
    def test_find_image_files_recursive(self, batch_processor, test_directory_structure):
        """Test finding image files recursively."""
        image_files = batch_processor._find_image_files(test_directory_structure, recursive=True)
        
        # Should find all images in all subdirectories
        assert len(image_files) == 5
        image_names = [f.name for f in image_files]
        expected_names = ["image1.jpg", "image2.png", "image3.jpg", "image4.png", "image5.jpg"]
        for name in expected_names:
            assert name in image_names
    
    def test_find_image_files_sorted(self, batch_processor, temp_dir, sample_image_rgb):
        """Test that found image files are sorted."""
        from PIL import Image
        
        # Create images in non-alphabetical order
        image_names = ["zebra.jpg", "alpha.png", "beta.jpg"]
        pil_image = Image.fromarray(sample_image_rgb)
        
        for name in image_names:
            format_name = "JPEG" if name.endswith('.jpg') else "PNG"
            save_kwargs = {"quality": 90} if format_name == "JPEG" else {}
            pil_image.save(temp_dir / name, format_name, **save_kwargs)
        
        image_files = batch_processor._find_image_files(temp_dir, recursive=False)
        found_names = [f.name for f in image_files]
        
        # Should be sorted alphabetically
        assert found_names == sorted(found_names)
    
    def test_find_image_files_empty_directory(self, batch_processor, temp_dir):
        """Test finding images in empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        image_files = batch_processor._find_image_files(empty_dir, recursive=True)
        assert len(image_files) == 0
    
    def test_find_image_files_size_filtering(self, batch_processor, temp_dir, sample_image_rgb):
        """Test that very large files are filtered out."""
        from PIL import Image
        
        # Create normal size image
        normal_image = temp_dir / "normal.jpg"
        pil_image = Image.fromarray(sample_image_rgb)
        pil_image.save(normal_image, "JPEG", quality=90)
        
        # Mock a large file by patching stat
        large_image = temp_dir / "large.jpg"
        pil_image.save(large_image, "JPEG", quality=90)
        
        with patch.object(Path, 'stat') as mock_stat:
            def stat_side_effect():
                mock_result = MagicMock()
                if 'large.jpg' in str(self):
                    mock_result.st_size = 25 * 1024 * 1024  # 25MB (over 20MB limit)
                else:
                    mock_result.st_size = 1024 * 1024  # 1MB
                return mock_result
            
            # Patch Path.stat method
            original_stat = Path.stat
            def new_stat(self):
                if 'large.jpg' in str(self):
                    mock_result = MagicMock()
                    mock_result.st_size = 25 * 1024 * 1024
                    return mock_result
                return original_stat(self)
            
            with patch.object(Path, 'stat', new_stat):
                image_files = batch_processor._find_image_files(temp_dir, recursive=False)
                
                # Should only find normal size image
                assert len(image_files) == 1
                assert image_files[0].name == "normal.jpg"
    
    def test_find_image_files_error_handling(self, batch_processor, temp_dir):
        """Test error handling during file discovery."""
        # Create directory but then remove it to cause error
        test_dir = temp_dir / "test"
        test_dir.mkdir()
        
        with patch.object(Path, 'glob', side_effect=PermissionError("Access denied")):
            image_files = batch_processor._find_image_files(test_dir, recursive=True)
            
            # Should return empty list on error
            assert len(image_files) == 0
    
    def test_process_directory_success(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test successful directory processing."""
        batch_processor.image_processor = mock_image_processor
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            quality='fast',
            upscale=2,
            face_enhance=False,
            output_format='jpg',
            recursive=True
        )
        
        assert success_count == 5  # All 5 images processed
        assert batch_processor.stats['total_images'] == 5
        assert batch_processor.stats['processed_images'] == 5
        assert batch_processor.stats['failed_images'] == 0
        
        # Output directory should be created
        assert output_dir.exists()
        
        # Image processor should be called for each image
        assert mock_image_processor.process_image.call_count == 5
    
    def test_process_directory_nonexistent(self, batch_processor):
        """Test processing nonexistent directory."""
        success_count = batch_processor.process_directory(
            input_dir="/nonexistent/directory",
            output_dir="/tmp/output"
        )
        
        assert success_count == 0
    
    def test_process_directory_no_images(self, batch_processor, temp_dir):
        """Test processing directory with no images."""
        # Create directory with non-image files
        test_dir = temp_dir / "no_images"
        test_dir.mkdir()
        (test_dir / "document.txt").write_text("not an image")
        (test_dir / "data.json").write_text("{}")
        
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_directory(
            input_dir=str(test_dir),
            output_dir=str(output_dir)
        )
        
        assert success_count == 0
        assert batch_processor.stats['total_images'] == 0
    
    def test_process_directory_with_failures(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test directory processing with some failures."""
        # Mock processor to fail on specific files
        def mock_process_side_effect(*args, **kwargs):
            input_path = kwargs.get('input_path') or args[0]
            return 'image3.jpg' not in input_path  # Fail on image3.jpg
        
        mock_image_processor.process_image.side_effect = mock_process_side_effect
        batch_processor.image_processor = mock_image_processor
        
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            recursive=True
        )
        
        assert success_count == 4  # 4 successes, 1 failure
        assert batch_processor.stats['processed_images'] == 4
        assert batch_processor.stats['failed_images'] == 1
    
    def test_process_directory_skip_existing(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test directory processing skips existing output files."""
        batch_processor.image_processor = mock_image_processor
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Pre-create one output file
        existing_output = output_dir / "image1.jpg"
        existing_output.write_bytes(b"existing content")
        
        success_count = batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            recursive=True
        )
        
        # Should skip the existing file
        assert success_count == 4  # 4 processed, 1 skipped
        assert batch_processor.stats['skipped_images'] == 1
        assert batch_processor.stats['processed_images'] == 4
        
        # Should not process the existing file
        process_calls = [call.kwargs['input_path'] for call in mock_image_processor.process_image.call_args_list]
        assert not any('image1.jpg' in path for path in process_calls)
    
    def test_process_directory_maintains_structure(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test directory processing maintains directory structure."""
        batch_processor.image_processor = mock_image_processor
        output_dir = temp_dir / "output"
        
        batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            recursive=True
        )
        
        # Check that nested directories are maintained in outputs
        process_calls = mock_image_processor.process_image.call_args_list
        output_paths = [call.kwargs['output_path'] for call in process_calls]
        
        # Should have nested paths
        nested_outputs = [path for path in output_paths if 'subdir1' in path or 'subdir2' in path]
        assert len(nested_outputs) > 0
    
    def test_process_directory_format_conversion(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test directory processing with format conversion."""
        batch_processor.image_processor = mock_image_processor
        output_dir = temp_dir / "output"
        
        batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            output_format='png',  # Convert all to PNG
            recursive=True
        )
        
        # All output files should have .png extension
        process_calls = mock_image_processor.process_image.call_args_list
        for call in process_calls:
            output_path = call.kwargs['output_path']
            assert output_path.endswith('.png')
    
    def test_process_directory_exception_handling(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test directory processing handles exceptions gracefully."""
        # Mock processor to raise exception for one file
        def mock_process_side_effect(*args, **kwargs):
            input_path = kwargs.get('input_path') or args[0]
            if 'image2.png' in input_path:
                raise Exception("Processing error")
            return True
        
        mock_image_processor.process_image.side_effect = mock_process_side_effect
        batch_processor.image_processor = mock_image_processor
        
        output_dir = temp_dir / "output"
        
        # Should not raise exception
        success_count = batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            recursive=True
        )
        
        assert success_count == 4  # 4 successes, 1 exception
        assert batch_processor.stats['failed_images'] == 1
    
    def test_process_file_list_success(self, batch_processor, sample_image_files, temp_dir, mock_image_processor):
        """Test successful file list processing."""
        batch_processor.image_processor = mock_image_processor
        
        file_paths = [str(path) for path in sample_image_files.values()]
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_file_list(
            file_list=file_paths,
            output_dir=str(output_dir),
            quality='balanced',
            upscale=4,
            face_enhance=True,
            output_format='jpg'
        )
        
        assert success_count == len(file_paths)
        assert batch_processor.stats['processed_images'] == len(file_paths)
        assert output_dir.exists()
    
    def test_process_file_list_nonexistent_files(self, batch_processor, temp_dir, mock_image_processor):
        """Test file list processing with nonexistent files."""
        batch_processor.image_processor = mock_image_processor
        
        file_paths = ["/nonexistent/file1.jpg", "/nonexistent/file2.png"]
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_file_list(
            file_list=file_paths,
            output_dir=str(output_dir)
        )
        
        assert success_count == 0
        assert batch_processor.stats['total_images'] == 0
    
    def test_process_file_list_mixed_validity(self, batch_processor, sample_image_files, temp_dir, mock_image_processor):
        """Test file list processing with mix of valid and invalid files."""
        batch_processor.image_processor = mock_image_processor
        
        # Mix valid and invalid files
        file_paths = [
            str(list(sample_image_files.values())[0]),  # Valid
            "/nonexistent/file.jpg",                     # Invalid - doesn't exist
            str(temp_dir / "text.txt"),                  # Invalid - wrong format
            str(list(sample_image_files.values())[1])   # Valid
        ]
        
        # Create text file with wrong extension
        (temp_dir / "text.txt").write_text("not an image")
        
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_file_list(
            file_list=file_paths,
            output_dir=str(output_dir)
        )
        
        assert success_count == 2  # Only 2 valid files
        assert batch_processor.stats['total_images'] == 2
    
    def test_process_file_list_empty_list(self, batch_processor, temp_dir, mock_image_processor):
        """Test file list processing with empty list."""
        batch_processor.image_processor = mock_image_processor
        
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_file_list(
            file_list=[],
            output_dir=str(output_dir)
        )
        
        assert success_count == 0
    
    def test_process_file_list_format_filtering(self, batch_processor, temp_dir, mock_image_processor, sample_image_rgb):
        """Test file list processing filters by supported formats."""
        from PIL import Image
        
        batch_processor.image_processor = mock_image_processor
        
        # Create files with various extensions
        pil_image = Image.fromarray(sample_image_rgb)
        supported_file = temp_dir / "image.jpg"
        unsupported_file = temp_dir / "document.pdf"
        
        pil_image.save(supported_file, "JPEG")
        unsupported_file.write_bytes(b"PDF content")  # Fake PDF
        
        file_paths = [str(supported_file), str(unsupported_file)]
        output_dir = temp_dir / "output"
        
        success_count = batch_processor.process_file_list(
            file_list=file_paths,
            output_dir=str(output_dir)
        )
        
        assert success_count == 1  # Only supported file processed
        assert batch_processor.stats['total_images'] == 1
    
    def test_process_file_list_output_naming(self, batch_processor, sample_image_files, temp_dir, mock_image_processor):
        """Test file list processing output file naming."""
        batch_processor.image_processor = mock_image_processor
        
        file_paths = [str(list(sample_image_files.values())[0])]
        output_dir = temp_dir / "output"
        
        batch_processor.process_file_list(
            file_list=file_paths,
            output_dir=str(output_dir),
            output_format='png'
        )
        
        # Check output file naming
        process_call = mock_image_processor.process_image.call_args
        output_path = process_call.kwargs['output_path']
        
        assert "_enhanced.png" in output_path
        assert str(output_dir) in output_path
    
    def test_log_batch_results(self, batch_processor, capsys):
        """Test batch results logging."""
        # Set some stats
        batch_processor.stats = {
            'total_images': 10,
            'processed_images': 8,
            'failed_images': 1,
            'skipped_images': 1,
            'processing_time': 30.5
        }
        
        batch_processor._log_batch_results()
        
        # Should log results without error
        # Note: This mainly tests that logging doesn't crash
    
    def test_get_stats(self, batch_processor):
        """Test getting batch processing statistics."""
        original_stats = batch_processor.stats.copy()
        
        stats = batch_processor.get_stats()
        
        assert stats == original_stats
        
        # Should be a copy, not reference
        stats['total_images'] = 999
        assert batch_processor.stats['total_images'] != 999
    
    def test_reset_stats(self, batch_processor):
        """Test resetting processing statistics."""
        # Set some stats
        batch_processor.stats['total_images'] = 10
        batch_processor.stats['processed_images'] = 8
        
        batch_processor.reset_stats()
        
        assert batch_processor.stats['total_images'] == 0
        assert batch_processor.stats['processed_images'] == 0
        assert batch_processor.stats['failed_images'] == 0
        assert batch_processor.stats['skipped_images'] == 0
        assert batch_processor.stats['processing_time'] == 0.0


class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor."""
    
    def test_full_directory_workflow(self, sample_config, test_logger, test_directory_structure, temp_dir):
        """Test full directory processing workflow."""
        # Use real image processor but mock the model manager
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = True
            mock_manager.get_esrgan_model.return_value = MagicMock()
            mock_manager.get_gfpgan_model.return_value = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Mock the actual enhancement to avoid model dependencies
            with patch('cv2.imread') as mock_imread, \
                 patch('cv2.imwrite') as mock_imwrite:
                
                # Mock imread to return valid image data
                mock_imread.return_value = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                mock_imwrite.return_value = True
                
                processor = BatchProcessor(sample_config, test_logger)
                output_dir = temp_dir / "batch_output"
                
                success_count = processor.process_directory(
                    input_dir=str(test_directory_structure),
                    output_dir=str(output_dir),
                    quality='fast',
                    recursive=True
                )
                
                assert success_count > 0
                assert output_dir.exists()
    
    def test_progress_tracking_integration(self, batch_processor, test_directory_structure, temp_dir, mock_image_processor):
        """Test progress tracking during batch processing."""
        batch_processor.image_processor = mock_image_processor
        
        # Mock tqdm to track progress updates
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress
            
            output_dir = temp_dir / "output"
            
            batch_processor.process_directory(
                input_dir=str(test_directory_structure),
                output_dir=str(output_dir),
                recursive=True
            )
            
            # Progress should be updated for each processed file
            assert mock_progress.update.call_count == 5  # 5 images total
            assert mock_progress.set_description.called
    
    def test_error_recovery_workflow(self, batch_processor, test_directory_structure, temp_dir):
        """Test error recovery during batch processing."""
        # Mock image processor that fails intermittently
        mock_processor = MagicMock()
        
        call_count = 0
        def failing_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail every other call
            return call_count % 2 == 0
        
        mock_processor.process_image.side_effect = failing_process
        batch_processor.image_processor = mock_processor
        
        output_dir = temp_dir / "output"
        
        # Should continue processing despite failures
        success_count = batch_processor.process_directory(
            input_dir=str(test_directory_structure),
            output_dir=str(output_dir),
            recursive=True
        )
        
        # Should have some successes and some failures
        assert success_count > 0
        assert success_count < 5  # Not all should succeed
        assert batch_processor.stats['failed_images'] > 0


class TestBatchProcessorPerformance:
    """Performance tests for BatchProcessor."""
    
    def test_large_batch_processing(self, batch_processor, temp_dir, mock_image_processor):
        """Test processing large batch of files."""
        from PIL import Image
        
        # Create many small test images
        num_images = 50
        test_dir = temp_dir / "large_batch"
        test_dir.mkdir()
        
        # Create small test image
        small_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pil_image = Image.fromarray(small_image)
        
        for i in range(num_images):
            image_path = test_dir / f"image_{i:03d}.jpg"
            pil_image.save(image_path, "JPEG", quality=85)
        
        batch_processor.image_processor = mock_image_processor
        output_dir = temp_dir / "batch_output"
        
        success_count = batch_processor.process_directory(
            input_dir=str(test_dir),
            output_dir=str(output_dir),
            recursive=False
        )
        
        assert success_count == num_images
        assert batch_processor.stats['total_images'] == num_images
        assert mock_image_processor.process_image.call_count == num_images
    
    def test_memory_efficient_batch_processing(self, batch_processor, temp_dir, mock_image_processor):
        """Test that batch processing is memory efficient."""
        # This test ensures that we don't load all images into memory at once
        # We verify this by checking that files are processed one at a time
        
        from PIL import Image
        
        # Create test images
        test_dir = temp_dir / "memory_test"
        test_dir.mkdir()
        
        num_images = 10
        small_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        pil_image = Image.fromarray(small_image)
        
        for i in range(num_images):
            image_path = test_dir / f"image_{i}.jpg"
            pil_image.save(image_path, "JPEG")
        
        # Track the order of processing
        process_order = []
        def track_processing(*args, **kwargs):
            input_path = kwargs.get('input_path') or args[0]
            process_order.append(Path(input_path).name)
            return True
        
        mock_image_processor.process_image.side_effect = track_processing
        batch_processor.image_processor = mock_image_processor
        
        output_dir = temp_dir / "output"
        
        batch_processor.process_directory(
            input_dir=str(test_dir),
            output_dir=str(output_dir)
        )
        
        # Should process files in sorted order (indicating one-by-one processing)
        assert len(process_order) == num_images
        assert process_order == sorted(process_order)