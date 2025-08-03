"""Comprehensive tests for CLI interface."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner

from photo_restore.cli import main
from photo_restore.processors.image_processor import ImageProcessor
from photo_restore.processors.batch_processor import BatchProcessor


class TestCLI:
    """Test CLI interface functionality."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_processors(self):
        """Mock both image and batch processors."""
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc, \
             patch('photo_restore.cli.BatchProcessor') as mock_batch_proc:
            
            # Mock image processor
            mock_img_instance = MagicMock()
            mock_img_instance.process_image.return_value = True
            mock_img_proc.return_value = mock_img_instance
            
            # Mock batch processor
            mock_batch_instance = MagicMock()
            mock_batch_instance.process_directory.return_value = 5
            mock_batch_proc.return_value = mock_batch_instance
            
            yield {
                'image_processor_class': mock_img_proc,
                'image_processor': mock_img_instance,
                'batch_processor_class': mock_batch_proc,
                'batch_processor': mock_batch_instance
            }
    
    def test_cli_single_image_basic(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test basic single image processing."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        assert "Successfully processed" in result.output or "Enhanced image saved to" in result.output
        mock_processors['image_processor'].process_image.assert_called_once()
    
    def test_cli_single_image_auto_output(self, cli_runner, sample_image_file, mock_processors):
        """Test single image processing with auto-generated output path."""
        result = cli_runner.invoke(main, [str(sample_image_file)])
        
        assert result.exit_code == 0
        mock_processors['image_processor'].process_image.assert_called_once()
        
        # Check auto-generated output path
        call_args = mock_processors['image_processor'].process_image.call_args
        output_path = call_args.kwargs['output_path']
        assert "_enhanced" in output_path
    
    def test_cli_batch_processing(self, cli_runner, test_directory_structure, temp_dir, mock_processors):
        """Test batch directory processing."""
        output_dir = temp_dir / "output"
        
        result = cli_runner.invoke(main, [
            '--batch',
            str(test_directory_structure),
            str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Successfully processed" in result.output
        mock_processors['batch_processor'].process_directory.assert_called_once()
    
    def test_cli_batch_auto_output(self, cli_runner, test_directory_structure, mock_processors):
        """Test batch processing with auto-generated output directory."""
        result = cli_runner.invoke(main, [
            '--batch',
            str(test_directory_structure)
        ])
        
        assert result.exit_code == 0
        
        # Check auto-generated output directory
        call_args = mock_processors['batch_processor'].process_directory.call_args
        output_dir = call_args.kwargs['output_dir']
        assert "_enhanced" in output_dir
    
    @pytest.mark.parametrize("quality", ['fast', 'balanced', 'best'])
    def test_cli_quality_options(self, cli_runner, sample_image_file, temp_dir, mock_processors, quality):
        """Test different quality options."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--quality', quality,
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        
        # Check quality was passed correctly
        call_args = mock_processors['image_processor'].process_image.call_args
        assert call_args.kwargs['quality'] == quality
    
    @pytest.mark.parametrize("upscale", [2, 4])
    def test_cli_upscale_options(self, cli_runner, sample_image_file, temp_dir, mock_processors, upscale):
        """Test different upscale factors."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--upscale', str(upscale),
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        
        # Check upscale was passed correctly
        call_args = mock_processors['image_processor'].process_image.call_args
        assert call_args.kwargs['upscale'] == upscale
    
    def test_cli_upscale_invalid(self, cli_runner, sample_image_file, temp_dir):
        """Test invalid upscale factor."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--upscale', '3',  # Invalid, only 2 or 4 allowed
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code != 0
        assert "Upscale factor must be 2 or 4" in result.output
    
    def test_cli_face_enhance_flag(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test face enhancement flag."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--face-enhance',
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        
        # Check face enhance was enabled
        call_args = mock_processors['image_processor'].process_image.call_args
        assert call_args.kwargs['face_enhance'] is True
    
    def test_cli_no_face_enhance(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test disabling face enhancement."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--no-face-enhance',
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        
        # Check face enhance was disabled
        call_args = mock_processors['image_processor'].process_image.call_args
        assert call_args.kwargs['face_enhance'] is False
    
    @pytest.mark.parametrize("format", ['jpg', 'png'])
    def test_cli_output_format(self, cli_runner, sample_image_file, temp_dir, mock_processors, format):
        """Test different output formats."""
        output_path = temp_dir / f"output.{format}"
        
        result = cli_runner.invoke(main, [
            '--output-format', format,
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        
        # Check output format was passed correctly
        call_args = mock_processors['image_processor'].process_image.call_args
        assert call_args.kwargs['output_format'] == format
    
    def test_cli_custom_config(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test custom configuration file."""
        # Create custom config file
        config_file = temp_dir / "custom_config.yaml"
        config_content = """
processing:
  max_image_size: 2048
models:
  cache_dir: /tmp/custom_models
"""
        config_file.write_text(config_content)
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--config', str(config_file),
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
    
    def test_cli_verbose_logging(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test verbose logging option."""
        output_path = temp_dir / "output.jpg"
        
        with patch('photo_restore.cli.setup_logger') as mock_setup_logger:
            result = cli_runner.invoke(main, [
                '--verbose',
                str(sample_image_file),
                str(output_path)
            ])
            
            assert result.exit_code == 0
            mock_setup_logger.assert_called_with(level='DEBUG')
    
    def test_cli_nonexistent_input(self, cli_runner):
        """Test CLI with nonexistent input file."""
        result = cli_runner.invoke(main, ['/nonexistent/file.jpg'])
        
        assert result.exit_code != 0
        # Click should handle the file existence check
    
    def test_cli_processing_failure(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test CLI when processing fails."""
        # Mock processor to fail
        mock_processors['image_processor'].process_image.return_value = False
        
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 1
        assert "Processing failed" in result.output or "❌" in result.output
    
    def test_cli_batch_processing_failure(self, cli_runner, test_directory_structure, temp_dir, mock_processors):
        """Test CLI when batch processing fails."""
        # Mock batch processor to return 0 (no images processed)
        mock_processors['batch_processor'].process_directory.return_value = 0
        
        output_dir = temp_dir / "output"
        
        result = cli_runner.invoke(main, [
            '--batch',
            str(test_directory_structure),
            str(output_dir)
        ])
        
        assert result.exit_code == 0  # Still succeeds, just processes 0 images
        assert "Successfully processed 0 images" in result.output
    
    def test_cli_exception_handling(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI handles exceptions gracefully."""
        # Mock Config.load to raise exception
        with patch('photo_restore.cli.Config.load', side_effect=Exception("Config error")):
            output_path = temp_dir / "output.jpg"
            
            result = cli_runner.invoke(main, [
                str(sample_image_file),
                str(output_path)
            ])
            
            assert result.exit_code == 1
            assert "Error:" in result.output
    
    def test_cli_progress_callback(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test progress callback integration."""
        output_path = temp_dir / "output.jpg"
        
        # Mock progress callback to capture calls
        def mock_process_image(*args, **kwargs):
            progress_callback = kwargs.get('progress_callback')
            if progress_callback:
                progress_callback(50)
                progress_callback(100)
            return True
        
        mock_processors['image_processor'].process_image.side_effect = mock_process_image
        
        result = cli_runner.invoke(main, [
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        # Progress should be displayed
        # Note: tqdm output might not appear in captured output during testing
    
    def test_cli_combined_options(self, cli_runner, sample_image_file, temp_dir, mock_processors):
        """Test CLI with multiple options combined."""
        output_path = temp_dir / "output.png"
        
        result = cli_runner.invoke(main, [
            '--quality', 'best',
            '--upscale', '4',
            '--face-enhance',
            '--output-format', 'png',
            '--verbose',
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code == 0
        
        # Check all options were passed correctly
        call_args = mock_processors['image_processor'].process_image.call_args
        assert call_args.kwargs['quality'] == 'best'
        assert call_args.kwargs['upscale'] == 4
        assert call_args.kwargs['face_enhance'] is True
        assert call_args.kwargs['output_format'] == 'png'
    
    def test_cli_batch_combined_options(self, cli_runner, test_directory_structure, temp_dir, mock_processors):
        """Test batch CLI with multiple options."""
        output_dir = temp_dir / "output"
        
        result = cli_runner.invoke(main, [
            '--batch',
            '--quality', 'fast',
            '--upscale', '2',
            '--no-face-enhance',
            '--output-format', 'jpg',
            str(test_directory_structure),
            str(output_dir)
        ])
        
        assert result.exit_code == 0
        
        # Check all options were passed to batch processor
        call_args = mock_processors['batch_processor'].process_directory.call_args
        assert call_args.kwargs['quality'] == 'fast'
        assert call_args.kwargs['upscale'] == 2
        assert call_args.kwargs['face_enhance'] is False
        assert call_args.kwargs['output_format'] == 'jpg'


class TestCLIHelp:
    """Test CLI help and documentation."""
    
    def test_cli_help(self, cli_runner):
        """Test CLI help output."""
        result = cli_runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Photo restoration CLI tool" in result.output
        assert "--batch" in result.output
        assert "--quality" in result.output
        assert "--upscale" in result.output
        assert "--face-enhance" in result.output
        assert "--output-format" in result.output
        assert "--config" in result.output
        assert "--verbose" in result.output
    
    def test_cli_examples_in_help(self, cli_runner):
        """Test that examples are shown in help."""
        result = cli_runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Examples:" in result.output
        assert "photo-restore input.jpg output.jpg" in result.output
        assert "photo-restore --batch" in result.output
    
    def test_cli_version_info(self, cli_runner):
        """Test CLI version information."""
        # Click applications typically support --version
        result = cli_runner.invoke(main, ['--version'])
        
        # May or may not be implemented, but shouldn't crash
        assert result.exit_code in [0, 2]  # 0 for success, 2 for no such option


class TestCLIIntegration:
    """Integration tests for CLI."""
    
    def test_cli_real_config_loading(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI with real configuration loading."""
        # Create a real config file
        config_file = temp_dir / "test_config.yaml"
        config_content = """
processing:
  max_image_size: 1024
  memory_limit_gb: 1.0

models:
  cache_dir: /tmp/test_models
  download_timeout: 60

logging:
  level: DEBUG
  format: simple
"""
        config_file.write_text(config_content)
        
        # Mock processors to avoid actual processing
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_img_proc.return_value = mock_instance
            
            output_path = temp_dir / "output.jpg"
            
            result = cli_runner.invoke(main, [
                '--config', str(config_file),
                str(sample_image_file),
                str(output_path)
            ])
            
            assert result.exit_code == 0
            
            # Verify config was used in processor creation
            call_args = mock_img_proc.call_args
            config = call_args[0][0]  # First argument should be config
            assert config.processing.max_image_size == 1024
            assert config.models.download_timeout == 60
    
    def test_cli_logger_integration(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI logger integration."""
        # Mock processors but allow real logger setup
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_img_proc.return_value = mock_instance
            
            output_path = temp_dir / "output.jpg"
            
            result = cli_runner.invoke(main, [
                '--verbose',
                str(sample_image_file),
                str(output_path)
            ])
            
            assert result.exit_code == 0
            
            # Logger should have been passed to processor
            call_args = mock_img_proc.call_args
            logger = call_args[0][1]  # Second argument should be logger
            assert logger is not None
    
    def test_cli_path_handling(self, cli_runner, temp_dir):
        """Test CLI path handling with different path formats."""
        # Create test image
        from PIL import Image
        import numpy as np
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Test relative path
        rel_input = "test_image.jpg"
        pil_image.save(temp_dir / rel_input, "JPEG")
        
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_img_proc.return_value = mock_instance
            
            # Change to temp directory to test relative paths
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                result = cli_runner.invoke(main, [rel_input])
                assert result.exit_code == 0
                
            finally:
                os.chdir(old_cwd)
    
    def test_cli_signal_handling(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI handles interruption gracefully."""
        # Mock processor to simulate long-running process
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            
            def long_process(*args, **kwargs):
                # Simulate KeyboardInterrupt during processing
                raise KeyboardInterrupt("User interrupted")
            
            mock_instance.process_image.side_effect = long_process
            mock_img_proc.return_value = mock_instance
            
            output_path = temp_dir / "output.jpg"
            
            result = cli_runner.invoke(main, [
                str(sample_image_file),
                str(output_path)
            ])
            
            # Should handle interruption gracefully
            assert result.exit_code == 1
    
    def test_cli_memory_constraints(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI behavior with memory constraints."""
        # Create config with low memory limit
        config_file = temp_dir / "low_memory_config.yaml"
        config_content = """
processing:
  max_image_size: 512
  memory_limit_gb: 0.1
  tile_size: 128
"""
        config_file.write_text(config_content)
        
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_img_proc.return_value = mock_instance
            
            output_path = temp_dir / "output.jpg"
            
            result = cli_runner.invoke(main, [
                '--config', str(config_file),
                str(sample_image_file),
                str(output_path)
            ])
            
            assert result.exit_code == 0
            
            # Verify low memory config was applied
            call_args = mock_img_proc.call_args
            config = call_args[0][0]
            assert config.processing.memory_limit_gb == 0.1
            assert config.processing.tile_size == 128


class TestCLIErrorCases:
    """Test CLI error handling."""
    
    def test_cli_invalid_quality(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI with invalid quality option."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--quality', 'invalid',
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code != 0
        # Click should validate the choice
    
    def test_cli_invalid_format(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI with invalid output format."""
        output_path = temp_dir / "output.jpg"
        
        result = cli_runner.invoke(main, [
            '--output-format', 'invalid',
            str(sample_image_file),
            str(output_path)
        ])
        
        assert result.exit_code != 0
        # Click should validate the choice
    
    def test_cli_permission_error(self, cli_runner, sample_image_file):
        """Test CLI with permission error on output."""
        # Try to write to a restricted location
        restricted_path = "/root/output.jpg"  # Typically not writable
        
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            mock_instance.process_image.side_effect = PermissionError("Permission denied")
            mock_img_proc.return_value = mock_instance
            
            result = cli_runner.invoke(main, [
                str(sample_image_file),
                restricted_path
            ])
            
            assert result.exit_code == 1
            assert "Error:" in result.output
    
    def test_cli_disk_full_error(self, cli_runner, sample_image_file, temp_dir):
        """Test CLI with disk full error."""
        output_path = temp_dir / "output.jpg"
        
        with patch('photo_restore.cli.ImageProcessor') as mock_img_proc:
            mock_instance = MagicMock()
            mock_instance.process_image.side_effect = OSError("No space left on device")
            mock_img_proc.return_value = mock_instance
            
            result = cli_runner.invoke(main, [
                str(sample_image_file),
                str(output_path)
            ])
            
            assert result.exit_code == 1
            assert "Error:" in result.output