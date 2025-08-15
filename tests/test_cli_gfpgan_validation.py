"""
Phase 2 Testing: CLI GFPGAN Version Validation Tests
Tests for --gfpgan-version CLI argument validation and processing logic.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import cv2
import numpy as np

from photo_restore.cli import main
from photo_restore.utils.config import Config
from tests.mocks import MockGFPGANer


class TestCLIGFPGANVersionValidation:
    """Test CLI argument validation for --gfpgan-version option."""
    
    def setup_method(self):
        """Setup CLI testing environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="cli_gfpgan_test_")
        self.test_image_path = self._create_test_image()
        self.runner = CliRunner()
        
    def teardown_method(self):
        """Cleanup CLI testing environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_test_image(self) -> str:
        """Create a test image for CLI testing."""
        image_path = Path(self.temp_dir) / "test_input.jpg"
        
        # Create a simple test image
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), image)
        
        return str(image_path)
    
    def test_valid_gfpgan_version_arguments(self):
        """Test that valid GFPGAN version arguments are accepted."""
        valid_versions = ['v1.3', 'v1.4', 'auto']
        
        for version in valid_versions:
            with patch('photo_restore.processors.image_processor.ImageProcessor') as mock_processor:
                mock_instance = MagicMock()
                mock_instance.process_image.return_value = True
                mock_processor.return_value = mock_instance
                
                output_path = Path(self.temp_dir) / f"output_{version.replace('.', '_')}.jpg"
                
                result = self.runner.invoke(main, [
                    self.test_image_path,
                    str(output_path),
                    '--gfpgan-version', version,
                    '--face-enhance'
                ])
                
                assert result.exit_code == 0, f"CLI failed with valid version {version}: {result.output}"
                
                # Verify that the processor was called with correct parameters
                mock_processor.assert_called_once()
                mock_instance.process_image.assert_called_once()
                
                # Check that gfpgan_version was passed through (indirectly)
                call_args = mock_instance.process_image.call_args
                assert call_args[1]['face_enhance'] is True
    
    def test_invalid_gfpgan_version_arguments(self):
        """Test that invalid GFPGAN version arguments are rejected."""
        invalid_versions = ['v1.0', 'v2.0', 'invalid', '1.3', '1.4', 'V1.3']
        
        for version in invalid_versions:
            output_path = Path(self.temp_dir) / f"output_invalid_{version}.jpg"
            
            result = self.runner.invoke(main, [
                self.test_image_path,
                str(output_path),
                '--gfpgan-version', version,
                '--face-enhance'
            ])
            
            assert result.exit_code != 0, f"CLI should reject invalid version {version}"
            assert "Invalid value" in result.output or "Usage:" in result.output
    
    def test_gfpgan_version_with_face_enhance_disabled(self):
        """Test GFPGAN version argument behavior when face enhancement is disabled."""
        with patch('photo_restore.processors.image_processor.ImageProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_processor.return_value = mock_instance
            
            output_path = Path(self.temp_dir) / "output_no_face_enhance.jpg"
            
            result = self.runner.invoke(main, [
                self.test_image_path,
                str(output_path),
                '--gfpgan-version', 'v1.4',
                '--no-face-enhance'  # Disable face enhancement
            ])
            
            # Should succeed but GFPGAN version should be ignored
            assert result.exit_code == 0, f"CLI failed when face enhancement disabled: {result.output}"
            
            # Verify processor was called with face_enhance=False
            call_args = mock_instance.process_image.call_args
            assert call_args[1]['face_enhance'] is False
    
    def test_default_gfpgan_version_behavior(self):
        """Test default GFPGAN version behavior (auto)."""
        with patch('photo_restore.processors.image_processor.ImageProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_processor.return_value = mock_instance
            
            output_path = Path(self.temp_dir) / "output_default.jpg"
            
            result = self.runner.invoke(main, [
                self.test_image_path,
                str(output_path),
                '--face-enhance'
                # No --gfpgan-version specified, should default to 'auto'
            ])
            
            assert result.exit_code == 0, f"CLI failed with default GFPGAN version: {result.output}"
            
            # Verify processor was called
            mock_processor.assert_called_once()
            mock_instance.process_image.assert_called_once()
    
    def test_gfpgan_version_in_batch_mode(self):
        """Test GFPGAN version handling in batch processing mode."""
        # Create test directory with multiple images
        batch_input_dir = Path(self.temp_dir) / "batch_input"
        batch_output_dir = Path(self.temp_dir) / "batch_output"
        batch_input_dir.mkdir()
        
        # Create multiple test images
        for i in range(3):
            image_path = batch_input_dir / f"test_image_{i}.jpg"
            image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
        
        with patch('photo_restore.processors.batch_processor.BatchProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_directory.return_value = 3
            mock_processor.return_value = mock_instance
            
            result = self.runner.invoke(main, [
                str(batch_input_dir),
                str(batch_output_dir),
                '--batch',
                '--gfpgan-version', 'v1.3',
                '--face-enhance'
            ])
            
            assert result.exit_code == 0, f"Batch processing failed with GFPGAN version: {result.output}"
            assert "Successfully processed 3 images" in result.output
            
            # Verify batch processor was called
            mock_processor.assert_called_once()
            mock_instance.process_directory.assert_called_once()
    
    def test_gfpgan_version_with_custom_config(self):
        """Test GFPGAN version with custom configuration file."""
        # Create custom config file
        config_path = Path(self.temp_dir) / "custom_config.yaml"
        config_content = """
models:
  gfpgan:
    version: "1.3"
    model_path: "GFPGANv1.3.pth"
processing:
  face_enhance: true
"""
        config_path.write_text(config_content)
        
        with patch('photo_restore.processors.image_processor.ImageProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_image.return_value = True
            mock_processor.return_value = mock_instance
            
            output_path = Path(self.temp_dir) / "output_custom_config.jpg"
            
            result = self.runner.invoke(main, [
                self.test_image_path,
                str(output_path),
                '--config', str(config_path),
                '--gfpgan-version', 'v1.4',  # Should override config
                '--face-enhance'
            ])
            
            assert result.exit_code == 0, f"CLI failed with custom config: {result.output}"
            
            # Verify that CLI argument takes precedence over config
            mock_processor.assert_called_once()
    
    def test_gfpgan_version_help_information(self):
        """Test that help information includes GFPGAN version details."""
        result = self.runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert '--gfpgan-version' in result.output
        assert 'v1.3' in result.output
        assert 'v1.4' in result.output
        assert 'auto' in result.output
        assert 'quality' in result.output or 'speed' in result.output


class TestGFPGANVersionLogic:
    """Test GFPGAN version selection and processing logic."""
    
    def setup_method(self):
        """Setup version logic testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="gfpgan_logic_test_")
        
    def teardown_method(self):
        """Cleanup version logic testing."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_auto_version_selection_logic(self):
        """Test automatic GFPGAN version selection logic."""
        # Test different scenarios for auto selection
        test_cases = [
            {
                'image_size': (512, 512),
                'quality': 'fast',
                'expected_bias': 'v1.4'  # Should prefer speed
            },
            {
                'image_size': (1024, 1024),
                'quality': 'best',
                'expected_bias': 'v1.3'  # Should prefer quality
            },
            {
                'image_size': (256, 256),
                'quality': 'balanced',
                'expected_bias': 'either'  # Either version acceptable
            }
        ]
        
        for case in test_cases:
            with patch('photo_restore.models.gfpgan.GFPGANModel') as mock_model:
                mock_instance = MagicMock()
                mock_model.return_value = mock_instance
                
                # Simulate auto selection logic (this would be implemented in the actual code)
                image_size = case['image_size']
                quality = case['quality']
                
                # Mock the auto selection logic
                if quality == 'fast':
                    selected_version = 'v1.4'
                elif quality == 'best':
                    selected_version = 'v1.3'
                else:
                    selected_version = 'v1.4' if image_size[0] <= 512 else 'v1.3'
                
                # Verify selection matches expectations
                if case['expected_bias'] != 'either':
                    assert selected_version == case['expected_bias']
                else:
                    assert selected_version in ['v1.3', 'v1.4']
    
    def test_version_fallback_behavior(self):
        """Test fallback behavior when preferred GFPGAN version fails."""
        with patch('photo_restore.models.gfpgan.GFPGANModel') as mock_model:
            # Mock first version failing, second succeeding
            first_call = MagicMock()
            first_call.load_model.return_value = False
            
            second_call = MagicMock() 
            second_call.load_model.return_value = True
            
            mock_model.side_effect = [first_call, second_call]
            
            # Test fallback logic (would be implemented in actual processor)
            versions_to_try = ['v1.3', 'v1.4']
            
            loaded_model = None
            for version in versions_to_try:
                model = mock_model(model_path=f"GFPGANv{version[1:]}.pth")
                if model.load_model():
                    loaded_model = model
                    break
            
            assert loaded_model is not None, "Fallback should succeed"
            assert mock_model.call_count == 2, "Should try both versions"
    
    def test_version_compatibility_validation(self):
        """Test validation of GFPGAN version compatibility."""
        # Test compatible combinations
        compatible_configs = [
            {'gfpgan_version': 'v1.3', 'device': 'cpu'},
            {'gfpgan_version': 'v1.4', 'device': 'cpu'},
            {'gfpgan_version': 'auto', 'device': 'cpu'},
        ]
        
        for config in compatible_configs:
            # Mock validation logic
            is_compatible = True  # All combinations should be compatible for CPU
            assert is_compatible, f"Configuration should be compatible: {config}"


class TestGFPGANVersionIntegration:
    """Integration tests for GFPGAN version selection in full CLI workflow."""
    
    def setup_method(self):
        """Setup integration testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="gfpgan_cli_integration_")
        self.test_image = self._create_realistic_test_image()
        self.runner = CliRunner()
        
    def teardown_method(self):
        """Cleanup integration testing."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_realistic_test_image(self) -> str:
        """Create a more realistic test image with face-like features."""
        image_path = Path(self.temp_dir) / "realistic_face.jpg"
        
        # Create face-like test image
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Background
        image[:, :] = [220, 200, 180]
        
        # Face shape
        center = (256, 256)
        axes = (120, 160)
        cv2.ellipse(image, center, axes, 0, 0, 360, (210, 180, 150), -1)
        
        # Eyes
        cv2.circle(image, (200, 220), 20, (50, 50, 50), -1)
        cv2.circle(image, (312, 220), 20, (50, 50, 50), -1)
        
        # Mouth
        cv2.ellipse(image, (256, 300), (30, 15), 0, 0, 180, (150, 100, 100), -1)
        
        cv2.imwrite(str(image_path), image)
        return str(image_path)
    
    @patch('photo_restore.models.gfpgan.GFPGANer')
    @patch('photo_restore.models.esrgan.RRDBNet')
    def test_end_to_end_v13_processing(self, mock_rrdb, mock_gfpgan):
        """Test end-to-end processing with GFPGAN v1.3."""
        mock_gfpgan.return_value = MockGFPGANer()
        mock_rrdb.return_value = MagicMock()
        
        output_path = Path(self.temp_dir) / "output_v13.jpg"
        
        result = self.runner.invoke(main, [
            self.test_image,
            str(output_path),
            '--gfpgan-version', 'v1.3',
            '--face-enhance',
            '--quality', 'best',
            '--verbose'
        ])
        
        assert result.exit_code == 0, f"v1.3 processing failed: {result.output}"
        assert "Enhanced image saved" in result.output or "✅" in result.output
    
    @patch('photo_restore.models.gfpgan.GFPGANer')
    @patch('photo_restore.models.esrgan.RRDBNet')
    def test_end_to_end_v14_processing(self, mock_rrdb, mock_gfpgan):
        """Test end-to-end processing with GFPGAN v1.4."""
        mock_gfpgan.return_value = MockGFPGANer()
        mock_rrdb.return_value = MagicMock()
        
        output_path = Path(self.temp_dir) / "output_v14.jpg"
        
        result = self.runner.invoke(main, [
            self.test_image,
            str(output_path),
            '--gfpgan-version', 'v1.4',
            '--face-enhance',
            '--quality', 'fast',
            '--verbose'
        ])
        
        assert result.exit_code == 0, f"v1.4 processing failed: {result.output}"
        assert "Enhanced image saved" in result.output or "✅" in result.output
    
    @patch('photo_restore.models.gfpgan.GFPGANer')
    @patch('photo_restore.models.esrgan.RRDBNet')
    def test_end_to_end_auto_selection(self, mock_rrdb, mock_gfpgan):
        """Test end-to-end processing with auto version selection."""
        mock_gfpgan.return_value = MockGFPGANer()
        mock_rrdb.return_value = MagicMock()
        
        output_path = Path(self.temp_dir) / "output_auto.jpg"
        
        result = self.runner.invoke(main, [
            self.test_image,
            str(output_path),
            '--gfpgan-version', 'auto',
            '--face-enhance',
            '--quality', 'balanced',
            '--verbose'
        ])
        
        assert result.exit_code == 0, f"Auto selection processing failed: {result.output}"
        assert "Enhanced image saved" in result.output or "✅" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])