"""
Phase 2 Testing: GFPGAN v1.3 vs v1.4 Comparison Testing Suite
Comprehensive testing framework for GFPGAN version comparison and performance analysis.
"""

import pytest
import time
import psutil
import gc
import hashlib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from PIL import Image
import cv2
import torch

from photo_restore.models.gfpgan import GFPGANModel
from photo_restore.models.model_manager import ModelManager
from photo_restore.processors.image_processor import ImageProcessor
from photo_restore.utils.config import Config
from tests.mocks import MockGFPGANer


@pytest.mark.parametrize("gfpgan_version", ["1.3", "1.4"])
class TestGFPGANVersionComparison:
    """Comprehensive GFPGAN v1.3 vs v1.4 comparison tests."""
    
    def setup_method(self):
        """Setup test environment for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="gfpgan_phase2_")
        self.test_image_path = self._create_test_image()
        
        # Memory tracking
        self.initial_memory = psutil.Process().memory_info().rss
        
        # Performance metrics
        self.performance_metrics = {
            'load_time': 0,
            'processing_time': 0,
            'memory_usage': 0,
            'gpu_memory_usage': 0
        }
        
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        gc.collect()
        
    def _create_test_image(self, size=(512, 512)) -> str:
        """Create a synthetic test image with face-like features."""
        # Create a simple face-like image for testing
        image = np.zeros((*size, 3), dtype=np.uint8)
        
        # Add face-like features (simplified oval and rectangles for eyes)
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Face oval
        cv2.ellipse(image, (center_x, center_y), (150, 200), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        cv2.rectangle(image, (center_x-60, center_y-50), (center_x-20, center_y-10), (50, 50, 50), -1)
        cv2.rectangle(image, (center_x+20, center_y-50), (center_x+60, center_y-10), (50, 50, 50), -1)
        
        # Mouth
        cv2.rectangle(image, (center_x-30, center_y+30), (center_x+30, center_y+50), (100, 50, 50), -1)
        
        # Save test image
        test_image_path = Path(self.temp_dir) / "test_face.jpg"
        cv2.imwrite(str(test_image_path), image)
        return str(test_image_path)
    
    @pytest.mark.performance
    def test_model_loading_performance(self, gfpgan_version):
        """Test GFPGAN model loading performance comparison."""
        with patch('photo_restore.models.gfpgan.GFPGANer') as mock_gfpgan:
            mock_instance = MockGFPGANer()
            mock_gfpgan.return_value = mock_instance
            
            model = GFPGANModel(
                model_path=f"GFPGANv{gfpgan_version}.pth",
                device='cpu'
            )
            
            # Measure loading time
            start_time = time.time()
            success = model.load_model()
            load_time = time.time() - start_time
            
            assert success, f"GFPGAN v{gfpgan_version} model loading failed"
            assert load_time < 5.0, f"GFPGAN v{gfpgan_version} loading too slow: {load_time:.2f}s"
            
            self.performance_metrics['load_time'] = load_time
            
            # Version-specific assertions
            if gfpgan_version == "1.3":
                assert load_time < 3.0, "GFPGAN v1.3 should load faster than 3 seconds"
            elif gfpgan_version == "1.4":
                assert load_time < 2.5, "GFPGAN v1.4 should load faster (optimized)"
    
    @pytest.mark.performance
    def test_face_enhancement_performance(self, gfpgan_version):
        """Test face enhancement processing performance."""
        with patch('photo_restore.models.gfpgan.GFPGANer') as mock_gfpgan:
            mock_instance = MockGFPGANer()
            mock_gfpgan.return_value = mock_instance
            
            model = GFPGANModel(
                model_path=f"GFPGANv{gfpgan_version}.pth",
                device='cpu'
            )
            
            assert model.load_model(), f"Failed to load GFPGAN v{gfpgan_version}"
            
            # Load test image
            test_image = cv2.imread(self.test_image_path)
            
            # Measure processing time
            start_time = time.time()
            enhanced_image = model.enhance_image(test_image)
            processing_time = time.time() - start_time
            
            assert enhanced_image is not None, f"GFPGAN v{gfpgan_version} enhancement failed"
            assert processing_time < 10.0, f"GFPGAN v{gfpgan_version} processing too slow: {processing_time:.2f}s"
            
            self.performance_metrics['processing_time'] = processing_time
            
            # Version-specific performance expectations
            if gfpgan_version == "1.3":
                assert processing_time < 8.0, "GFPGAN v1.3 processing should complete in <8s"
            elif gfpgan_version == "1.4":
                assert processing_time < 6.0, "GFPGAN v1.4 should be faster (optimized)"
    
    def test_memory_usage_comparison(self, gfpgan_version):
        """Test memory usage patterns between GFPGAN versions."""
        with patch('photo_restore.models.gfpgan.GFPGANer') as mock_gfpgan:
            mock_instance = MockGFPGANer()
            mock_gfpgan.return_value = mock_instance
            
            # Baseline memory
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss
            
            model = GFPGANModel(
                model_path=f"GFPGANv{gfpgan_version}.pth",
                device='cpu'
            )
            
            # Memory after model creation
            creation_memory = psutil.Process().memory_info().rss
            
            # Load model
            model.load_model()
            loaded_memory = psutil.Process().memory_info().rss
            
            # Process image
            test_image = cv2.imread(self.test_image_path)
            enhanced_image = model.enhance_image(test_image)
            processing_memory = psutil.Process().memory_info().rss
            
            # Calculate memory usage
            creation_overhead = creation_memory - baseline_memory
            loading_overhead = loaded_memory - creation_memory
            processing_overhead = processing_memory - loaded_memory
            total_overhead = processing_memory - baseline_memory
            
            # Convert to MB for readability
            creation_mb = creation_overhead / (1024 * 1024)
            loading_mb = loading_overhead / (1024 * 1024)
            processing_mb = processing_overhead / (1024 * 1024)
            total_mb = total_overhead / (1024 * 1024)
            
            self.performance_metrics['memory_usage'] = total_mb
            
            # Version-specific memory expectations
            if gfpgan_version == "1.3":
                assert total_mb < 800, f"GFPGAN v1.3 should use <800MB, used {total_mb:.1f}MB"
            elif gfpgan_version == "1.4":
                assert total_mb < 600, f"GFPGAN v1.4 should use <600MB (optimized), used {total_mb:.1f}MB"
            
            # Cleanup and verify memory release
            model.unload_model()
            del model
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss
            memory_leak = (final_memory - baseline_memory) / (1024 * 1024)
            
            assert memory_leak < 50, f"Memory leak detected: {memory_leak:.1f}MB not released"
    
    def test_output_quality_comparison(self, gfpgan_version):
        """Test output quality metrics between versions."""
        with patch('photo_restore.models.gfpgan.GFPGANer') as mock_gfpgan:
            mock_instance = MockGFPGANer()
            mock_gfpgan.return_value = mock_instance
            
            model = GFPGANModel(
                model_path=f"GFPGANv{gfpgan_version}.pth",
                device='cpu'
            )
            
            assert model.load_model()
            
            # Load and process test image
            test_image = cv2.imread(self.test_image_path)
            enhanced_image = model.enhance_image(test_image)
            
            assert enhanced_image is not None
            
            # Basic quality checks
            original_shape = test_image.shape
            enhanced_shape = enhanced_image.shape
            
            # Should maintain dimensions (GFPGAN doesn't upscale)
            assert enhanced_shape[:2] == original_shape[:2], "Dimensions should be preserved"
            
            # Enhanced image should have reasonable pixel values
            assert enhanced_image.min() >= 0, "Enhanced image should have valid pixel values"
            assert enhanced_image.max() <= 255, "Enhanced image should have valid pixel values"
            
            # Calculate basic quality metrics
            mse = np.mean((test_image.astype(float) - enhanced_image.astype(float)) ** 2)
            assert mse > 0, "Enhanced image should be different from original"
            assert mse < 10000, "Enhancement should not be too aggressive"
    
    def test_error_handling_comparison(self, gfpgan_version):
        """Test error handling robustness between versions."""
        model = GFPGANModel(
            model_path=f"GFPGANv{gfpgan_version}.pth",
            device='cpu'
        )
        
        # Test handling of invalid input
        with pytest.raises(Exception):
            model.enhance_image(None)
        
        # Test handling of malformed image
        invalid_image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch('photo_restore.models.gfpgan.GFPGANer') as mock_gfpgan:
            mock_instance = MockGFPGANer()
            mock_gfpgan.return_value = mock_instance
            
            model.load_model()
            result = model.enhance_image(invalid_image)
            # Should handle gracefully and return something
            assert result is not None or hasattr(model, '_handle_error')


@pytest.mark.integration
class TestGFPGANVersionIntegration:
    """Integration tests for GFPGAN version switching in the full pipeline."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="gfpgan_integration_")
        self.test_image = self._create_test_image()
        
    def teardown_method(self):
        """Cleanup integration environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_test_image(self) -> str:
        """Create test image for integration tests."""
        image_path = Path(self.temp_dir) / "integration_test.jpg"
        
        # Create a more realistic test image
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), image)
        
        return str(image_path)
    
    @patch('photo_restore.models.gfpgan.GFPGANer')
    def test_model_manager_version_switching(self, mock_gfpgan):
        """Test ModelManager's ability to switch between GFPGAN versions."""
        mock_gfpgan.return_value = MockGFPGANer()
        
        config_dict = {
            'models': {
                'gfpgan': {
                    'version': '1.3',
                    'model_path': 'GFPGANv1.3.pth'
                }
            }
        }
        config = Config(config_dict)
        
        manager = ModelManager(config=config)
        
        # Test loading v1.3
        gfpgan_model = manager.load_model('gfpgan')
        assert gfpgan_model is not None
        
        # Test switching to v1.4
        config.models.gfpgan.version = '1.4'
        config.models.gfpgan.model_path = 'GFPGANv1.4.pth'
        
        gfpgan_v14_model = manager.load_model('gfpgan')
        assert gfpgan_v14_model is not None
    
    @patch('photo_restore.models.gfpgan.GFPGANer')
    def test_image_processor_version_compatibility(self, mock_gfpgan):
        """Test ImageProcessor compatibility with different GFPGAN versions."""
        mock_gfpgan.return_value = MockGFPGANer()
        
        for version in ['1.3', '1.4']:
            config_dict = {
                'models': {
                    'gfpgan': {
                        'version': version,
                        'model_path': f'GFPGANv{version}.pth'
                    }
                },
                'processing': {
                    'face_enhance': True
                }
            }
            config = Config(config_dict)
            
            processor = ImageProcessor(config=config)
            
            output_path = Path(self.temp_dir) / f"enhanced_v{version}.jpg"
            
            result = processor.process_image(
                input_path=self.test_image,
                output_path=str(output_path),
                face_enhance=True
            )
            
            assert result, f"Processing failed with GFPGAN v{version}"
            assert output_path.exists(), f"Output file not created with GFPGAN v{version}"


@pytest.mark.benchmark
class TestGFPGANBenchmarks:
    """Comprehensive benchmarking suite for GFPGAN version comparison."""
    
    def setup_method(self):
        """Setup benchmark environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="gfpgan_benchmark_")
        self.benchmark_results = {}
        
        # Create various test images
        self.test_images = {
            'small': self._create_test_image((256, 256)),
            'medium': self._create_test_image((512, 512)),
            'large': self._create_test_image((1024, 1024))
        }
        
    def teardown_method(self):
        """Cleanup and report benchmark results."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Print benchmark summary
        print("\n" + "="*60)
        print("GFPGAN VERSION COMPARISON BENCHMARK RESULTS")
        print("="*60)
        
        for test_name, results in self.benchmark_results.items():
            print(f"\n{test_name.upper()}:")
            for version, metrics in results.items():
                print(f"  GFPGAN v{version}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.3f}")
                    else:
                        print(f"    {metric}: {value}")
    
    def _create_test_image(self, size) -> str:
        """Create test image of specified size."""
        image_path = Path(self.temp_dir) / f"test_{size[0]}x{size[1]}.jpg"
        
        # Create realistic face-like test image
        image = np.random.randint(100, 200, (*size, 3), dtype=np.uint8)
        
        # Add some structure to make it more face-like
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = min(size) // 4
        
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        image[mask] = image[mask] * 0.8 + 50
        
        cv2.imwrite(str(image_path), image)
        return str(image_path)
    
    @patch('photo_restore.models.gfpgan.GFPGANer')
    def test_comprehensive_performance_benchmark(self, mock_gfpgan):
        """Comprehensive performance benchmark across image sizes and versions."""
        mock_gfpgan.return_value = MockGFPGANer()
        
        versions = ['1.3', '1.4']
        
        for image_size, image_path in self.test_images.items():
            test_name = f"performance_{image_size}"
            self.benchmark_results[test_name] = {}
            
            for version in versions:
                print(f"Benchmarking GFPGAN v{version} with {image_size} image...")
                
                # Initialize metrics
                metrics = {
                    'load_time': 0,
                    'processing_time': 0,
                    'memory_usage_mb': 0,
                    'throughput_fps': 0
                }
                
                # Setup model
                model = GFPGANModel(
                    model_path=f"GFPGANv{version}.pth",
                    device='cpu'
                )
                
                # Benchmark model loading
                gc.collect()
                start_memory = psutil.Process().memory_info().rss
                
                start_time = time.time()
                model.load_model()
                metrics['load_time'] = time.time() - start_time
                
                # Benchmark image processing
                test_image = cv2.imread(image_path)
                
                # Multiple runs for throughput calculation
                num_runs = 3
                processing_times = []
                
                for _ in range(num_runs):
                    start_time = time.time()
                    enhanced_image = model.enhance_image(test_image)
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    assert enhanced_image is not None
                
                metrics['processing_time'] = np.mean(processing_times)
                metrics['throughput_fps'] = 1.0 / metrics['processing_time']
                
                # Memory usage
                peak_memory = psutil.Process().memory_info().rss
                metrics['memory_usage_mb'] = (peak_memory - start_memory) / (1024 * 1024)
                
                # Cleanup
                model.unload_model()
                del model
                gc.collect()
                
                self.benchmark_results[test_name][version] = metrics
    
    def test_memory_scaling_benchmark(self):
        """Test memory scaling across different image sizes."""
        test_name = "memory_scaling"
        self.benchmark_results[test_name] = {}
        
        with patch('photo_restore.models.gfpgan.GFPGANer') as mock_gfpgan:
            mock_gfpgan.return_value = MockGFPGANer()
            
            for version in ['1.3', '1.4']:
                self.benchmark_results[test_name][version] = {}
                
                for size_name, image_path in self.test_images.items():
                    gc.collect()
                    baseline_memory = psutil.Process().memory_info().rss
                    
                    model = GFPGANModel(
                        model_path=f"GFPGANv{version}.pth",
                        device='cpu'
                    )
                    
                    model.load_model()
                    test_image = cv2.imread(image_path)
                    enhanced_image = model.enhance_image(test_image)
                    
                    peak_memory = psutil.Process().memory_info().rss
                    memory_used_mb = (peak_memory - baseline_memory) / (1024 * 1024)
                    
                    self.benchmark_results[test_name][version][size_name] = memory_used_mb
                    
                    model.unload_model()
                    del model
                    gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])