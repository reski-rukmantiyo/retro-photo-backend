"""Performance benchmark tests for CPU processing."""

import pytest
import time
import psutil
import gc
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from photo_restore.processors.image_processor import ImageProcessor
from photo_restore.processors.batch_processor import BatchProcessor
from photo_restore.utils.config import Config
from tests.mocks import MockRealESRGANer, MockGFPGANer


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def performance_config(self):
        """Create performance-optimized configuration."""
        config_dict = {
            'processing': {
                'max_image_size': 2048,
                'memory_limit_gb': 2.0,
                'temp_cleanup': True,
                'tile_size': 256,  # Smaller tiles for testing
                'tile_overlap': 16
            },
            'models': {
                'cache_dir': '/tmp/perf_test_models',
                'download_timeout': 60
            },
            'logging': {
                'level': 'WARNING'  # Reduce logging overhead
            }
        }
        return Config(config_dict)
    
    @pytest.fixture
    def memory_monitor(self):
        """Monitor memory usage during tests."""
        class MemoryMonitor:
            def __init__(self):
                self.initial_memory = None
                self.peak_memory = 0
                self.final_memory = None
            
            def start(self):
                gc.collect()  # Clean up before measuring
                self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = self.initial_memory
                return self
            
            def update(self):
                current = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if current > self.peak_memory:
                    self.peak_memory = current
            
            def stop(self):
                gc.collect()
                self.final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                return {
                    'initial_mb': self.initial_memory,
                    'peak_mb': self.peak_memory,
                    'final_mb': self.final_memory,
                    'max_increase_mb': self.peak_memory - self.initial_memory
                }
        
        return MemoryMonitor()
    
    def create_test_images(self, temp_dir, sizes, count=1):
        """Create test images with specified sizes."""
        images = []
        for size in sizes:
            for i in range(count):
                # Create test image with some pattern
                image_data = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                
                # Add some structured content to make it more realistic
                h, w = size[1], size[0]
                # Add gradient
                for y in range(h):
                    for x in range(w):
                        image_data[y, x, 0] = min(255, int(255 * x / w))  # Red gradient
                        image_data[y, x, 1] = min(255, int(255 * y / h))  # Green gradient
                
                pil_image = Image.fromarray(image_data)
                image_path = temp_dir / f"test_{size[0]}x{size[1]}_{i}.jpg"
                pil_image.save(image_path, "JPEG", quality=90)
                images.append(image_path)
        
        return images
    
    def test_small_image_processing_performance(self, performance_config, temp_dir, memory_monitor):
        """Test performance with small images (256x256)."""
        # Create small test images
        test_image = self.create_test_images(temp_dir, [(256, 256)])[0]
        output_path = temp_dir / "small_output.jpg"
        
        # Mock model manager for consistent performance
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)  # Use 2x for faster processing
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(performance_config)
            
            # Benchmark processing
            memory_monitor.start()
            start_time = time.time()
            
            success = processor.process_image(
                input_path=str(test_image),
                output_path=str(output_path),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            
            end_time = time.time()
            memory_stats = memory_monitor.stop()
            
            processing_time = end_time - start_time
            
            # Performance assertions
            assert success is True
            assert processing_time < 5.0, f"Small image processing took {processing_time:.2f}s, expected < 5.0s"
            assert memory_stats['max_increase_mb'] < 200, f"Memory increase {memory_stats['max_increase_mb']:.1f}MB too high"
            
            print(f"\nSmall image (256x256) performance:")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Memory increase: {memory_stats['max_increase_mb']:.1f}MB")
    
    def test_medium_image_processing_performance(self, performance_config, temp_dir, memory_monitor):
        """Test performance with medium images (1024x1024)."""
        test_image = self.create_test_images(temp_dir, [(1024, 1024)])[0]
        output_path = temp_dir / "medium_output.jpg"
        
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(performance_config)
            
            memory_monitor.start()
            start_time = time.time()
            
            success = processor.process_image(
                input_path=str(test_image),
                output_path=str(output_path),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            
            end_time = time.time()
            memory_stats = memory_monitor.stop()
            
            processing_time = end_time - start_time
            
            assert success is True
            assert processing_time < 15.0, f"Medium image processing took {processing_time:.2f}s"
            assert memory_stats['max_increase_mb'] < 500, f"Memory increase {memory_stats['max_increase_mb']:.1f}MB too high"
            
            print(f"\nMedium image (1024x1024) performance:")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Memory increase: {memory_stats['max_increase_mb']:.1f}MB")
    
    def test_large_image_tiling_performance(self, performance_config, temp_dir, memory_monitor):
        """Test performance with large images requiring tiling (2048x2048)."""
        test_image = self.create_test_images(temp_dir, [(2048, 2048)])[0]
        output_path = temp_dir / "large_output.jpg"
        
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(performance_config)
            
            memory_monitor.start()
            start_time = time.time()
            
            success = processor.process_image(
                input_path=str(test_image),
                output_path=str(output_path),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            
            end_time = time.time()
            memory_stats = memory_monitor.stop()
            
            processing_time = end_time - start_time
            
            assert success is True
            assert processing_time < 30.0, f"Large image processing took {processing_time:.2f}s"
            # Large images use tiling, so memory should still be controlled
            assert memory_stats['max_increase_mb'] < 800, f"Memory increase {memory_stats['max_increase_mb']:.1f}MB too high"
            
            print(f"\nLarge image (2048x2048) with tiling performance:")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Memory increase: {memory_stats['max_increase_mb']:.1f}MB")
    
    def test_batch_processing_performance(self, performance_config, temp_dir, memory_monitor):
        """Test batch processing performance."""
        # Create multiple test images of varying sizes
        test_images = []
        test_images.extend(self.create_test_images(temp_dir, [(256, 256)], count=3))
        test_images.extend(self.create_test_images(temp_dir, [(512, 512)], count=2))
        
        output_dir = temp_dir / "batch_output"
        
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            batch_processor = BatchProcessor(performance_config)
            batch_processor.image_processor.model_manager = mock_manager
            
            memory_monitor.start()
            start_time = time.time()
            
            success_count = batch_processor.process_directory(
                input_dir=str(temp_dir),
                output_dir=str(output_dir),
                quality='fast',
                upscale=2,
                face_enhance=False,
                recursive=False
            )
            
            end_time = time.time()
            memory_stats = memory_monitor.stop()
            
            processing_time = end_time - start_time
            avg_time_per_image = processing_time / len(test_images)
            
            assert success_count == len(test_images)
            assert processing_time < 25.0, f"Batch processing took {processing_time:.2f}s"
            assert avg_time_per_image < 5.0, f"Average time per image {avg_time_per_image:.2f}s too high"
            
            print(f"\nBatch processing ({len(test_images)} images) performance:")
            print(f"  Total time: {processing_time:.2f}s")
            print(f"  Average per image: {avg_time_per_image:.2f}s")
            print(f"  Memory increase: {memory_stats['max_increase_mb']:.1f}MB")
    
    def test_memory_efficiency_multiple_runs(self, performance_config, temp_dir, memory_monitor):
        """Test memory efficiency across multiple processing runs."""
        test_image = self.create_test_images(temp_dir, [(512, 512)])[0]
        
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(performance_config)
            
            memory_monitor.start()
            
            # Run multiple processing cycles
            for i in range(5):
                output_path = temp_dir / f"output_{i}.jpg"
                
                success = processor.process_image(
                    input_path=str(test_image),
                    output_path=str(output_path),
                    quality='fast',
                    upscale=2,
                    face_enhance=False
                )
                
                assert success is True
                memory_monitor.update()
                
                # Force garbage collection between runs
                gc.collect()
            
            memory_stats = memory_monitor.stop()
            
            # Memory should not grow significantly across runs
            assert memory_stats['max_increase_mb'] < 300, f"Memory increase {memory_stats['max_increase_mb']:.1f}MB indicates memory leak"
            
            print(f"\nMultiple runs memory efficiency:")
            print(f"  Max memory increase: {memory_stats['max_increase_mb']:.1f}MB")
    
    def test_tiling_efficiency(self, performance_config, temp_dir, memory_monitor):
        """Test that tiling maintains memory efficiency."""
        # Create image larger than tile size to force tiling
        test_image = self.create_test_images(temp_dir, [(1024, 1024)])[0]
        output_path = temp_dir / "tiled_output.jpg"
        
        # Use small tile size to force more tiles
        performance_config.processing.tile_size = 256
        
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            
            # Create mock that tracks tile processing
            tile_calls = []
            
            class TileTrackingESRGAN(MockRealESRGANer):
                def enhance(self, img, outscale=None):
                    tile_calls.append(img.shape)
                    return super().enhance(img, outscale)
            
            mock_esrgan = TileTrackingESRGAN(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(performance_config)
            
            memory_monitor.start()
            start_time = time.time()
            
            success = processor.process_image(
                input_path=str(test_image),
                output_path=str(output_path),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            
            end_time = time.time()
            memory_stats = memory_monitor.stop()
            
            processing_time = end_time - start_time
            
            assert success is True
            assert len(tile_calls) > 1, "Tiling should have processed multiple tiles"
            
            # Each tile should be reasonable size
            for tile_shape in tile_calls:
                assert tile_shape[0] <= 288, f"Tile height {tile_shape[0]} too large"  # 256 + overlap
                assert tile_shape[1] <= 288, f"Tile width {tile_shape[1]} too large"
            
            # Memory should be controlled despite large image
            assert memory_stats['max_increase_mb'] < 400, f"Tiling memory increase {memory_stats['max_increase_mb']:.1f}MB too high"
            
            print(f"\nTiling efficiency (1024x1024 → {len(tile_calls)} tiles):")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Memory increase: {memory_stats['max_increase_mb']:.1f}MB")
            print(f"  Tiles processed: {len(tile_calls)}")
    
    def test_cpu_utilization_efficiency(self, performance_config, temp_dir):
        """Test CPU utilization during processing."""
        test_image = self.create_test_images(temp_dir, [(512, 512)])[0]
        output_path = temp_dir / "cpu_test_output.jpg"
        
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            
            # Create CPU-intensive mock model
            class CPUIntensiveESRGAN(MockRealESRGANer):
                def enhance(self, img, outscale=None):
                    # Simulate CPU-intensive work
                    start_time = time.time()
                    while time.time() - start_time < 0.1:  # 100ms of work
                        _ = np.random.random((100, 100)).sum()
                    return super().enhance(img, outscale)
            
            mock_esrgan = CPUIntensiveESRGAN(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(performance_config)
            
            # Monitor CPU usage
            initial_cpu = psutil.cpu_percent(interval=0.1)
            
            start_time = time.time()
            
            success = processor.process_image(
                input_path=str(test_image),
                output_path=str(output_path),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            
            end_time = time.time()
            final_cpu = psutil.cpu_percent(interval=0.1)
            
            processing_time = end_time - start_time
            
            assert success is True
            
            print(f"\nCPU utilization efficiency:")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  CPU usage during processing: ~{final_cpu:.1f}%")
    
    @pytest.mark.slow
    def test_scalability_benchmark(self, performance_config, temp_dir, memory_monitor):
        """Test performance scalability with increasing load."""
        # Test with increasing numbers of images
        batch_sizes = [1, 3, 5, 10]
        results = []
        
        for batch_size in batch_sizes:
            # Create test images
            test_images = self.create_test_images(temp_dir / f"batch_{batch_size}", [(256, 256)], count=batch_size)
            output_dir = temp_dir / f"output_{batch_size}"
            
            with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
                mock_manager = MagicMock()
                mock_esrgan = MockRealESRGANer(scale=2)
                mock_manager.get_esrgan_model.return_value = mock_esrgan
                mock_manager.is_esrgan_loaded.return_value = True
                mock_manager.is_gfpgan_loaded.return_value = False
                mock_manager_class.return_value = mock_manager
                
                batch_processor = BatchProcessor(performance_config)
                batch_processor.image_processor.model_manager = mock_manager
                
                memory_monitor.start()
                start_time = time.time()
                
                success_count = batch_processor.process_directory(
                    input_dir=str(temp_dir / f"batch_{batch_size}"),
                    output_dir=str(output_dir),
                    quality='fast',
                    upscale=2,
                    face_enhance=False
                )
                
                end_time = time.time()
                memory_stats = memory_monitor.stop()
                
                processing_time = end_time - start_time
                avg_time = processing_time / batch_size
                
                results.append({
                    'batch_size': batch_size,
                    'total_time': processing_time,
                    'avg_time': avg_time,
                    'memory_mb': memory_stats['max_increase_mb'],
                    'success_count': success_count
                })
                
                assert success_count == batch_size
        
        # Analyze scalability
        print(f"\nScalability benchmark results:")
        for result in results:
            print(f"  {result['batch_size']} images: {result['total_time']:.2f}s total, "
                  f"{result['avg_time']:.2f}s avg, {result['memory_mb']:.1f}MB memory")
        
        # Check that average time per image doesn't increase significantly
        avg_times = [r['avg_time'] for r in results]
        assert max(avg_times) / min(avg_times) < 2.0, "Performance degradation with scale"


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests."""
    
    def test_baseline_performance_regression(self, performance_config, temp_dir):
        """Test baseline performance to catch regressions."""
        # Define baseline expectations (adjust based on typical hardware)
        BASELINE_TIMES = {
            'small_image': 2.0,   # 256x256 should process in < 2s
            'medium_image': 8.0,  # 512x512 should process in < 8s
            'batch_5_images': 15.0  # 5 small images should process in < 15s
        }
        
        results = {}
        
        # Test small image
        small_image = self.create_test_image(temp_dir, (256, 256), "small.jpg")
        results['small_image'] = self.benchmark_single_image(performance_config, small_image, temp_dir / "small_out.jpg")
        
        # Test medium image
        medium_image = self.create_test_image(temp_dir, (512, 512), "medium.jpg")
        results['medium_image'] = self.benchmark_single_image(performance_config, medium_image, temp_dir / "medium_out.jpg")
        
        # Test batch processing
        batch_dir = temp_dir / "batch"
        batch_dir.mkdir()
        for i in range(5):
            self.create_test_image(batch_dir, (256, 256), f"batch_{i}.jpg")
        results['batch_5_images'] = self.benchmark_batch_processing(performance_config, batch_dir, temp_dir / "batch_out")
        
        # Check against baselines
        for test_name, processing_time in results.items():
            baseline = BASELINE_TIMES[test_name]
            assert processing_time < baseline, f"{test_name} took {processing_time:.2f}s, baseline is {baseline}s"
            
            print(f"{test_name}: {processing_time:.2f}s (baseline: {baseline}s)")
    
    def create_test_image(self, directory, size, filename):
        """Create a single test image."""
        image_data = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_data)
        image_path = directory / filename
        pil_image.save(image_path, "JPEG", quality=90)
        return image_path
    
    def benchmark_single_image(self, config, input_path, output_path):
        """Benchmark single image processing."""
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            processor = ImageProcessor(config)
            
            start_time = time.time()
            success = processor.process_image(
                input_path=str(input_path),
                output_path=str(output_path),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            end_time = time.time()
            
            assert success is True
            return end_time - start_time
    
    def benchmark_batch_processing(self, config, input_dir, output_dir):
        """Benchmark batch processing."""
        with patch('photo_restore.processors.image_processor.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_esrgan = MockRealESRGANer(scale=2)
            mock_manager.get_esrgan_model.return_value = mock_esrgan
            mock_manager.is_esrgan_loaded.return_value = True
            mock_manager.is_gfpgan_loaded.return_value = False
            mock_manager_class.return_value = mock_manager
            
            batch_processor = BatchProcessor(config)
            batch_processor.image_processor.model_manager = mock_manager
            
            start_time = time.time()
            success_count = batch_processor.process_directory(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                quality='fast',
                upscale=2,
                face_enhance=False
            )
            end_time = time.time()
            
            assert success_count > 0
            return end_time - start_time