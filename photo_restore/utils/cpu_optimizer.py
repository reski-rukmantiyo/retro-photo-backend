"""CPU-specific optimization utilities for photo restoration."""

import gc
import logging
import os
import psutil
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

import numpy as np
import torch


class CPUOptimizer:
    """CPU optimization utilities for memory management and performance."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize CPU optimizer."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # CPU configuration
        self.cpu_config = config.get('cpu_optimization', {})
        self.memory_threshold = self.cpu_config.get('memory_cleanup_threshold', 0.8)
        
        # Setup optimizations
        self._setup_cpu_threads()
        self._setup_memory_monitoring()
    
    def _setup_cpu_threads(self) -> None:
        """Configure CPU thread limits."""
        try:
            # Set PyTorch threads
            torch_threads = self.cpu_config.get('torch_threads', 2)
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(torch_threads)
            
            # Set NumPy threads
            numpy_threads = self.cpu_config.get('numpy_threads', 2)
            os.environ['OMP_NUM_THREADS'] = str(numpy_threads)
            os.environ['MKL_NUM_THREADS'] = str(numpy_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(numpy_threads)
            
            # Disable CUDA if available
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            self.logger.info(f"CPU threads configured: PyTorch={torch_threads}, NumPy={numpy_threads}")
            
        except Exception as e:
            self.logger.warning(f"CPU thread setup failed: {str(e)}")
    
    def _setup_memory_monitoring(self) -> None:
        """Setup memory monitoring."""
        self.memory_stats = {
            'peak_usage': 0.0,
            'cleanup_count': 0,
            'last_cleanup': time.time()
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        memory_usage = self.get_memory_usage()
        return memory_usage['percent'] / 100.0 > self.memory_threshold
    
    def cleanup_memory(self, force: bool = False) -> bool:
        """Perform memory cleanup."""
        if not force and not self.check_memory_pressure():
            return False
        
        try:
            # Python garbage collection
            collected = gc.collect()
            
            # PyTorch cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force memory release
            gc.set_threshold(0)
            gc.collect()
            gc.set_threshold(700, 10, 10)  # Reset to defaults
            
            # Update stats
            self.memory_stats['cleanup_count'] += 1
            self.memory_stats['last_cleanup'] = time.time()
            
            memory_after = self.get_memory_usage()
            self.logger.debug(f"Memory cleanup: collected {collected} objects, "
                            f"memory usage: {memory_after['percent']:.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {str(e)}")
            return False
    
    @contextmanager
    def memory_context(self, cleanup_after: bool = True):
        """Context manager for memory-aware operations."""
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            if cleanup_after:
                self.cleanup_memory()
            
            memory_after = self.get_memory_usage()
            duration = time.time() - start_time
            
            # Track peak usage
            peak_usage = max(memory_before['percent'], memory_after['percent'])
            self.memory_stats['peak_usage'] = max(self.memory_stats['peak_usage'], peak_usage)
            
            self.logger.debug(f"Memory context: {duration:.2f}s, "
                            f"peak: {peak_usage:.1f}%, "
                            f"final: {memory_after['percent']:.1f}%")
    
    def optimize_numpy_arrays(self, *arrays: np.ndarray) -> None:
        """Optimize NumPy arrays for memory efficiency."""
        for arr in arrays:
            if arr is not None and hasattr(arr, 'flags'):
                # Set arrays to be more memory efficient
                if arr.flags.writeable:
                    arr.flags.writeable = False  # Make read-only if possible
    
    def get_optimal_tile_size(self, 
                            image_shape: tuple, 
                            base_tile_size: int = 512,
                            memory_factor: float = 0.8) -> int:
        """Calculate optimal tile size based on available memory and image size."""
        memory_info = self.get_memory_usage()
        available_memory_gb = memory_info['available_gb']
        
        # Estimate memory needed per pixel (rough approximation)
        # Factors: input image (3 channels) + processed image (3 channels) + model overhead
        memory_per_pixel_mb = 0.024  # 24 bytes per pixel (8 bytes per channel * 3 channels)
        
        # Calculate maximum tile size based on available memory
        available_memory_mb = available_memory_gb * 1024 * memory_factor
        max_pixels = int(available_memory_mb / memory_per_pixel_mb)
        max_tile_size = int(np.sqrt(max_pixels))
        
        # Choose conservative tile size
        optimal_tile_size = min(base_tile_size, max_tile_size)
        
        # Ensure minimum tile size
        optimal_tile_size = max(128, optimal_tile_size)
        
        # Adjust for high memory pressure
        if self.check_memory_pressure():
            optimal_tile_size = int(optimal_tile_size * 0.7)
        
        self.logger.debug(f"Optimal tile size: {optimal_tile_size} "
                         f"(base: {base_tile_size}, max: {max_tile_size}, "
                         f"available memory: {available_memory_gb:.1f}GB)")
        
        return optimal_tile_size
    
    def warmup_models(self, models: Dict[str, Any]) -> None:
        """Warm up models for better performance."""
        self.logger.info("Warming up models...")
        
        try:
            with self.memory_context():
                for model_name, model in models.items():
                    if model is not None:
                        # Create dummy input for warmup
                        if 'esrgan' in model_name.lower():
                            dummy_input = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                            with torch.no_grad():
                                _ = model.enhance(dummy_input, outscale=2)
                        elif 'gfpgan' in model_name.lower():
                            dummy_input = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                            with torch.no_grad():
                                _ = model.enhance(dummy_input)
                        
                        self.logger.debug(f"Warmed up {model_name}")
            
            self.logger.info("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and memory statistics."""
        memory_info = self.get_memory_usage()
        
        return {
            'memory': memory_info,
            'memory_stats': self.memory_stats.copy(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'torch_threads': torch.get_num_threads(),
            'optimization_config': self.cpu_config
        }


class TileProcessor:
    """Optimized tile processing for large images."""
    
    def __init__(self, optimizer: CPUOptimizer, logger: Optional[logging.Logger] = None):
        """Initialize tile processor."""
        self.optimizer = optimizer
        self.logger = logger or logging.getLogger(__name__)
    
    def process_with_tiles(self, 
                          image: np.ndarray,
                          process_fn: callable,
                          tile_size: int = 512,
                          overlap: int = 32,
                          **kwargs) -> np.ndarray:
        """
        Process large image with overlapping tiles.
        
        Args:
            image: Input image
            process_fn: Processing function to apply to each tile
            tile_size: Size of each tile
            overlap: Overlap between tiles
            **kwargs: Additional arguments for process_fn
            
        Returns:
            Processed image
        """
        h, w = image.shape[:2]
        
        # Use optimizer to get optimal tile size
        optimal_tile_size = self.optimizer.get_optimal_tile_size(
            image.shape, tile_size
        )
        
        if optimal_tile_size != tile_size:
            self.logger.info(f"Adjusted tile size: {tile_size} -> {optimal_tile_size}")
            tile_size = optimal_tile_size
        
        # Calculate tiles
        tiles = self._calculate_tiles(h, w, tile_size, overlap)
        
        self.logger.info(f"Processing {len(tiles)} tiles of size {tile_size}x{tile_size}")
        
        # Process tiles with memory management
        processed_tiles = []
        
        with self.optimizer.memory_context():
            for i, (y1, y2, x1, x2) in enumerate(tiles):
                # Extract tile
                tile = image[y1:y2, x1:x2]
                
                # Process tile
                try:
                    processed_tile = process_fn(tile, **kwargs)
                    processed_tiles.append((processed_tile, y1, y2, x1, x2))
                    
                    # Cleanup memory periodically
                    if i % 4 == 0:  # Every 4 tiles
                        self.optimizer.cleanup_memory()
                        
                except Exception as e:
                    self.logger.error(f"Failed to process tile {i}: {str(e)}")
                    # Use original tile as fallback
                    processed_tiles.append((tile, y1, y2, x1, x2))
        
        # Reconstruct image
        return self._reconstruct_from_tiles(processed_tiles, h, w, overlap)
    
    def _calculate_tiles(self, height: int, width: int, 
                        tile_size: int, overlap: int) -> list:
        """Calculate tile coordinates."""
        tiles = []
        step = tile_size - overlap
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                y1 = y
                x1 = x
                y2 = min(y + tile_size, height)
                x2 = min(x + tile_size, width)
                
                # Skip very small tiles
                if (y2 - y1) > overlap and (x2 - x1) > overlap:
                    tiles.append((y1, y2, x1, x2))
        
        return tiles
    
    def _reconstruct_from_tiles(self, tiles: list, height: int, width: int, 
                               overlap: int) -> np.ndarray:
        """Reconstruct image from processed tiles."""
        if not tiles:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Determine output dimensions
        first_tile = tiles[0][0]
        scale_factor = first_tile.shape[0] // (tiles[0][2] - tiles[0][1])
        
        out_height = height * scale_factor
        out_width = width * scale_factor
        
        # Create output image
        result = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        weight_map = np.zeros((out_height, out_width), dtype=np.float32)
        
        # Blend tiles
        for tile, y1, y2, x1, x2 in tiles:
            # Scale coordinates
            sy1, sy2 = y1 * scale_factor, y2 * scale_factor
            sx1, sx2 = x1 * scale_factor, x2 * scale_factor
            
            # Create weight mask for smooth blending
            tile_h, tile_w = tile.shape[:2]
            weight = np.ones((tile_h, tile_w), dtype=np.float32)
            
            # Apply feathering at edges
            if overlap > 0:
                fade = min(overlap * scale_factor, tile_h // 4, tile_w // 4)
                if fade > 0:
                    weight[:fade, :] *= np.linspace(0, 1, fade)[:, None]
                    weight[-fade:, :] *= np.linspace(1, 0, fade)[:, None]
                    weight[:, :fade] *= np.linspace(0, 1, fade)[None, :]
                    weight[:, -fade:] *= np.linspace(1, 0, fade)[None, :]
            
            # Blend tile into result
            for c in range(3):
                result[sy1:sy2, sx1:sx2, c] += (tile[:, :, c] * weight).astype(np.uint8)
            weight_map[sy1:sy2, sx1:sx2] += weight
        
        # Normalize by weight
        weight_map = np.maximum(weight_map, 1e-6)
        for c in range(3):
            result[:, :, c] = (result[:, :, c] / weight_map).astype(np.uint8)
        
        return result