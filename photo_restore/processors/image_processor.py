"""Core image processing engine for photo restoration."""

import logging
import os
import time
import gc
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

from ..models.model_manager import ModelManager
from ..utils.config import Config
from ..utils.logger import PerformanceLogger


class ImageProcessor:
    """Main image processing engine for photo restoration."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """Initialize image processor."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_manager = ModelManager(config, logger)
        
        # Processing statistics
        self.stats = {
            'processed_images': 0,
            'processing_time': 0.0,
            'enhancement_success_rate': 0.0
        }
    
    def process_image(
        self,
        input_path: str,
        output_path: str,
        quality: str = 'balanced',
        upscale: int = 4,
        face_enhance: bool = True,
        output_format: str = 'jpg',
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> bool:
        """
        Process a single image with AI enhancement.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            quality: Quality level (fast/balanced/best)
            upscale: Upscale factor (2 or 4)
            face_enhance: Enable face enhancement
            output_format: Output format (jpg/png)
            progress_callback: Optional progress callback function
            
        Returns:
            Success status
        """
        with PerformanceLogger(self.logger, f"processing {Path(input_path).name}"):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(5)
                
                # Load and validate image
                image = self._load_image(input_path)
                if image is None:
                    self.logger.error(f"Failed to load image: {input_path}")
                    return False
                
                if progress_callback:
                    progress_callback(15)
                
                # Load models based on requirements
                self._ensure_models_loaded(upscale, face_enhance, quality)
                
                if progress_callback:
                    progress_callback(30)
                
                # Apply Real-ESRGAN enhancement
                enhanced_image = self._apply_esrgan_enhancement(image, upscale, quality)
                if enhanced_image is None:
                    self.logger.error("ESRGAN enhancement failed")
                    return False
                
                if progress_callback:
                    progress_callback(70)
                
                # Apply face enhancement if enabled
                if face_enhance:
                    enhanced_image = self._apply_face_enhancement(enhanced_image)
                
                if progress_callback:
                    progress_callback(90)
                
                # Save result
                success = self._save_image(enhanced_image, output_path, output_format)
                
                if progress_callback:
                    progress_callback(100)
                
                if success:
                    self.stats['processed_images'] += 1
                    self.logger.info(f"Successfully processed: {input_path} -> {output_path}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Image processing failed: {str(e)}")
                return False
    
    def _load_image(self, input_path: str) -> Optional[np.ndarray]:
        """Load and validate input image."""
        try:
            # Check file format
            file_ext = Path(input_path).suffix.lower().lstrip('.')
            if file_ext not in self.config.processing.supported_formats:
                self.logger.error(f"Unsupported format: {file_ext}")
                return None
            
            # Load with OpenCV
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if image is None:
                self.logger.error(f"Failed to read image: {input_path}")
                return None
            
            # Check image dimensions
            h, w = image.shape[:2]
            if min(h, w) < 100:
                self.logger.warning(f"Image too small: {w}x{h}")
                return None
            
            if max(h, w) > self.config.processing.max_image_size:
                self.logger.warning(f"Image too large: {w}x{h}, will be resized")
                # Resize if too large
                scale = self.config.processing.max_image_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            self.logger.debug(f"Loaded image: {w}x{h} -> {image.shape[1]}x{image.shape[0]}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {input_path}: {str(e)}")
            return None
    
    def _ensure_models_loaded(self, upscale: int, face_enhance: bool, quality: str = 'balanced') -> None:
        """Ensure required models are loaded with quality-based selection."""
        # Select model variant based on quality setting
        quality_settings = self._get_quality_settings(quality)
        model_variant = quality_settings.get('model_variant', 'standard')
        
        # Load appropriate ESRGAN model
        if not self.model_manager.is_esrgan_loaded():
            if model_variant == 'light' and upscale == 4:
                # Try to load lightweight variant first
                if not self.model_manager.load_esrgan_model(upscale, model_type='light'):
                    # Fallback to standard model
                    self.model_manager.load_esrgan_model(upscale)
            else:
                self.model_manager.load_esrgan_model(upscale)
        
        # Load GFPGAN model if needed
        if face_enhance and not self.model_manager.is_gfpgan_loaded():
            if model_variant == 'light':
                # Try lightweight GFPGAN first
                if not self.model_manager.load_gfpgan_model(model_type='light'):
                    # Fallback to standard model
                    self.model_manager.load_gfpgan_model()
            else:
                self.model_manager.load_gfpgan_model()
    
    def _get_quality_settings(self, quality: str) -> Dict[str, Any]:
        """Get quality-specific settings."""
        if hasattr(self.config.processing, 'quality_settings'):
            return self.config.processing.quality_settings.get(quality, {})
        
        # Fallback defaults
        defaults = {
            'fast': {'tile_size': 256, 'tile_overlap': 10, 'model_variant': 'light'},
            'balanced': {'tile_size': 400, 'tile_overlap': 20, 'model_variant': 'standard'},
            'best': {'tile_size': 512, 'tile_overlap': 32, 'model_variant': 'standard'}
        }
        return defaults.get(quality, defaults['balanced'])
    
    def _apply_esrgan_enhancement(self, image: np.ndarray, upscale: int, quality: str) -> Optional[np.ndarray]:
        """Apply Real-ESRGAN enhancement."""
        try:
            esrgan_model = self.model_manager.get_esrgan_model()
            if esrgan_model is None:
                self.logger.error("ESRGAN model not available")
                return None
            
            # Get quality settings
            quality_settings = self._get_quality_settings(quality)
            tile_size = quality_settings.get('tile_size', 512)
            
            # Apply adaptive tile sizing
            if hasattr(self.model_manager, '_adaptive_tile_size'):
                tile_size = self.model_manager._adaptive_tile_size(tile_size)
            
            # Process with memory-aware approach
            try:
                with torch.no_grad():  # Disable gradient computation for inference
                    if max(image.shape[:2]) > tile_size:
                        enhanced = self._process_with_tiling(image, esrgan_model, tile_size)
                    else:
                        enhanced, _ = esrgan_model.enhance(image, outscale=upscale)
                
                # Clean up memory
                if hasattr(self.model_manager, '_check_memory_usage'):
                    if self.model_manager._check_memory_usage() > 0.8:
                        gc.collect()
                
                return enhanced
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.logger.warning(f"Memory error, retrying with smaller tiles: {str(e)}")
                    # Retry with smaller tile size
                    smaller_tile_size = max(128, tile_size // 2)
                    with torch.no_grad():
                        enhanced = self._process_with_tiling(image, esrgan_model, smaller_tile_size)
                    return enhanced
                else:
                    raise
            
        except Exception as e:
            self.logger.error(f"ESRGAN enhancement failed: {str(e)}")
            return None
    
    def _apply_face_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply GFPGAN face enhancement with CPU optimization."""
        try:
            gfpgan_model = self.model_manager.get_gfpgan_model()
            if gfpgan_model is None:
                self.logger.warning("GFPGAN model not available, skipping face enhancement")
                return image
            
            # Memory-aware face enhancement
            try:
                with torch.no_grad():  # Disable gradient computation
                    # Apply face enhancement with optimized settings
                    _, _, enhanced_image = gfpgan_model.enhance(
                        image, 
                        has_aligned=False,
                        paste_back=True,
                        weight=0.5  # Blend ratio for more natural results
                    )
                
                # Clean up memory if needed
                if hasattr(self.model_manager, '_check_memory_usage'):
                    if self.model_manager._check_memory_usage() > 0.8:
                        gc.collect()
                
                return enhanced_image if enhanced_image is not None else image
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.logger.warning(f"Face enhancement memory error: {str(e)}")
                    # Skip face enhancement on memory error
                    return image
                else:
                    raise
            
        except Exception as e:
            self.logger.warning(f"Face enhancement failed: {str(e)}")
            return image  # Return original if face enhancement fails
    
    def _process_with_tiling(self, image: np.ndarray, model, tile_size: int) -> np.ndarray:
        """Process large image with tiling to manage memory."""
        h, w = image.shape[:2]
        overlap = self.config.processing.tile_overlap
        
        # Calculate tile positions
        tiles = []
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tiles.append((y, y_end, x, x_end))
        
        # Process tiles
        enhanced_tiles = []
        for y, y_end, x, x_end in tiles:
            tile = image[y:y_end, x:x_end]
            enhanced_tile, _ = model.enhance(tile)
            enhanced_tiles.append((enhanced_tile, y, y_end, x, x_end))
        
        # Reconstruct image from tiles
        scale_factor = enhanced_tiles[0][0].shape[0] // (enhanced_tiles[0][2] - enhanced_tiles[0][1])
        enhanced_h, enhanced_w = h * scale_factor, w * scale_factor
        enhanced_image = np.zeros((enhanced_h, enhanced_w, 3), dtype=np.uint8)
        
        for enhanced_tile, y, y_end, x, x_end in enhanced_tiles:
            scaled_y = y * scale_factor
            scaled_y_end = y_end * scale_factor
            scaled_x = x * scale_factor
            scaled_x_end = x_end * scale_factor
            
            enhanced_image[scaled_y:scaled_y_end, scaled_x:scaled_x_end] = enhanced_tile
        
        return enhanced_image
    
    def _save_image(self, image: np.ndarray, output_path: str, format: str) -> bool:
        """Save processed image."""
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert format if needed
            if format.lower() == 'jpg':
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif format.lower() == 'png':
                cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(output_path, image)
            
            # Verify file was created
            if not Path(output_path).exists():
                self.logger.error(f"Output file not created: {output_path}")
                return False
            
            file_size = Path(output_path).stat().st_size
            self.logger.debug(f"Saved image: {output_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save image {output_path}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()