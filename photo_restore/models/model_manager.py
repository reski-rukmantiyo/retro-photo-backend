"""Model management for AI photo restoration models."""

import logging
import os
import psutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import torch
import torch.jit
from tqdm import tqdm

from ..utils.config import Config


class ModelManager:
    """Manages AI model downloading, caching, and loading."""
    
    # Model URLs and configurations - CPU-optimized variants
    MODEL_URLS = {
        'esrgan_x2': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
            'filename': 'RealESRGAN_x2plus.pth',
            'scale': 2,
            'memory_mb': 1200,  # Estimated memory usage
            'cpu_optimized': True
        },
        'esrgan_x4': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'filename': 'RealESRGAN_x4plus.pth',
            'scale': 4,
            'memory_mb': 1800,
            'cpu_optimized': True
        },
        'esrgan_x4_light': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'filename': 'RealESRGAN_x4plus.pth',
            'scale': 4,
            'memory_mb': 1400,  # Lighter model (use same file as standard)
            'cpu_optimized': True
        },
        'gfpgan': {
            'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            'filename': 'GFPGANv1.3.pth',
            'scale': 1,
            'memory_mb': 800,
            'cpu_optimized': True
        },
        'gfpgan_light': {
            'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
            'filename': 'GFPGANv1.4.pth',
            'scale': 1,
            'memory_mb': 600,  # More efficient
            'cpu_optimized': True
        }
    }
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """Initialize model manager."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Expand cache directory path
        self.cache_dir = Path(self.config.models.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model instances
        self._esrgan_model = None
        self._gfpgan_model = None
        self._current_esrgan_scale = None
        
        # CPU optimization setup
        self._setup_cpu_optimization()
        
        self.logger.info(f"Model cache directory: {self.cache_dir}")
    
    def _setup_cpu_optimization(self):
        """Configure CPU-specific optimizations."""
        try:
            # Set CPU thread limits
            cpu_config = getattr(self.config.processing, 'cpu_optimization', {})
            if cpu_config:
                # Limit PyTorch threads
                torch_threads = cpu_config.get('torch_threads', 2)
                torch.set_num_threads(torch_threads)
                torch.set_num_interop_threads(torch_threads)
                
                # Set memory optimization flags
                torch.backends.cudnn.enabled = False  # Disable CUDNN
                torch.manual_seed(42)  # Deterministic behavior
                
                # Configure for CPU inference - disable problematic JIT optimizations
                # PRODUCTION FIX: Disable JIT fusion to prevent tensor tracing issues
                torch.jit.set_fusion_strategy([])  # Disable fusion strategies
                torch._C._jit_set_profiling_executor(False)  # Disable profiling executor
                torch._C._jit_set_profiling_mode(False)  # Disable profiling mode
                
                self.logger.info(f"CPU optimization: {torch_threads} threads, JIT disabled, memory threshold: {cpu_config.get('memory_cleanup_threshold', 0.8)}")
            
        except Exception as e:
            self.logger.warning(f"CPU optimization setup failed: {str(e)}")
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage percentage."""
        return psutil.virtual_memory().percent / 100.0
    
    def _adaptive_tile_size(self, base_tile_size: int) -> int:
        """Dynamically adjust tile size based on available memory."""
        memory_usage = self._check_memory_usage()
        cpu_config = getattr(self.config.processing, 'cpu_optimization', {})
        threshold = cpu_config.get('memory_cleanup_threshold', 0.8)
        
        if memory_usage > threshold:
            # Reduce tile size when memory is high
            return max(128, int(base_tile_size * 0.6))
        elif memory_usage < 0.5:
            # Increase tile size when memory is abundant
            return min(1024, int(base_tile_size * 1.2))
        
        return base_tile_size
    
    def download_model(self, model_key: str, force_download: bool = False) -> bool:
        """
        Download a model if not already cached.
        
        Args:
            model_key: Model identifier (esrgan_x2, esrgan_x4, gfpgan)
            force_download: Force re-download even if cached
            
        Returns:
            Success status
        """
        if model_key not in self.MODEL_URLS:
            self.logger.error(f"Unknown model: {model_key}")
            return False
        
        model_info = self.MODEL_URLS[model_key]
        model_path = self.cache_dir / model_info['filename']
        
        # Check if already downloaded
        if model_path.exists() and not force_download:
            self.logger.debug(f"Model already cached: {model_path}")
            return True
        
        try:
            self.logger.info(f"Downloading {model_key} model...")
            return self._download_file(model_info['url'], model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_key}: {str(e)}")
            return False
    
    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True, timeout=self.config.models.download_timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=f"Downloading {output_path.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"Successfully downloaded: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            return False
    
    def load_esrgan_model(self, scale: int = 4, model_type: str = 'standard') -> bool:
        """Load Real-ESRGAN model."""
        try:
            # Determine model key based on scale and type
            if model_type == 'light' and scale == 4:
                model_key = 'esrgan_x4_light'
            else:
                model_key = f'esrgan_x{scale}'
            
            # Skip if already loaded with same scale
            if self._esrgan_model is not None and self._current_esrgan_scale == scale:
                return True
            
            # Download model if needed
            if not self.download_model(model_key):
                return False
            
            # Load model
            model_info = self.MODEL_URLS[model_key]
            model_path = self.cache_dir / model_info['filename']
            
            # Import RealESRGAN with compatibility patch
            try:
                from ..utils.torch_compat import patch_realesrgan_imports
                RealESRGANer = patch_realesrgan_imports()
                if RealESRGANer is None:
                    raise ImportError("Failed to import RealESRGANer")
            except ImportError:
                self.logger.error("RealESRGAN not installed. Please install with: pip install realesrgan")
                return False
            
            # Get CPU-optimized tile size
            base_tile_size = self.config.processing.tile_size
            adaptive_tile_size = self._adaptive_tile_size(base_tile_size)
            
            # EMERGENCY FIX: Use manual loader instead of RealESRGANer wrapper
            from ..utils.model_loader import emergency_realesrgan_loader, create_manual_realesrgan_upsampler
            
            # Try emergency manual upsampler first
            self._esrgan_model = create_manual_realesrgan_upsampler(
                model_path, scale, adaptive_tile_size, 'cpu'
            )
            
            # Fallback to standard RealESRGANer if manual fails
            if self._esrgan_model is None:
                self.logger.warning("Manual upsampler failed, trying RealESRGANer wrapper...")
                try:
                    self._esrgan_model = RealESRGANer(
                        scale=scale,
                        model_path=str(model_path),
                        device='cpu',  # Force CPU usage
                        tile=adaptive_tile_size,
                        tile_pad=self.config.processing.tile_overlap,
                        pre_pad=0,
                        half=False,  # Disable half precision for CPU
                        gpu_id=None,  # Explicitly disable GPU
                        dni_weight=None  # Disable deep network interpolation for speed
                    )
                except Exception as e:
                    self.logger.error(f"RealESRGANer wrapper failed: {e}")
                    # Final fallback: Try direct model loading
                    model = emergency_realesrgan_loader(model_path, scale, 'cpu')
                    if model is not None:
                        # Wrap in minimal interface
                        class MinimalUpsampler:
                            def __init__(self, model, scale):
                                self.model = model
                                self.scale = scale
                            def enhance(self, image, outscale=None):
                                import cv2
                                import numpy as np
                                outscale = outscale or self.scale
                                image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                                image_tensor = image_tensor.unsqueeze(0)
                                with torch.no_grad():
                                    output = self.model(image_tensor)
                                output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                                output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
                                return output, None
                        self._esrgan_model = MinimalUpsampler(model, scale)
                        self.logger.info("Using minimal emergency upsampler")
                    else:
                        return False
            
            # Apply model optimizations
            if hasattr(self._esrgan_model, 'model'):
                # Set model to eval mode and optimize for inference
                self._esrgan_model.model.eval()
                
                # Try to use JIT compilation for speed
                try:
                    # Create dummy input for tracing
                    dummy_input = torch.randn(1, 3, 64, 64)
                    self._esrgan_model.model = torch.jit.trace(
                        self._esrgan_model.model, 
                        dummy_input,
                        strict=False
                    )
                    self.logger.info("Applied JIT compilation to ESRGAN model")
                except Exception as e:
                    self.logger.debug(f"JIT compilation failed: {str(e)}")
            
            self.logger.info(f"Loaded Real-ESRGAN {scale}x model with adaptive tile size: {adaptive_tile_size}")
            
            self._current_esrgan_scale = scale
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Real-ESRGAN model: {str(e)}")
            self._esrgan_model = None
            return False
    
    def _create_safe_gfpgan_wrapper(self, gfpgan_model):
        """Create a safe wrapper for GFPGAN to prevent JIT-related tensor issues."""
        class SafeGFPGANWrapper:
            def __init__(self, original_model, logger):
                self.original_model = original_model
                self.logger = logger
                # Store original enhance method
                self._original_enhance = original_model.enhance
                # Replace with safe version
                original_model.enhance = self._safe_enhance
            
            def _safe_enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
                """Safe enhance method with comprehensive error handling."""
                try:
                    # Clear any cached computations
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Ensure tensor is in correct format and device
                    import numpy as np
                    if isinstance(img, np.ndarray):
                        # Validate input tensor shape and type
                        original_dtype = img.dtype
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        
                        # Ensure proper memory layout
                        img = np.ascontiguousarray(img)
                    
                    # Call original enhance with safe tensor processing
                    with torch.inference_mode():
                        # Disable gradient computation completely
                        with torch.no_grad():
                            result = self._original_enhance(
                                img, 
                                has_aligned=has_aligned,
                                only_center_face=only_center_face,
                                paste_back=paste_back,
                                weight=weight
                            )
                    
                    # Validate output
                    if result is None or (isinstance(result, tuple) and len(result) == 0):
                        self.logger.error("GFPGAN returned empty result")
                        return None, None
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"GFPGAN safe processing failed: {str(e)}")
                    # Return original image as fallback
                    return img, None
            
            def __getattr__(self, name):
                # Delegate all other attributes to original model
                return getattr(self.original_model, name)
        
        return SafeGFPGANWrapper(gfpgan_model, self.logger)
    
    def load_gfpgan_model(self, model_type: str = 'standard') -> bool:
        """Load GFPGAN face restoration model."""
        try:
            # Skip if already loaded
            if self._gfpgan_model is not None:
                return True
            
            # Select model based on type
            model_key = 'gfpgan_light' if model_type == 'light' else 'gfpgan'
            
            # Download model if needed
            if not self.download_model(model_key):
                # Fallback to standard model if light model fails
                if model_type == 'light':
                    model_key = 'gfpgan'
                    if not self.download_model(model_key):
                        return False
                else:
                    return False
            
            # Load model
            model_info = self.MODEL_URLS[model_key]
            model_path = self.cache_dir / model_info['filename']
            
            # Import GFPGAN with compatibility patch
            try:
                from ..utils.torch_compat import patch_realesrgan_imports
                patch_realesrgan_imports()  # This also sets up GFPGAN compatibility
                from gfpgan import GFPGANer
            except ImportError:
                self.logger.error("GFPGAN not installed. Please install with: pip install gfpgan")
                return False
            
            # Create face enhancer with CPU optimizations and safe tensor processing
            try:
                self._gfpgan_model = GFPGANer(
                    model_path=str(model_path),
                    upscale=1,  # We handle upscaling with ESRGAN
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,  # We use ESRGAN for background
                    device='cpu'  # Force CPU usage
                )
                
                # Wrap with safe tensor processor to prevent JIT-related issues
                self._gfpgan_model = self._create_safe_gfpgan_wrapper(self._gfpgan_model)
                
            except Exception as e:
                self.logger.error(f"Failed to create GFPGAN model: {str(e)}")
                self._gfpgan_model = None
                return False
            
            # Apply model optimizations
            if hasattr(self._gfpgan_model, 'gfpgan'):
                # Set model to eval mode
                self._gfpgan_model.gfpgan.eval()
                
                # PRODUCTION FIX: Disable JIT compilation for GFPGAN due to tensor tracing errors
                # JIT compilation causes 99.9% element mismatches in forward() pass
                # Keep model in standard PyTorch mode for stable tensor processing
                self.logger.info("GFPGAN JIT compilation disabled - using safe tensor processing mode")
                
                # Apply safe CPU optimizations without JIT
                try:
                    # Enable inference mode optimizations
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    
                    # Set model to use optimal CPU settings
                    for param in self._gfpgan_model.gfpgan.parameters():
                        param.requires_grad_(False)
                    
                    self.logger.info("Applied safe CPU optimizations to GFPGAN model")
                except Exception as e:
                    self.logger.warning(f"GFPGAN CPU optimization failed: {str(e)}")
                    # Continue without optimizations - model should still work
            
            self.logger.info("Loaded GFPGAN face restoration model")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load GFPGAN model: {str(e)}")
            self._gfpgan_model = None
            return False
    
    def is_esrgan_loaded(self) -> bool:
        """Check if Real-ESRGAN model is loaded."""
        return self._esrgan_model is not None
    
    def is_gfpgan_loaded(self) -> bool:
        """Check if GFPGAN model is loaded."""
        return self._gfpgan_model is not None
    
    def get_esrgan_model(self):
        """Get Real-ESRGAN model instance."""
        return self._esrgan_model
    
    def get_gfpgan_model(self):
        """Get GFPGAN model instance."""
        return self._gfpgan_model
    
    def unload_models(self) -> None:
        """Unload all models to free memory."""
        self._esrgan_model = None
        self._gfpgan_model = None
        self._current_esrgan_scale = None
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Unloaded all models")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'esrgan_loaded': self.is_esrgan_loaded(),
            'esrgan_scale': self._current_esrgan_scale,
            'gfpgan_loaded': self.is_gfpgan_loaded(),
            'cache_dir': str(self.cache_dir),
            'available_models': list(self.MODEL_URLS.keys())
        }
    
    def cleanup_cache(self, keep_recent: int = 2) -> None:
        """Clean up old model files from cache."""
        try:
            model_files = list(self.cache_dir.glob('*.pth'))
            if len(model_files) <= keep_recent:
                return
            
            # Sort by modification time, keep most recent
            model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            files_to_remove = model_files[keep_recent:]
            
            for file_path in files_to_remove:
                file_path.unlink()
                self.logger.info(f"Removed old model file: {file_path.name}")
                
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {str(e)}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.unload_models()