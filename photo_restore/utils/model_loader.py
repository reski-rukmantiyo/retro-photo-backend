"""Emergency model loading fixes for Real-ESRGAN NoneType error."""

import torch
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def emergency_realesrgan_loader(model_path: Path, scale: int = 4, device: str = 'cpu'):
    """
    Emergency Real-ESRGAN loader with explicit architecture creation.
    Fixes NoneType load_state_dict error.
    """
    try:
        # Import architecture directly
        from basicsr.archs.rrdbnet_arch import RRDBNet
        logger.info("Using BasicSR RRDBNet architecture")
    except ImportError:
        try:
            from realesrgan.archs.rrdbnet_arch import RRDBNet
            logger.info("Using RealESRGAN RRDBNet architecture")
        except ImportError:
            logger.error("No RRDBNet architecture available")
            return None
    
    try:
        # Create model architecture explicitly
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        elif scale == 4:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        else:
            logger.error(f"Unsupported scale: {scale}")
            return None
        
        logger.info(f"Created RRDBNet model: {type(model)}")
        
        # Load state dict manually
        state_dict = torch.load(str(model_path), map_location=device)
        
        # Handle different state dict formats
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Clean state dict keys if needed
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            key = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_state_dict[key] = v
        
        # Load weights into model
        model.load_state_dict(cleaned_state_dict, strict=True)
        model.eval()
        model = model.to(device)
        
        logger.info(f"Successfully loaded Real-ESRGAN {scale}x model")
        return model
        
    except Exception as e:
        logger.error(f"Emergency loader failed: {e}")
        logger.error(f"Model path: {model_path}")
        logger.error(f"Model exists: {model_path.exists()}")
        return None


def create_manual_realesrgan_upsampler(model_path: Path, scale: int = 4, 
                                      tile_size: int = 512, device: str = 'cpu'):
    """
    Create RealESRGANer manually if automatic creation fails.
    """
    try:
        # Load model manually
        model = emergency_realesrgan_loader(model_path, scale, device)
        if model is None:
            return None
        
        # Create manual upsampler class
        class ManualRealESRGANer:
            def __init__(self, model, scale, tile_size, device):
                self.model = model
                self.scale = scale
                self.tile_size = tile_size
                self.device = device
                
            def enhance(self, image, outscale=None):
                """Manual enhancement using the loaded model."""
                import cv2
                import numpy as np
                
                outscale = outscale or self.scale
                
                # Convert to tensor
                image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Process with model
                with torch.no_grad():
                    output_tensor = self.model(image_tensor)
                
                # Convert back to numpy
                output = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
                
                return output, None
        
        upsampler = ManualRealESRGANer(model, scale, tile_size, device)
        logger.info("Created manual Real-ESRGAN upsampler")
        return upsampler
        
    except Exception as e:
        logger.error(f"Manual upsampler creation failed: {e}")
        return None


def diagnose_realesrgan_issue(model_path: Path):
    """
    Comprehensive diagnosis of Real-ESRGAN loading issues.
    """
    diagnosis = {
        'model_file_exists': model_path.exists(),
        'model_file_size': model_path.stat().st_size if model_path.exists() else 0,
        'torch_version': torch.__version__,
        'device': 'cpu',
        'architecture_available': False,
        'state_dict_loadable': False,
        'error_details': []
    }
    
    # Check architecture availability
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        diagnosis['architecture_available'] = True
        diagnosis['architecture_source'] = 'basicsr'
    except ImportError:
        try:
            from realesrgan.archs.rrdbnet_arch import RRDBNet
            diagnosis['architecture_available'] = True
            diagnosis['architecture_source'] = 'realesrgan'
        except ImportError:
            diagnosis['error_details'].append("No RRDBNet architecture available")
    
    # Check state dict loading
    if model_path.exists():
        try:
            state_dict = torch.load(str(model_path), map_location='cpu')
            diagnosis['state_dict_loadable'] = True
            diagnosis['state_dict_keys'] = len(state_dict)
            if isinstance(state_dict, dict):
                diagnosis['top_level_keys'] = list(state_dict.keys())
        except Exception as e:
            diagnosis['error_details'].append(f"State dict loading failed: {e}")
    
    return diagnosis