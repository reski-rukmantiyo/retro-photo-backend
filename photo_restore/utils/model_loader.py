"""Emergency model loading fixes for Real-ESRGAN NoneType error."""

import torch
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_manual_rrdbnet():
    """Create RRDBNet architecture manually to avoid import issues."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ResidualDenseBlock(nn.Module):
        """Residual Dense Block for RRDBNet."""
        def __init__(self, num_feat=64, num_grow_ch=32):
            super(ResidualDenseBlock, self).__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            
        def forward(self, x):
            x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
            x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), 1)), negative_slope=0.2, inplace=True)
            x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)), negative_slope=0.2, inplace=True)
            x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)), negative_slope=0.2, inplace=True)
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x
    
    class RRDB(nn.Module):
        """Residual in Residual Dense Block."""
        def __init__(self, num_feat, num_grow_ch=32):
            super(RRDB, self).__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
            
        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x
    
    class RRDBNet(nn.Module):
        """Networks consisting of Residual in Residual Dense Block."""
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
            super(RRDBNet, self).__init__()
            self.scale = scale
            if scale == 2:
                num_upsample = 1
            elif scale == 1:
                num_upsample = 0
            elif scale == 4:
                num_upsample = 2
            elif scale == 8:
                num_upsample = 3
            else:
                raise ValueError(f'scale {scale} is not supported.')
            
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.ModuleList([RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            
            # upsample
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            
            self.num_upsample = num_upsample
            
        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = feat
            for block in self.body:
                body_feat = block(body_feat)
            body_feat = self.conv_body(body_feat)
            feat = feat + body_feat
            
            # upsample
            if self.num_upsample >= 1:
                feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')), negative_slope=0.2, inplace=True)
            if self.num_upsample >= 2:
                feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')), negative_slope=0.2, inplace=True)
            
            out = self.conv_last(F.leaky_relu(self.conv_hr(feat), negative_slope=0.2, inplace=True))
            return out
    
    return RRDBNet


def emergency_realesrgan_loader(model_path: Path, scale: int = 4, device: str = 'cpu'):
    """
    Emergency Real-ESRGAN loader with explicit architecture creation.
    Fixes NoneType load_state_dict error.
    """
    try:
        # Try direct architecture import (bypass problematic modules)
        try:
            # Import just the architecture file directly
            import sys
            import importlib.util
            from pathlib import Path as ImportPath
            
            # Try to find and import RRDBNet directly
            venv_path = ImportPath(__file__).parent.parent.parent / 'venv'
            basicsr_arch_path = venv_path / 'lib/python3.12/site-packages/basicsr/archs/rrdbnet_arch.py'
            
            if basicsr_arch_path.exists():
                spec = importlib.util.spec_from_file_location("rrdbnet_arch", basicsr_arch_path)
                rrdbnet_module = importlib.util.module_from_spec(spec)
                
                # Mock torch.nn to avoid dependency issues
                import torch.nn as nn
                sys.modules['torch.nn'] = nn
                
                spec.loader.exec_module(rrdbnet_module)
                RRDBNet = rrdbnet_module.RRDBNet
                logger.info("Using direct BasicSR RRDBNet import")
            else:
                raise ImportError("Direct import failed")
                
        except ImportError:
            # Fallback: Create RRDBNet architecture manually
            logger.warning("Creating RRDBNet architecture manually")
            RRDBNet = create_manual_rrdbnet()
            if RRDBNet is None:
                return None
    except Exception as e:
        logger.error(f"Failed to get RRDBNet architecture: {e}")
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
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
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