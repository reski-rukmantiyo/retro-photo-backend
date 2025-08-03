"""
PyTorch compatibility utilities for newer versions.
Handles compatibility issues between Real-ESRGAN and newer PyTorch versions.
"""

import sys
import types
import warnings
from typing import Any

def setup_torchvision_compatibility():
    """
    Set up compatibility for torchvision with newer PyTorch versions.
    This fixes the missing 'functional_tensor' module issue.
    """
    try:
        # Check if functional_tensor already exists
        import torchvision.transforms.functional_tensor
        return True
    except ImportError:
        pass
    
    try:
        # Import the functional module
        from torchvision.transforms import functional as F
        import torchvision.transforms.functional
        
        # Create the missing functional_tensor module
        functional_tensor = types.ModuleType('functional_tensor')
        
        # Add the required functions from functional
        if hasattr(F, 'rgb_to_grayscale'):
            functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
        else:
            # Fallback implementation
            import torch
            def rgb_to_grayscale(image: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
                """Convert RGB image to grayscale."""
                if image.dim() < 3:
                    raise ValueError("Input image should have at least 3 dimensions")
                
                # Use standard RGB to grayscale conversion weights
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
                
                if num_output_channels == 1:
                    return gray.unsqueeze(0)
                elif num_output_channels == 3:
                    return torch.stack([gray, gray, gray], dim=0)
                else:
                    raise ValueError("num_output_channels should be either 1 or 3")
            
            functional_tensor.rgb_to_grayscale = rgb_to_grayscale
        
        # Add other potentially needed functions
        for attr_name in dir(F):
            if not attr_name.startswith('_') and not hasattr(functional_tensor, attr_name):
                try:
                    setattr(functional_tensor, attr_name, getattr(F, attr_name))
                except:
                    pass
        
        # Register the module
        sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
        
        # Also try to fix any other potential import issues
        try:
            import torchvision.transforms
            torchvision.transforms.functional_tensor = functional_tensor
        except:
            pass
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to set up torchvision compatibility: {e}")
        return False

def patch_realesrgan_imports():
    """
    Patch Real-ESRGAN imports to work with newer PyTorch versions.
    """
    # Set up torchvision compatibility first
    setup_torchvision_compatibility()
    
    try:
        # Try to import Real-ESRGAN after patching
        from realesrgan import RealESRGANer
        return RealESRGANer
    except ImportError as e:
        warnings.warn(f"Could not import Real-ESRGAN even after patching: {e}")
        return None

# Automatically apply compatibility fixes when this module is imported
setup_torchvision_compatibility()