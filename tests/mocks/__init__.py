"""Mock modules for testing."""

from .ai_models import (
    MockRealESRGANer,
    MockGFPGANer,
    MockBasicSRModel,
    MockModelFactory,
    patch_realesrgan_import,
    patch_gfpgan_import,
    patch_basicsr_import
)

__all__ = [
    'MockRealESRGANer',
    'MockGFPGANer', 
    'MockBasicSRModel',
    'MockModelFactory',
    'patch_realesrgan_import',
    'patch_gfpgan_import',
    'patch_basicsr_import'
]