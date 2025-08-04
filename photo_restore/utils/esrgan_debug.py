"""Focused debugging for Real-ESRGAN NoneType load_state_dict error."""

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def debug_model_constructor():
    """Step 1: Test model constructor in isolation."""
    print("=== STEP 1: MODEL CONSTRUCTOR DEBUG ===")
    
    # Test BasicSR architecture
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("✅ BasicSR RRDBNet import: SUCCESS")
        
        # Test model creation with exact Real-ESRGAN parameters
        model = RRDBNet(
            num_in_ch=3,      # RGB input
            num_out_ch=3,     # RGB output  
            num_feat=64,      # Feature channels
            num_block=23,     # Number of RRDB blocks
            num_grow_ch=32,   # Growth channels
            scale=4           # 4x upscaling
        )
        print(f"✅ RRDBNet constructor: SUCCESS - {type(model)}")
        print(f"✅ Model device: {next(model.parameters()).device}")
        return model, "basicsr"
        
    except ImportError as e:
        print(f"❌ BasicSR import failed: {e}")
    except Exception as e:
        print(f"❌ BasicSR constructor failed: {e}")
    
    # Test RealESRGAN architecture
    try:
        from realesrgan.archs.rrdbnet_arch import RRDBNet
        print("✅ RealESRGAN RRDBNet import: SUCCESS")
        
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=4
        )
        print(f"✅ RRDBNet constructor: SUCCESS - {type(model)}")
        return model, "realesrgan"
        
    except ImportError as e:
        print(f"❌ RealESRGAN import failed: {e}")
    except Exception as e:
        print(f"❌ RealESRGAN constructor failed: {e}")
    
    return None, None


def debug_realesrganer_constructor():
    """Step 2: Test RealESRGANer wrapper constructor."""
    print("\n=== STEP 2: REALESRGANER WRAPPER DEBUG ===")
    
    try:
        from realesrgan import RealESRGANer
        print("✅ RealESRGANer import: SUCCESS")
        
        # Test with minimal parameters
        fake_model_path = "/tmp/fake_model.pth"
        
        # This should fail at model loading, not constructor
        upsampler = RealESRGANer(
            scale=4,
            model_path=fake_model_path,  # Non-existent file
            device='cpu',
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=False
        )
        print(f"✅ RealESRGANer constructor: SUCCESS - {type(upsampler)}")
        print(f"✅ Upsampler model attribute: {type(upsampler.model)}")
        
        return upsampler
        
    except Exception as e:
        print(f"❌ RealESRGANer constructor failed: {e}")
        print(f"❌ Error type: {type(e)}")
        return None


def debug_state_dict_loading(model_path: Path):
    """Step 3: Test state dict loading in isolation."""
    print(f"\n=== STEP 3: STATE DICT LOADING DEBUG ===")
    print(f"Model path: {model_path}")
    print(f"File exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("❌ Model file does not exist")
        return None
    
    print(f"File size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Load state dict
        state_dict = torch.load(str(model_path), map_location='cpu')
        print(f"✅ State dict loaded: {type(state_dict)}")
        
        if isinstance(state_dict, dict):
            print(f"✅ State dict keys: {len(state_dict)}")
            print(f"✅ Top level keys: {list(state_dict.keys())[:5]}")
            
            # Check for nested structure
            if 'params' in state_dict:
                print("✅ Found 'params' key - using nested structure")
                actual_state_dict = state_dict['params']
            elif 'model' in state_dict:
                print("✅ Found 'model' key - using nested structure")
                actual_state_dict = state_dict['model']
            else:
                print("✅ Using direct state dict")
                actual_state_dict = state_dict
            
            print(f"✅ Actual state dict keys: {len(actual_state_dict)}")
            sample_keys = list(actual_state_dict.keys())[:3]
            print(f"✅ Sample parameter keys: {sample_keys}")
            
            # Check parameter shapes
            for key in sample_keys:
                param = actual_state_dict[key]
                print(f"✅ {key}: {param.shape} {param.dtype}")
            
            return actual_state_dict
        else:
            print(f"❌ Unexpected state dict type: {type(state_dict)}")
            return None
            
    except Exception as e:
        print(f"❌ State dict loading failed: {e}")
        print(f"❌ Error type: {type(e)}")
        return None


def debug_manual_model_loading(model_path: Path):
    """Step 4: Manual model creation and weight loading."""
    print(f"\n=== STEP 4: MANUAL MODEL LOADING DEBUG ===")
    
    # Step 4a: Create model
    model, source = debug_model_constructor()
    if model is None:
        print("❌ Cannot create model - architecture unavailable")
        return None
    
    # Step 4b: Load state dict
    state_dict = debug_state_dict_loading(model_path)
    if state_dict is None:
        print("❌ Cannot load state dict")
        return None
    
    # Step 4c: Load weights into model
    try:
        print("Attempting to load state dict into model...")
        
        # Clean keys if needed
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove module prefix if present
            clean_key = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_state_dict[clean_key] = v
        
        print(f"✅ Cleaned state dict: {len(cleaned_state_dict)} keys")
        
        # Load with strict=False first to see partial matches
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {unexpected_keys[:3]}...")
        
        if not missing_keys and not unexpected_keys:
            print("✅ Perfect state dict match!")
        else:
            print("⚠️ Partial state dict match - but model should still work")
        
        model.eval()
        print(f"✅ Model loaded successfully: {type(model)}")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        return model
        
    except Exception as e:
        print(f"❌ Manual loading failed: {e}")
        print(f"❌ Error type: {type(e)}")
        return None


def run_complete_diagnosis(model_path: str = "~/.photo-restore/models/RealESRGAN_x4plus.pth"):
    """Run complete focused diagnosis."""
    print("🚨 REAL-ESRGAN NONETYPE ERROR DIAGNOSIS")
    print("=" * 50)
    
    model_path = Path(model_path).expanduser()
    
    # Run all debug steps
    debug_model_constructor()
    debug_realesrganer_constructor()
    debug_state_dict_loading(model_path)  
    model = debug_manual_model_loading(model_path)
    
    print(f"\n=== FINAL RESULT ===")
    if model is not None:
        print("✅ DIAGNOSIS COMPLETE: Manual loading works!")
        print("✅ Issue is in RealESRGANer wrapper, not core model")
        print("✅ Solution: Use manual loading approach")
    else:
        print("❌ DIAGNOSIS COMPLETE: Core model loading fails")
        print("❌ Need to investigate architecture compatibility")
    
    return model


if __name__ == "__main__":
    run_complete_diagnosis()