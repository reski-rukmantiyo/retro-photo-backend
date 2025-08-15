# Technical Phase Analysis - Model Implementation

## Phase Requirements Summary 📋

Based on `backend_spec.md` analysis:

**Phase 1**: Real-ESRGAN x2, Real-ESRGAN x4, GFPGAN v1.3  
**Phase 2**: GFPGAN v1.4  

CLI arguments must specify models for direct testing capability.

---

## 1. Implementation Complexity Assessment ⚖️

### Phase 1 Complexity: **LOW-MEDIUM** ✅
**Current Status**: ~80% already implemented

#### Existing Infrastructure ✅
- **Real-ESRGAN x2/x4**: Already configured in `ModelManager.MODEL_URLS`
- **GFPGAN v1.3**: Default model in configurations
- **Model loading logic**: Complete with fallbacks
- **CLI framework**: Click-based with extensible options
- **Security patches**: Hash verification ready

#### Required Work
```python
# Estimated Development Time: 2-3 days
Tasks:
1. Add CLI model selection arguments (4 hours)
2. Extend ModelManager model selection logic (6 hours) 
3. Update configuration system (2 hours)
4. Add model validation (4 hours)
5. Testing and documentation (8 hours)
```

### Phase 2 Complexity: **LOW** ✅  
**Current Status**: ~60% already implemented

#### Existing Infrastructure ✅
- **GFPGAN v1.4**: Already configured as `gfpgan_light` in ModelManager
- **Download URLs**: Configured for v1.3.4 release  
- **Security hashes**: Defined in security patches
- **Fallback logic**: Built into existing model loading

#### Required Work
```python
# Estimated Development Time: 1-2 days  
Tasks:
1. Update CLI to include v1.4 selection (2 hours)
2. Test model compatibility (4 hours)
3. Update documentation (2 hours)
4. Integration testing (4 hours)
```

---

## 2. Code Changes Required 🔧

### A. CLI Argument Extension

#### Current CLI Structure Analysis
```python
# Existing arguments in cli.py:
@click.option('--quality', type=click.Choice(['fast', 'balanced', 'best']))
@click.option('--upscale', type=int, default=4, help='Upscale factor (2 or 4)')
@click.option('--face-enhance', is_flag=True, default=True)
```

#### Required New Arguments
```python
# Add to cli.py:
@click.option('--esrgan-model', 
              type=click.Choice(['x2', 'x4', 'auto']), 
              default='auto',
              help='Real-ESRGAN model version (auto selects based on quality)')

@click.option('--gfpgan-model', 
              type=click.Choice(['v1.3', 'v1.4', 'auto']), 
              default='auto',
              help='GFPGAN model version (auto selects based on performance)')

@click.option('--list-models', is_flag=True, 
              help='List available models and exit')
```

### B. Model Manager Extensions

#### Current Model Structure (MODEL_URLS)
```python
# Already exists in model_manager.py:
MODEL_URLS = {
    'esrgan_x2': {...},      # Phase 1 ✅
    'esrgan_x4': {...},      # Phase 1 ✅  
    'gfpgan': {...},         # Phase 1 (v1.3) ✅
    'gfpgan_light': {...}    # Phase 2 (v1.4) ✅
}
```

#### Required Model Selection Logic
```python
def select_esrgan_model(self, model_spec: str, quality: str, upscale: int) -> str:
    """
    Select appropriate ESRGAN model based on specifications.
    
    Args:
        model_spec: 'x2', 'x4', or 'auto'
        quality: 'fast', 'balanced', 'best'  
        upscale: 2 or 4
        
    Returns:
        Model key for MODEL_URLS
    """
    if model_spec == 'auto':
        if quality == 'fast' or upscale == 2:
            return 'esrgan_x2'
        else:
            return 'esrgan_x4'
    else:
        return f'esrgan_{model_spec}'

def select_gfpgan_model(self, model_spec: str, quality: str) -> str:
    """
    Select appropriate GFPGAN model based on specifications.
    
    Args:
        model_spec: 'v1.3', 'v1.4', or 'auto'
        quality: 'fast', 'balanced', 'best'
        
    Returns:
        Model key for MODEL_URLS  
    """
    if model_spec == 'auto':
        if quality == 'fast':
            return 'gfpgan_light'  # v1.4 - more efficient
        else:
            return 'gfpgan'        # v1.3 - better quality
    else:
        version_map = {
            'v1.3': 'gfpgan',
            'v1.4': 'gfpgan_light'
        }
        return version_map[model_spec]
```

### C. Configuration System Updates

#### Extend Config Classes
```python
# Add to utils/config.py:
@dataclass
class ModelSelectionConfig:
    """Model selection configuration."""
    esrgan_model: str = "auto"        # x2, x4, auto
    gfpgan_model: str = "auto"        # v1.3, v1.4, auto
    auto_selection_enabled: bool = True
    prefer_performance: bool = False   # vs quality
    
    def get_esrgan_preference(self, quality: str, upscale: int) -> str:
        """Get ESRGAN model based on preferences."""
        if self.esrgan_model != "auto":
            return self.esrgan_model
            
        if self.prefer_performance or quality == "fast":
            return "x2" if upscale == 2 else "x4"
        else:
            return "x4"  # Always use x4 for quality
    
    def get_gfpgan_preference(self, quality: str) -> str:
        """Get GFPGAN model based on preferences.""" 
        if self.gfpgan_model != "auto":
            return self.gfpgan_model
            
        if self.prefer_performance or quality == "fast":
            return "v1.4"  # More efficient
        else:
            return "v1.3"  # Better quality
```

### D. Image Processor Integration

#### Update Processing Pipeline
```python
# Modify processors/image_processor.py:
def process_image(
    self,
    input_path: str, 
    output_path: str,
    quality: str = 'balanced',
    upscale: int = 4,
    face_enhance: bool = True,
    output_format: str = 'jpg',
    esrgan_model: str = 'auto',      # NEW
    gfpgan_model: str = 'auto',      # NEW
    progress_callback: Optional[Callable[[int], None]] = None
) -> bool:
    """Enhanced process_image with model selection."""
    
    # Model selection logic
    selected_esrgan = self.config.model_selection.get_esrgan_preference(
        quality, upscale) if esrgan_model == 'auto' else esrgan_model
        
    selected_gfpgan = self.config.model_selection.get_gfpgan_preference(
        quality) if gfpgan_model == 'auto' else gfpgan_model
    
    # Load models with specific versions
    self._ensure_models_loaded(
        upscale=upscale, 
        face_enhance=face_enhance, 
        quality=quality,
        esrgan_model=selected_esrgan,    # NEW
        gfpgan_model=selected_gfpgan     # NEW
    )
    
    # Continue with existing processing logic...
```

---

## 3. Compatibility Issues Analysis 🔍

### A. Model Compatibility Matrix

| Model Combination | Compatibility | Issues | Mitigation |
|------------------|---------------|--------|------------|
| ESRGAN x2 + GFPGAN v1.3 | ✅ HIGH | None | Already implemented |
| ESRGAN x4 + GFPGAN v1.3 | ✅ HIGH | None | Already implemented |
| ESRGAN x2 + GFPGAN v1.4 | ✅ HIGH | None | Direct swap |
| ESRGAN x4 + GFPGAN v1.4 | ✅ HIGH | None | Direct swap |

### B. Version-Specific Considerations

#### GFPGAN v1.3 vs v1.4 Differences
```python
# Model-specific optimizations needed:
GFPGAN_VERSION_CONFIG = {
    'v1.3': {
        'file': 'GFPGANv1.3.pth',
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'memory_mb': 800,
        'quality_score': 95,      # Higher quality
        'speed_score': 75,        # Slower
        'features': ['identity_preservation', 'high_fidelity']
    },
    'v1.4': {
        'file': 'GFPGANv1.4.pth', 
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
        'memory_mb': 600,         # More efficient
        'quality_score': 90,      # Slightly lower quality  
        'speed_score': 90,        # Faster
        'features': ['efficiency', 'speed_optimized']
    }
}
```

### C. Backward Compatibility

#### Configuration Compatibility
```python
# Ensure old configurations still work:
def migrate_config(old_config: dict) -> dict:
    """Migrate old configuration to new format."""
    new_config = old_config.copy()
    
    # Add model selection defaults if missing
    if 'model_selection' not in new_config:
        new_config['model_selection'] = {
            'esrgan_model': 'auto',
            'gfpgan_model': 'auto',  
            'auto_selection_enabled': True
        }
    
    return new_config
```

#### CLI Backward Compatibility
```python
# All existing CLI commands continue to work:
# OLD: photo-restore image.jpg --quality best --face-enhance
# NEW: photo-restore image.jpg --quality best --face-enhance --gfpgan-model v1.4
# BOTH: Work identically (new args optional with smart defaults)
```

---

## 4. Testing Approach 🧪

### A. Test Strategy Overview

#### Phase 1 Testing Scope
```python
# Test matrix for Phase 1:
TEST_COMBINATIONS_PHASE1 = [
    # (esrgan_model, gfpgan_model, quality, expected_result)
    ('x2', 'v1.3', 'fast', 'success'),
    ('x4', 'v1.3', 'balanced', 'success'), 
    ('x4', 'v1.3', 'best', 'success'),
    ('auto', 'auto', 'fast', 'selects_x2_v1.3'),
    ('auto', 'auto', 'balanced', 'selects_x4_v1.3'),
    ('auto', 'auto', 'best', 'selects_x4_v1.3'),
]
```

#### Phase 2 Testing Scope
```python  
# Test matrix for Phase 2:
TEST_COMBINATIONS_PHASE2 = [
    # Add v1.4 combinations
    ('x2', 'v1.4', 'fast', 'success'),
    ('x4', 'v1.4', 'balanced', 'success'),
    ('x4', 'v1.4', 'best', 'success'),
    ('auto', 'v1.4', 'fast', 'success'),
    # Mixed version testing
    ('x2', 'auto', 'fast', 'selects_v1.4'),    # Auto prefers v1.4 for speed
    ('x4', 'auto', 'best', 'selects_v1.3'),    # Auto prefers v1.3 for quality
]
```

### B. Test Implementation

#### Unit Tests
```python
# tests/test_model_selection.py
class TestModelSelection:
    
    def test_esrgan_model_selection(self):
        """Test ESRGAN model selection logic."""
        config = ModelSelectionConfig()
        
        # Test explicit selection
        assert config.get_esrgan_preference('best', 4, 'x2') == 'x2'
        assert config.get_esrgan_preference('best', 4, 'x4') == 'x4'
        
        # Test auto selection  
        assert config.get_esrgan_preference('fast', 2, 'auto') == 'x2'
        assert config.get_esrgan_preference('best', 4, 'auto') == 'x4'
    
    def test_gfpgan_model_selection(self):
        """Test GFPGAN model selection logic."""
        config = ModelSelectionConfig()
        
        # Test explicit selection
        assert config.get_gfpgan_preference('best', 'v1.3') == 'v1.3'
        assert config.get_gfpgan_preference('fast', 'v1.4') == 'v1.4'
        
        # Test auto selection (quality vs performance)
        assert config.get_gfpgan_preference('fast', 'auto') == 'v1.4'  # Performance
        assert config.get_gfpgan_preference('best', 'auto') == 'v1.3'  # Quality

    @pytest.mark.parametrize("esrgan,gfpgan,quality,expected_success", 
                           TEST_COMBINATIONS_PHASE1)
    def test_phase1_combinations(self, esrgan, gfpgan, quality, expected_success):
        """Test all Phase 1 model combinations."""
        processor = create_test_processor()
        
        result = processor.process_image(
            TEST_IMAGE_PATH,
            OUTPUT_PATH, 
            quality=quality,
            esrgan_model=esrgan,
            gfpgan_model=gfpgan
        )
        
        if expected_success == 'success':
            assert result is True
        else:
            # Test auto-selection logic
            assert processor.selected_models == expected_success
```

#### Integration Tests
```python
# tests/test_cli_model_selection.py  
class TestCLIModelSelection:
    
    def test_cli_model_arguments(self):
        """Test new CLI model selection arguments."""
        from click.testing import CliRunner
        from photo_restore.cli import main
        
        runner = CliRunner()
        
        # Test new arguments
        result = runner.invoke(main, [
            'test.jpg',
            '--esrgan-model', 'x2',
            '--gfpgan-model', 'v1.4'
        ])
        
        assert result.exit_code == 0
        assert 'Using ESRGAN x2' in result.output
        assert 'Using GFPGAN v1.4' in result.output
    
    def test_model_listing(self):
        """Test --list-models functionality."""
        runner = CliRunner()
        result = runner.invoke(main, ['--list-models'])
        
        assert result.exit_code == 0
        assert 'Available ESRGAN models:' in result.output
        assert 'x2, x4' in result.output
        assert 'Available GFPGAN models:' in result.output  
        assert 'v1.3, v1.4' in result.output
```

#### Performance Tests
```python
# tests/test_model_performance.py
class TestModelPerformance:
    
    def test_model_speed_comparison(self):
        """Compare processing speed between model versions.""" 
        test_image = load_test_image()
        
        # Benchmark GFPGAN versions
        time_v13 = benchmark_gfpgan_processing(test_image, 'v1.3')
        time_v14 = benchmark_gfpgan_processing(test_image, 'v1.4')
        
        # v1.4 should be faster
        assert time_v14 < time_v13
        
        # ESRGAN benchmarks
        time_x2 = benchmark_esrgan_processing(test_image, 'x2')  
        time_x4 = benchmark_esrgan_processing(test_image, 'x4')
        
        # x2 should be faster than x4
        assert time_x2 < time_x4

    def test_memory_usage_comparison(self):
        """Compare memory usage between model versions."""
        memory_usage = {}
        
        for model in ['v1.3', 'v1.4']:
            usage = measure_memory_usage(lambda: process_with_gfpgan(model))
            memory_usage[model] = usage
        
        # v1.4 should use less memory
        assert memory_usage['v1.4'] < memory_usage['v1.3']
```

### C. Test Data Requirements

#### Test Image Sets
```python
TEST_IMAGE_CATEGORIES = {
    'portraits': [
        'face_high_res.jpg',      # High quality face photo
        'face_low_res.jpg',       # Low resolution face 
        'multiple_faces.jpg',     # Multiple people
        'side_profile.jpg'        # Profile view
    ],
    'landscapes': [
        'nature_4k.jpg',          # High resolution landscape
        'old_photo.jpg',          # Vintage/degraded photo
        'architecture.jpg'        # Buildings and structures
    ],
    'edge_cases': [
        'very_small.jpg',         # 100x100 pixels
        'very_large.jpg',         # 8000x8000 pixels  
        'corrupted.jpg',          # Partially corrupted
        'monochrome.jpg'          # Black and white
    ]
}
```

#### Expected Output Validation
```python
def validate_enhancement_results(input_path: str, output_path: str, 
                               esrgan_model: str, gfpgan_model: str):
    """Validate that enhancement produces expected improvements."""
    
    input_img = load_image(input_path)
    output_img = load_image(output_path)
    
    # Basic validation
    assert output_img.shape[0] >= input_img.shape[0]  # Height increased
    assert output_img.shape[1] >= input_img.shape[1]  # Width increased
    
    # Model-specific validation
    if esrgan_model == 'x4':
        # Should be approximately 4x larger
        assert output_img.shape[0] / input_img.shape[0] >= 3.8
        assert output_img.shape[1] / input_img.shape[1] >= 3.8
    
    if gfpgan_model in ['v1.3', 'v1.4']:
        # Should preserve or improve image quality metrics
        input_quality = calculate_image_quality(input_img)
        output_quality = calculate_image_quality(output_img)
        assert output_quality >= input_quality * 0.95  # Allow 5% tolerance
```

---

## 5. CLI Argument Structure Design 🖥️

### A. Proposed CLI Architecture

#### Enhanced Command Structure
```bash
# Basic usage (unchanged for backward compatibility)
photo-restore input.jpg output.jpg

# Quality-based processing (existing)  
photo-restore input.jpg --quality best --face-enhance

# Phase 1: Explicit model selection
photo-restore input.jpg --esrgan-model x2 --gfpgan-model v1.3

# Phase 2: Include v1.4 option
photo-restore input.jpg --esrgan-model x4 --gfpgan-model v1.4

# Auto-selection with preferences (smart defaults)
photo-restore input.jpg --quality fast  # Auto-selects x2 + v1.4
photo-restore input.jpg --quality best  # Auto-selects x4 + v1.3

# Advanced usage
photo-restore batch/ --batch \
  --esrgan-model x4 \
  --gfpgan-model v1.3 \
  --quality balanced \
  --verbose

# Model discovery
photo-restore --list-models
photo-restore --model-info gfpgan-v1.4
```

### B. Detailed Argument Specifications

#### Model Selection Arguments
```python
@click.option('--esrgan-model', 
              type=click.Choice(['x2', 'x4', 'auto'], case_sensitive=False),
              default='auto',
              show_default=True,
              help='Real-ESRGAN model: x2 (fast), x4 (quality), auto (smart selection)')

@click.option('--gfpgan-model',
              type=click.Choice(['v1.3', 'v1.4', 'auto'], case_sensitive=False), 
              default='auto',
              show_default=True,
              help='GFPGAN model: v1.3 (quality), v1.4 (speed), auto (smart selection)')

@click.option('--list-models', 
              is_flag=True,
              help='List all available models and their characteristics')

@click.option('--model-info',
              type=str,
              help='Show detailed information about a specific model')

@click.option('--prefer-speed', 
              is_flag=True,
              help='Prefer speed over quality in auto-selection mode')

@click.option('--prefer-quality',
              is_flag=True, 
              help='Prefer quality over speed in auto-selection mode (default)')
```

#### Enhanced Help System
```python
def print_model_list():
    """Print comprehensive model information.""" 
    click.echo("\n🤖 Available AI Models:\n")
    
    click.echo("📈 Real-ESRGAN Models:")
    click.echo("  x2    - 2x upscaling, faster processing, lower memory")
    click.echo("  x4    - 4x upscaling, higher quality, more memory")
    click.echo("  auto  - Smart selection based on quality setting")
    
    click.echo("\n👤 GFPGAN Face Enhancement Models:")
    click.echo("  v1.3  - Higher quality faces, more processing time") 
    click.echo("  v1.4  - Faster processing, efficient memory usage")
    click.echo("  auto  - Smart selection based on quality setting")
    
    click.echo("\n🔧 Auto-Selection Rules:")
    click.echo("  Fast quality    → x2 + v1.4 (optimized for speed)")
    click.echo("  Balanced quality → x4 + v1.3 (balanced performance)")  
    click.echo("  Best quality    → x4 + v1.3 (maximum quality)")
    
    click.echo("\n💡 Examples:")
    click.echo("  photo-restore image.jpg --esrgan-model x2")
    click.echo("  photo-restore image.jpg --gfpgan-model v1.4 --prefer-speed")
    click.echo("  photo-restore image.jpg --quality fast  # Auto: x2 + v1.4")

def print_model_info(model_name: str):
    """Print detailed information about specific model."""
    model_db = {
        'esrgan-x2': {
            'name': 'Real-ESRGAN x2',
            'upscale_factor': '2x',
            'memory_usage': '~800MB',
            'processing_time': 'Fast',
            'best_for': 'Quick enhancement, limited memory systems'
        },
        'esrgan-x4': {
            'name': 'Real-ESRGAN x4', 
            'upscale_factor': '4x',
            'memory_usage': '~1.8GB',
            'processing_time': 'Slower',
            'best_for': 'High quality upscaling, detailed enhancement'
        },
        'gfpgan-v1.3': {
            'name': 'GFPGAN v1.3',
            'specialty': 'Face restoration',
            'memory_usage': '~800MB', 
            'processing_time': 'Standard',
            'best_for': 'High quality face enhancement'
        },
        'gfpgan-v1.4': {
            'name': 'GFPGAN v1.4',
            'specialty': 'Face restoration', 
            'memory_usage': '~600MB',
            'processing_time': 'Fast',
            'best_for': 'Quick face enhancement, batch processing'
        }
    }
    
    if model_name.lower() not in model_db:
        click.echo(f"❌ Model '{model_name}' not found. Use --list-models to see all available models.")
        return
        
    info = model_db[model_name.lower()]
    click.echo(f"\n📋 {info['name']} Information:")
    for key, value in info.items():
        if key != 'name':
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")
```

### C. Advanced CLI Features

#### Model Validation and Feedback
```python
def validate_model_arguments(esrgan_model: str, gfpgan_model: str, 
                           quality: str, upscale: int) -> tuple[str, str]:
    """Validate and resolve model selections with user feedback."""
    
    # Resolve auto selections
    if esrgan_model == 'auto':
        if quality == 'fast' or upscale == 2:
            resolved_esrgan = 'x2'
        else:
            resolved_esrgan = 'x4'
        click.echo(f"🔄 Auto-selected ESRGAN: {resolved_esrgan}")
    else:
        resolved_esrgan = esrgan_model
    
    if gfpgan_model == 'auto':
        if quality == 'fast':
            resolved_gfpgan = 'v1.4'
        else:
            resolved_gfpgan = 'v1.3' 
        click.echo(f"🔄 Auto-selected GFPGAN: {resolved_gfpgan}")
    else:
        resolved_gfpgan = gfpgan_model
    
    # Validation warnings
    if resolved_esrgan == 'x2' and upscale == 4:
        click.echo("⚠️  Warning: x2 model selected with 4x upscale factor")
        click.echo("   Consider using --esrgan-model x4 for better 4x results")
    
    if resolved_gfpgan == 'v1.4' and quality == 'best':
        click.echo("💡 Tip: For best quality, consider --gfpgan-model v1.3")
    
    return resolved_esrgan, resolved_gfpgan
```

#### Configuration Integration  
```python
# Enhanced main function signature:
def main(input_path: str, output_path: Optional[str], batch: bool, quality: str,
         upscale: int, face_enhance: bool, output_format: str, 
         config: Optional[str], verbose: bool,
         # NEW MODEL ARGUMENTS
         esrgan_model: str, gfpgan_model: str, list_models: bool,
         model_info: Optional[str], prefer_speed: bool, prefer_quality: bool) -> None:
    
    # Handle utility functions
    if list_models:
        print_model_list()
        return
        
    if model_info:
        print_model_info(model_info)
        return
    
    # Validate and resolve models
    resolved_esrgan, resolved_gfpgan = validate_model_arguments(
        esrgan_model, gfpgan_model, quality, upscale
    )
    
    # Continue with existing processing logic...
    # Pass resolved model selections to processors
```

---

## 6. Implementation Timeline & Recommendations 🚀

### Phase 1 Implementation (2-3 days)
```python
Day 1: Core Infrastructure
- ✅ Add CLI arguments for model selection  
- ✅ Extend ModelManager selection logic
- ✅ Update configuration system
- ✅ Basic validation and error handling

Day 2: Integration & Testing  
- ✅ Update ImageProcessor with model selection
- ✅ Add model listing and info functions
- ✅ Unit tests for model selection logic
- ✅ CLI integration testing

Day 3: Polish & Documentation
- ✅ Performance validation
- ✅ Error handling edge cases
- ✅ Documentation updates
- ✅ End-to-end testing
```

### Phase 2 Implementation (1-2 days)
```python
Day 1: GFPGAN v1.4 Integration
- ✅ Add v1.4 to CLI choices
- ✅ Test model compatibility
- ✅ Update auto-selection logic
- ✅ Performance benchmarking

Day 2: Validation & Deployment
- ✅ Cross-version testing
- ✅ Documentation updates  
- ✅ Final integration tests
- ✅ Release preparation
```

### Risk Assessment: **LOW** ✅

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| **Breaking Changes** | LOW | Backward compatible design |
| **Model Compatibility** | LOW | Models already configured |
| **Performance Impact** | LOW | Selection logic is lightweight |
| **User Confusion** | MEDIUM | Comprehensive help system |
| **Testing Complexity** | MEDIUM | Systematic test matrix |

---

## Summary Recommendations 📝

### ✅ **PROCEED with Implementation**

**Reasons for Confidence:**
1. **Low technical complexity** - Most infrastructure already exists
2. **High compatibility** - Models already configured and tested  
3. **Backward compatibility** - Existing CLI continues to work
4. **Clear user benefit** - Direct model control for testing
5. **Incremental approach** - Phase 1 → Phase 2 reduces risk

### **Implementation Priority:**
1. **Start with Phase 1** - Immediate value with existing models
2. **Parallel development** - CLI args + ModelManager + Tests
3. **Thorough testing** - Model combinations and edge cases
4. **Documentation first** - Clear examples and help system
5. **Phase 2 follow-up** - Quick addition of v1.4 support

### **Success Metrics:**
- ✅ All existing functionality preserved
- ✅ New model selection works as expected  
- ✅ Performance benchmarks validate model differences
- ✅ User can directly specify models for testing
- ✅ Auto-selection provides sensible defaults

The implementation is **technically sound** and **low-risk** with **high user value**.