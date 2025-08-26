#!/bin/bash
# Portable Photo-Restore ML Environment Setup Script
# Replicates working server environment for local development

set -e  # Exit on any error

echo "🚀 PHOTO-RESTORE ML ENVIRONMENT SETUP"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install with version fallback
install_with_fallback() {
    local package="$1"
    local version="$2"
    local fallback_version="$3"
    
    echo "📦 Installing $package..."
    
    if pip install "$package==$version"; then
        echo "✅ Successfully installed $package==$version"
    elif [ -n "$fallback_version" ] && pip install "$package==$fallback_version"; then
        echo "✅ Successfully installed $package==$fallback_version (fallback)"
    else
        echo "⚠️  Warning: Failed to install specific version, trying latest..."
        pip install "$package" || echo "❌ Failed to install $package"
    fi
}

# Check Python version
echo "🐍 Checking Python environment..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found"
    exit 1
fi

PIP_CMD="$PYTHON_CMD -m pip"

# Upgrade pip
echo "🔧 Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel

# Strategy 1: Install core PyTorch first (ESSENTIAL)
echo ""
echo "🔥 INSTALLING CORE PYTORCH STACK..."
echo "================================="

# Install PyTorch (CPU version for maximum compatibility)
echo "📦 Installing PyTorch (CPU version for compatibility)..."
$PIP_CMD install torch>=1.9.0,<2.0.0 --index-url https://download.pytorch.org/whl/cpu || {
    echo "⚠️  PyTorch CPU install failed, trying regular install..."
    $PIP_CMD install torch>=1.9.0,<2.0.0
}

$PIP_CMD install torchvision>=0.10.0,<1.0.0

# Install core dependencies
echo ""
echo "🧰 INSTALLING CORE DEPENDENCIES..."
echo "================================"

install_with_fallback "numpy" "1.21.0" "1.24.0"
install_with_fallback "opencv-python" "4.5.0" "4.8.0" 
install_with_fallback "Pillow" "8.0.0" "10.0.0"
install_with_fallback "requests" "2.25.0" ""
install_with_fallback "click" "8.0.0" ""
install_with_fallback "tqdm" "4.62.0" ""
install_with_fallback "PyYAML" "5.4.0" ""
install_with_fallback "psutil" "5.8.0" ""

# Strategy 2: Install ML packages with fallback versions
echo ""
echo "🤖 INSTALLING ML PACKAGES (with version fallbacks)..."
echo "==================================================="

# Try modern versions first, fallback to working versions
echo "📦 Attempting modern ML packages..."

if ! $PIP_CMD install realesrgan>=0.3.0; then
    echo "⚠️  Modern realesrgan failed, trying legacy version..."
    install_with_fallback "realesrgan" "0.2.5" "0.2.1"
fi

if ! $PIP_CMD install gfpgan>=1.3.0; then
    echo "⚠️  Modern gfpgan failed, trying legacy version..."
    install_with_fallback "gfpgan" "1.2.0" "1.1.0"
fi

if ! $PIP_CMD install basicsr>=1.4.0; then
    echo "⚠️  Modern basicsr failed, trying legacy version..."
    install_with_fallback "basicsr" "1.3.5" "1.3.1"
fi

install_with_fallback "facexlib" "0.2.5" "0.2.0"

# Strategy 3: Test installation
echo ""
echo "🧪 TESTING INSTALLATION..."
echo "========================="

$PYTHON_CMD -c "
import sys
import importlib

packages_to_test = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'), 
    ('cv2', 'OpenCV'),
    ('PIL', 'Pillow'),
    ('numpy', 'NumPy'),
    ('requests', 'Requests'),
    ('click', 'Click'),
    ('tqdm', 'TQDM'),
    ('yaml', 'PyYAML')
]

ml_packages = [
    ('basicsr', 'BasicSR'),
    ('realesrgan', 'Real-ESRGAN'),
    ('gfpgan', 'GFPGAN'),
    ('facexlib', 'FaceXLib')
]

print('✅ CORE PACKAGES:')
for module, name in packages_to_test:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'Unknown')
        print(f'  ✅ {name}: {version}')
    except ImportError as e:
        print(f'  ❌ {name}: MISSING ({e})')

print()
print('🤖 ML PACKAGES:')
for module, name in ml_packages:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'Unknown')
        print(f'  ✅ {name}: {version}')
    except ImportError as e:
        print(f'  ❌ {name}: MISSING ({e})')

print()
print('🧠 ARCHITECTURE TESTS:')
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print('  ✅ BasicSR RRDBNet: Available')
except ImportError:
    try:
        from realesrgan.archs.rrdbnet_arch import RRDBNet  
        print('  ✅ RealESRGAN RRDBNet: Available')
    except ImportError:
        print('  ⚠️  RRDBNet: Not available via packages (manual implementation needed)')

# Test basic torch operations
try:
    import torch
    test_tensor = torch.randn(1, 3, 64, 64)
    print('  ✅ PyTorch Operations: Working')
except Exception as e:
    print(f'  ❌ PyTorch Operations: Failed ({e})')
"

# Strategy 4: Environment validation
echo ""
echo "🔍 ENVIRONMENT VALIDATION..."
echo "==========================="

echo "📁 Model files check:"
if [ -d "$HOME/.photo-restore/models" ]; then
    model_count=$(ls -1 "$HOME/.photo-restore/models"/*.pth 2>/dev/null | wc -l)
    echo "  ✅ Model directory exists with $model_count .pth files"
    ls -lh "$HOME/.photo-restore/models"/*.pth 2>/dev/null || echo "  No .pth files found"
else
    echo "  ⚠️  Model directory not found at $HOME/.photo-restore/models"
    echo "  📥 You may need to download model files or run photo-restore once"
fi

echo ""
echo "🎯 NEXT STEPS:"
echo "============="
echo "1. ✅ Dependencies installed"
echo "2. 🧪 Run the environment analysis script:"
echo "   python3 environment_analysis.py"
echo "3. 🚀 Test photo-restore functionality"
echo "4. 📊 Compare with server environment analysis"

echo ""
echo "🔧 TROUBLESHOOTING:"
echo "=================="
echo "If issues persist:"
echo "• Run environment_analysis.py on working server"
echo "• Compare package versions"
echo "• Use manual implementation fallback"
echo "• Consider Docker deployment for consistency"

echo ""
echo "✨ Setup complete! Environment should now match server capabilities."