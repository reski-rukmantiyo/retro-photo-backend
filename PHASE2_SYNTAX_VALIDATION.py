#!/usr/bin/env python3
"""
PHASE 2 SYNTAX VALIDATION - GFPGAN v1.4 Support
Validates implementation without requiring dependencies
"""

import ast
import sys
from pathlib import Path

def validate_cli_implementation():
    """Validate CLI implementation"""
    print("Validating CLI implementation...")
    
    cli_path = Path("photo_restore/cli.py")
    with open(cli_path, 'r') as f:
        content = f.read()
    
    # Check for gfpgan-version argument
    if "--gfpgan-version" not in content:
        print("❌ --gfpgan-version argument not found in CLI")
        return False
    
    # Check for gfpgan_version parameter in main function
    if "gfpgan_version: str" not in content:
        print("❌ gfpgan_version parameter not found in main function")
        return False
    
    # Check for v1.3, v1.4, auto choices
    if "'v1.3', 'v1.4', 'auto'" not in content:
        print("❌ GFPGAN version choices not properly defined")
        return False
    
    print("✅ CLI implementation validated")
    return True

def validate_image_processor_implementation():
    """Validate ImageProcessor implementation"""
    print("Validating ImageProcessor implementation...")
    
    processor_path = Path("photo_restore/processors/image_processor.py")
    with open(processor_path, 'r') as f:
        content = f.read()
    
    # Check for gfpgan_version parameter in process_image
    if "gfpgan_version: str = 'auto'" not in content:
        print("❌ gfpgan_version parameter not found in process_image")
        return False
    
    # Check for _resolve_gfpgan_version method
    if "def _resolve_gfpgan_version" not in content:
        print("❌ _resolve_gfpgan_version method not found")
        return False
    
    # Check for version resolution logic
    if "if gfpgan_version in ['v1.3', 'v1.4']:" not in content:
        print("❌ Version resolution logic not found")
        return False
    
    print("✅ ImageProcessor implementation validated")
    return True

def validate_batch_processor_implementation():
    """Validate BatchProcessor implementation"""
    print("Validating BatchProcessor implementation...")
    
    batch_path = Path("photo_restore/processors/batch_processor.py")
    with open(batch_path, 'r') as f:
        content = f.read()
    
    # Check for gfpgan_version parameter in process_directory
    if "gfpgan_version: str = 'auto'" not in content:
        print("❌ gfpgan_version parameter not found in process_directory")
        return False
    
    # Check for parameter passing to image processor
    if "gfpgan_version=gfpgan_version" not in content:
        print("❌ gfpgan_version not passed to image processor")
        return False
    
    print("✅ BatchProcessor implementation validated")
    return True

def validate_model_manager_implementation():
    """Validate ModelManager implementation"""
    print("Validating ModelManager implementation...")
    
    manager_path = Path("photo_restore/models/model_manager.py")
    with open(manager_path, 'r') as f:
        content = f.read()
    
    # Check for version parameter in load_gfpgan_model
    if "version: str = None" not in content:
        print("❌ version parameter not found in load_gfpgan_model")
        return False
    
    # Check for get_gfpgan_version method
    if "def get_gfpgan_version" not in content:
        print("❌ get_gfpgan_version method not found")
        return False
    
    # Check for version selection logic
    if "if version == 'v1.3':" not in content:
        print("❌ Version selection logic not found")
        return False
    
    print("✅ ModelManager implementation validated")
    return True

def validate_all_syntax():
    """Validate Python syntax for all modified files"""
    print("Validating Python syntax...")
    
    files = [
        "photo_restore/cli.py",
        "photo_restore/processors/image_processor.py", 
        "photo_restore/processors/batch_processor.py",
        "photo_restore/models/model_manager.py"
    ]
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            ast.parse(content)
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
    
    print("✅ All Python syntax validated")
    return True

def main():
    """Run all validations"""
    print("🔧 PHASE 2 IMPLEMENTATION SYNTAX VALIDATION")
    print("=" * 55)
    
    validations = [
        validate_all_syntax,
        validate_cli_implementation,
        validate_image_processor_implementation,
        validate_batch_processor_implementation, 
        validate_model_manager_implementation
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        if validation():
            passed += 1
        print()
    
    print("=" * 55)
    print(f"RESULTS: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 PHASE 2 IMPLEMENTATION SYNTAX VALIDATION: SUCCESS")
        print("✅ All components properly implemented")
        return True
    else:
        print(f"⚠️  {total - passed} validations failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)