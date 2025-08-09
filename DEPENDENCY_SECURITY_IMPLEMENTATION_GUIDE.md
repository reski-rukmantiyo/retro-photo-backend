# Dependency Security Implementation Guide

## Quick Start - Immediate Actions Required

### 1. Update requirements.txt NOW
Replace the current requirements.txt with this secure version:

```txt
# Secure requirements.txt - Updated 2025-08-06
# CRITICAL: These versions address known CVEs

torch==2.2.1+cpu
opencv-python==4.9.0.80
Pillow==10.3.0
numpy==1.26.4
click==8.1.7
tqdm==4.66.2
PyYAML==6.0.1
requests==2.31.0
psutil==5.9.8

# AI/ML packages - pin after security review
realesrgan==0.3.0
gfpgan==1.3.8
basicsr==1.4.2
facexlib==0.3.0

# Security tools
pip-audit==2.7.0
safety==3.0.1
```

### 2. Install Security Tools
```bash
pip install pip-audit safety
```

### 3. Run Vulnerability Scan
```bash
# Check current vulnerabilities
pip-audit
safety check

# Use our custom scanner
python security_patches/dependency_security.py
```

### 4. Update Model Loading Code

Replace all instances of `torch.load()` with:

```python
from security_patches.dependency_security import secure_torch_load

# OLD (VULNERABLE):
model = torch.load('model.pth')

# NEW (SECURE):
model = secure_torch_load(Path('model.pth'), device='cpu')
```

### 5. Add Model Verification

Before loading any model:

```python
from security_patches.dependency_security import SecureDependencyManager

security_mgr = SecureDependencyManager()

# Verify model before loading
try:
    security_mgr.verify_model_integrity(Path('models/RealESRGAN_x4plus.pth'))
    # Safe to load model
except ModelIntegrityError as e:
    print(f"Model verification failed: {e}")
    # DO NOT load the model
```

## Integration Steps

### Step 1: Update model_loader.py

```python
# In photo_restore/utils/model_loader.py

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from security_patches.dependency_security import secure_torch_load, SecureDependencyManager

class ModelLoader:
    def __init__(self):
        self.security_mgr = SecureDependencyManager()
    
    def load_model(self, model_path: str, model_name: str):
        # Verify integrity first
        path = Path(model_path)
        
        try:
            self.security_mgr.verify_model_integrity(path, model_name)
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            raise
        
        # Load securely
        return secure_torch_load(path, device='cpu')
```

### Step 2: Add Download Verification

```python
# When downloading models

from security_patches.dependency_security import SecureDependencyManager

security_mgr = SecureDependencyManager()

# Download and verify
model_path = security_mgr.secure_download_model(
    'RealESRGAN_x4plus.pth',
    Path('models/RealESRGAN_x4plus.pth')
)
```

### Step 3: Set Up CI/CD Security Scanning

Add to your CI/CD pipeline:

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pip-audit safety
    
    - name: Run security scan
      run: |
        pip-audit
        safety check
        python security_patches/dependency_security.py
```

### Step 4: Add Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: security-scan
        name: Dependency Security Scan
        entry: pip-audit
        language: system
        pass_filenames: false
        always_run: true
```

## Critical Security Fixes by Package

### PyTorch (torch)
**Current Risk: CRITICAL**
- Update from `>=1.9.0` to `==2.2.1+cpu`
- Fixes CVE-2022-45907, CVE-2023-43654, CVE-2024-27894, CVE-2024-31580
- Use `secure_torch_load()` for all model loading

### Pillow
**Current Risk: HIGH**
- Update from `>=8.0.0` to `==10.3.0`
- Fixes CVE-2022-22817, CVE-2022-45198, CVE-2023-44271, CVE-2023-50447
- Critical for image processing security

### OpenCV
**Current Risk: HIGH**
- Update from `>=4.5.0` to `==4.9.0.80`
- Fixes CVE-2023-2617, CVE-2023-2618, CVE-2024-28330
- Prevents arbitrary code execution via malformed images

## Model Security Checklist

- [ ] All models have SHA256 hashes in `TRUSTED_MODEL_HASHES`
- [ ] Model downloads use HTTPS with SSL verification
- [ ] Model integrity is verified before every load
- [ ] Using `secure_torch_load()` instead of `torch.load()`
- [ ] Model files have restricted permissions (600)
- [ ] Download directory is not world-writable

## Testing Security Patches

```bash
# Run security tests
python -m pytest security_patches/test_dependency_security.py -v

# Test specific security features
python -m pytest security_patches/test_dependency_security.py::TestSecureDependencyManager -v

# Integration test
python security_patches/test_dependency_security.py
```

## Monitoring and Maintenance

### Weekly Tasks
1. Run `pip-audit` to check for new vulnerabilities
2. Review security advisories for AI/ML packages
3. Check for dependency updates

### Monthly Tasks
1. Full security audit with report generation
2. Review and update trusted model hashes
3. Test disaster recovery procedures

### Automated Monitoring Script

```bash
#!/bin/bash
# security_monitor.sh

echo "Running weekly security check..."

# Update security tools
pip install --upgrade pip-audit safety

# Run scans
pip-audit --desc
safety check

# Custom scan
python security_patches/dependency_security.py

# Generate report
python -c "
from security_patches.dependency_security import SecureDependencyManager, generate_security_report
mgr = SecureDependencyManager()
report = generate_security_report(mgr)
print(report)
" > weekly_security_report_$(date +%Y%m%d).md

echo "Security check complete. Check weekly_security_report_*.md for details."
```

## Emergency Response

If a vulnerability is discovered:

1. **Immediate**: Stop processing untrusted images
2. **Within 1 hour**: Update affected packages
3. **Within 4 hours**: Deploy patches to production
4. **Within 24 hours**: Audit all processed files

## Contact for Security Issues

For security concerns related to this implementation:
- Create a private security advisory in the GitHub repository
- Do NOT create public issues for security vulnerabilities
- Include: affected version, POC (if applicable), suggested fix

## Summary

The most critical actions are:
1. Update PyTorch to 2.2.1+ immediately (CRITICAL)
2. Update Pillow to 10.3.0+ (HIGH)
3. Implement secure model loading (CRITICAL)
4. Add model integrity verification (HIGH)
5. Set up automated vulnerability scanning (MEDIUM)