# Security Patches for Photo Restoration CLI

This directory contains security patches and improvements for the Retro Photo Restoration CLI application based on the comprehensive security audit.

## Files Included

### 1. `path_validator.py`
**Purpose:** Prevents path traversal attacks and validates file paths  
**Key Features:**
- Validates paths against allowed directories
- Prevents directory traversal using `..`, `~`, absolute paths
- Detects and resolves symbolic links
- Sanitizes filenames to remove dangerous characters
- Implements defense-in-depth approach

**Usage:**
```python
from security_patches.path_validator import SecurePathValidator

# Initialize with allowed directories
validator = SecurePathValidator(['/safe/dir', '/another/safe/dir'])

# Validate paths
safe_path = validator.validate_path(user_input, base_path)
```

### 2. `secure_image_validator.py`
**Purpose:** Validates image files to prevent malicious uploads  
**Key Features:**
- Magic byte (file signature) validation
- File size and dimension limits
- Extension vs content validation
- Malicious content detection
- Memory usage estimation

**Usage:**
```python
from security_patches.secure_image_validator import SecureImageValidator

validator = SecureImageValidator(
    max_file_size=20 * 1024 * 1024,  # 20MB
    max_dimensions=(8000, 8000)
)

# Validate image
results = validator.validate_image(file_path)
```

### 3. `secure_model_loader.py`
**Purpose:** Safely loads AI models without dynamic code execution  
**Key Features:**
- SHA256 hash verification for model files
- No dynamic imports or exec()
- Restricted unpickling for older PyTorch versions
- Pre-defined model architectures only
- Trusted model whitelist

**Usage:**
```python
from security_patches.secure_model_loader import SecureModelLoader

loader = SecureModelLoader(model_dir)
model = loader.load_realesrgan_model('RealESRGAN_x4plus.pth', scale=4)
```

### 4. `security_test_suite.py`
**Purpose:** Comprehensive test suite for all security patches  
**Test Coverage:**
- Path traversal attempts
- File size and format validation
- Model hash verification
- Resource exhaustion prevention
- Integration tests

**Running Tests:**
```bash
python security_test_suite.py
# or
pytest security_test_suite.py -v
```

## Integration Guide

### Step 1: Update CLI Input Validation

Replace the current input validation in `cli.py`:

```python
# OLD CODE (VULNERABLE)
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(), required=False)

# NEW CODE (SECURE)
from security_patches.path_validator import SecurePathValidator

# Initialize validator with allowed paths
path_validator = SecurePathValidator([
    os.getcwd(),
    os.path.expanduser("~/Pictures"),
    "/tmp/photo_restore"
])

# In the main function:
try:
    safe_input = path_validator.validate_path(input_path)
    if output_path:
        safe_output = path_validator.validate_output_path(output_path)
    else:
        safe_output = create_secure_output_path(safe_input, os.path.dirname(safe_input))
except PathSecurityError as e:
    click.echo(f"Security error: {e}")
    sys.exit(1)
```

### Step 2: Add Image Validation

Update `image_processor.py` to validate images before processing:

```python
from security_patches.secure_image_validator import SecureImageValidator

class ImageProcessor:
    def __init__(self, config, logger):
        # ... existing code ...
        self.image_validator = SecureImageValidator()
    
    def process_image(self, input_path, output_path, **kwargs):
        # Validate image first
        try:
            validation_results = self.image_validator.validate_image(input_path)
            self.logger.info(f"Image validation passed: {validation_results['format']}")
        except ImageSecurityError as e:
            self.logger.error(f"Image validation failed: {e}")
            return False
        
        # Continue with existing processing...
```

### Step 3: Replace Dynamic Model Loading

Remove the dangerous dynamic loading from `model_loader.py` and use the secure version:

```python
# Remove this dangerous code:
# spec = importlib.util.spec_from_file_location("rrdbnet_arch", basicsr_arch_path)
# spec.loader.exec_module(rrdbnet_module)

# Replace with:
from security_patches.secure_model_loader import SecureModelLoader

class ModelManager:
    def __init__(self, config, logger):
        self.secure_loader = SecureModelLoader(config.models.cache_dir)
        # ... rest of initialization ...
    
    def load_esrgan_model(self, scale=4):
        try:
            model = self.secure_loader.load_realesrgan_model(
                'RealESRGAN_x4plus.pth', 
                scale=scale
            )
            return model
        except ModelSecurityError as e:
            self.logger.error(f"Model loading failed security check: {e}")
            return None
```

### Step 4: Update Error Handling

Sanitize error messages to prevent information disclosure:

```python
# OLD CODE (Information Disclosure)
logger.error(f"Failed to load image: {input_path}")
logger.error(f"Error processing {image_file}: {str(e)}")

# NEW CODE (Secure)
# Log detailed errors internally
logger.error(f"Failed to load image: {input_path}")

# Return generic error to user
if verbose:
    click.echo(f"Error: Failed to process image. Check logs for details.")
else:
    click.echo(f"Error: Failed to process image.")
```

### Step 5: Add Security Configuration

Create a security configuration section:

```python
@dataclass
class SecurityConfig:
    """Security configuration."""
    max_file_size: int = 20 * 1024 * 1024  # 20MB
    max_image_dimensions: tuple = (8000, 8000)
    allowed_directories: list = field(default_factory=list)
    enable_hash_verification: bool = True
    trusted_models_file: str = "trusted_models.json"
```

## Security Best Practices

1. **Always validate user input** - Never trust user-provided paths or filenames
2. **Use allowlists, not denylists** - Explicitly allow safe operations rather than blocking known bad ones
3. **Fail securely** - When in doubt, reject the operation
4. **Log security events** - Keep audit trails of security-related events
5. **Regular updates** - Keep dependencies updated and scan for vulnerabilities
6. **Principle of least privilege** - Only grant necessary permissions

## Testing Security Patches

Run the comprehensive test suite:

```bash
# Run all security tests
python security_test_suite.py

# Run specific test categories
pytest security_test_suite.py::TestPathSecurity -v
pytest security_test_suite.py::TestImageSecurity -v
pytest security_test_suite.py::TestModelSecurity -v
```

## Deployment Checklist

- [ ] Replace all path handling with SecurePathValidator
- [ ] Add image validation before processing
- [ ] Remove all dynamic imports and exec() calls
- [ ] Implement model hash verification
- [ ] Update error messages to prevent info disclosure
- [ ] Add security logging
- [ ] Run full security test suite
- [ ] Update documentation with security guidelines
- [ ] Train team on secure coding practices

## Future Enhancements

1. **Rate Limiting**: Add rate limiting for batch processing to prevent DoS
2. **Audit Logging**: Implement comprehensive security event logging
3. **Sandboxing**: Run image processing in isolated environment
4. **Network Security**: If adding network features, implement proper SSL/TLS
5. **Authentication**: For multi-user scenarios, add proper authentication

## References

- OWASP Top 10: https://owasp.org/Top10/
- CWE-22 (Path Traversal): https://cwe.mitre.org/data/definitions/22.html
- CWE-434 (Unrestricted Upload): https://cwe.mitre.org/data/definitions/434.html
- PyTorch Security: https://pytorch.org/docs/stable/notes/security.html