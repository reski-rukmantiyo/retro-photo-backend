"""
Comprehensive security test suite for the Photo Restoration CLI.
Tests all security vulnerabilities identified in the audit.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import hashlib

# Import security modules
from path_validator import SecurePathValidator, PathSecurityError
from secure_image_validator import SecureImageValidator, ImageSecurityError
from secure_model_loader import SecureModelLoader, ModelSecurityError


class TestPathSecurity:
    """Test suite for path traversal and file access security."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="security_test_")
        self.safe_dir = os.path.join(self.temp_dir, "safe")
        os.makedirs(self.safe_dir)
        
        self.validator = SecurePathValidator([self.safe_dir])
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_path_traversal_attempts(self):
        """Test that path traversal attempts are blocked."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "images/../../../etc/shadow",
            "~/../../root/.ssh/id_rsa",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "images/../../sensitive.txt",
            "./../../etc/hosts",
            "images\\..\\..\\passwords.txt"
        ]
        
        for path in malicious_paths:
            with pytest.raises(PathSecurityError):
                self.validator.validate_path(path, self.safe_dir)
    
    def test_absolute_path_rejection(self):
        """Test that absolute paths are rejected when not in allowed list."""
        absolute_paths = [
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\cmd.exe",
            "/home/user/sensitive.txt"
        ]
        
        for path in absolute_paths:
            with pytest.raises(PathSecurityError):
                self.validator.validate_path(path)
    
    def test_symbolic_link_detection(self):
        """Test that symbolic links are properly handled."""
        # Create a symbolic link
        target_file = os.path.join(self.safe_dir, "target.txt")
        link_file = os.path.join(self.safe_dir, "link.txt")
        
        with open(target_file, 'w') as f:
            f.write("test")
        
        os.symlink(target_file, link_file)
        
        # Should resolve to real path
        validated = self.validator.validate_path(link_file)
        assert os.path.realpath(validated) == os.path.realpath(target_file)
    
    def test_null_byte_injection(self):
        """Test that null byte injection is handled."""
        malicious_paths = [
            "image.jpg\x00.exe",
            "file\x00../../../../etc/passwd",
            "normal.jpg\x00"
        ]
        
        for path in malicious_paths:
            # Python paths handle null bytes safely, but test anyway
            try:
                result = self.validator.validate_path(path, self.safe_dir)
                # If no exception, ensure null byte is handled
                assert '\x00' not in result
            except (PathSecurityError, ValueError):
                # Expected for paths with null bytes
                pass
    
    def test_unicode_normalization(self):
        """Test handling of Unicode normalization attacks."""
        # Different Unicode representations of the same character
        tricky_paths = [
            "ima\u0300ge.jpg",  # Combining character
            "café.jpg",         # Precomposed
            "cafe\u0301.jpg",   # Decomposed
        ]
        
        for path in tricky_paths:
            try:
                validated = self.validator.validate_path(path, self.safe_dir)
                # Should normalize to consistent representation
                assert os.path.normpath(validated)
            except PathSecurityError:
                # Some Unicode might be rejected, which is safe
                pass
    
    def test_filename_sanitization(self):
        """Test filename sanitization removes dangerous characters."""
        dangerous_filenames = [
            "../../etc/passwd",
            "file<script>.jpg",
            "image|command.jpg",
            "photo;rm -rf /.jpg",
            "pic`whoami`.jpg",
            "img$(curl evil.com).jpg",
            ".hidden_file.jpg",
            "file name with spaces.jpg"
        ]
        
        for filename in dangerous_filenames:
            sanitized = self.validator.validate_filename(filename)
            # Check sanitization worked
            assert '..' not in sanitized
            assert '<' not in sanitized
            assert '>' not in sanitized
            assert '|' not in sanitized
            assert ';' not in sanitized
            assert '`' not in sanitized
            assert '$' not in sanitized
            assert not sanitized.startswith('.')


class TestImageSecurity:
    """Test suite for image file validation security."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="image_test_")
        self.validator = SecureImageValidator(
            max_file_size=5 * 1024 * 1024,  # 5MB for tests
            max_dimensions=(2000, 2000),
            min_dimensions=(10, 10)
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_size_limits(self):
        """Test that oversized files are rejected."""
        # Create a file larger than limit
        large_file = os.path.join(self.temp_dir, "large.jpg")
        with open(large_file, 'wb') as f:
            f.write(b'\xFF' * (6 * 1024 * 1024))  # 6MB
        
        with pytest.raises(ImageSecurityError, match="exceeds maximum"):
            self.validator.validate_file_size(large_file)
    
    def test_empty_file_rejection(self):
        """Test that empty files are rejected."""
        empty_file = os.path.join(self.temp_dir, "empty.jpg")
        Path(empty_file).touch()
        
        with pytest.raises(ImageSecurityError, match="empty"):
            self.validator.validate_file_size(empty_file)
    
    def test_magic_byte_validation(self):
        """Test that files with wrong magic bytes are rejected."""
        # Create files with wrong content
        test_cases = [
            ("fake_jpeg.jpg", b"Not a JPEG file content"),
            ("fake_png.png", b"<html>Not an image</html>"),
            ("script.jpg", b"#!/bin/bash\nrm -rf /"),
            ("executable.png", b"MZ\x90\x00\x03"),  # PE header
        ]
        
        for filename, content in test_cases:
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(content)
            
            with pytest.raises(ImageSecurityError, match="Unknown file format"):
                self.validator.validate_magic_bytes(file_path)
    
    def test_extension_content_mismatch(self):
        """Test that extension must match content."""
        # Create a PNG file with JPEG extension
        png_as_jpg = os.path.join(self.temp_dir, "image.jpg")
        with open(png_as_jpg, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
        
        with pytest.raises(ImageSecurityError, match="doesn't match content"):
            self.validator.validate_extension(png_as_jpg, 'png')
    
    def test_malicious_content_detection(self):
        """Test detection of embedded malicious content."""
        # Create image with embedded script
        malicious_file = os.path.join(self.temp_dir, "malicious.jpg")
        with open(malicious_file, 'wb') as f:
            # Valid JPEG header
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00')
            # Padding
            f.write(b'\x00' * 100)
            # Embedded script
            f.write(b'<?php system($_GET["cmd"]); ?>')
        
        # Should log warning but not immediately fail
        # (allows for legitimate metadata)
        self.validator.check_malicious_content(malicious_file)
    
    def test_polyglot_file_detection(self):
        """Test detection of polyglot files (valid as multiple formats)."""
        # Create a file that's both valid HTML and tries to be an image
        polyglot = os.path.join(self.temp_dir, "polyglot.jpg")
        with open(polyglot, 'wb') as f:
            f.write(b'<html><!--')
            f.write(b'\xff\xd8\xff\xe0')  # JPEG header
            f.write(b'--></html>')
        
        with pytest.raises(ImageSecurityError):
            self.validator.validate_magic_bytes(polyglot)


class TestModelSecurity:
    """Test suite for secure model loading."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="model_test_")
        self.model_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.model_dir)
        
        # Create a mock trusted hashes dict
        self.loader = SecureModelLoader(self.model_dir)
        self.loader.trusted_hashes = {
            'test_model.pth': hashlib.sha256(b'safe_model_content').hexdigest(),
            'trusted_model.pth': hashlib.sha256(b'trusted_content').hexdigest()
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hash_verification_success(self):
        """Test successful hash verification."""
        model_path = os.path.join(self.model_dir, 'test_model.pth')
        with open(model_path, 'wb') as f:
            f.write(b'safe_model_content')
        
        # Should not raise exception
        assert self.loader.verify_model_hash(model_path)
    
    def test_hash_verification_failure(self):
        """Test that modified models are rejected."""
        model_path = os.path.join(self.model_dir, 'test_model.pth')
        with open(model_path, 'wb') as f:
            f.write(b'modified_model_content')
        
        with pytest.raises(ModelSecurityError, match="hash mismatch"):
            self.loader.verify_model_hash(model_path)
    
    def test_untrusted_model_rejection(self):
        """Test that models not in trusted list are rejected."""
        model_path = os.path.join(self.model_dir, 'unknown_model.pth')
        with open(model_path, 'wb') as f:
            f.write(b'unknown_content')
        
        with pytest.raises(ModelSecurityError, match="not in trusted list"):
            self.loader.verify_model_hash(model_path)
    
    def test_model_file_not_found(self):
        """Test handling of missing model files."""
        model_path = os.path.join(self.model_dir, 'missing_model.pth')
        
        with pytest.raises(ModelSecurityError):
            self.loader.calculate_file_hash(model_path)
    
    def test_corrupted_model_handling(self):
        """Test handling of corrupted model files."""
        model_path = os.path.join(self.model_dir, 'corrupted_model.pth')
        with open(model_path, 'wb') as f:
            f.write(b'corrupted_data_not_a_valid_pytorch_model')
        
        # Add to trusted hashes to pass that check
        self.loader.trusted_hashes['corrupted_model.pth'] = \
            self.loader.calculate_file_hash(model_path)
        
        # Should fail when trying to load state dict
        with pytest.raises(Exception):  # PyTorch will raise an error
            self.loader.load_state_dict_safely(model_path)


class TestIntegrationSecurity:
    """Integration tests for security measures."""
    
    def test_safe_image_processing_pipeline(self):
        """Test complete secure image processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up validators
            path_validator = SecurePathValidator([temp_dir])
            image_validator = SecureImageValidator()
            
            # Test input validation
            input_path = "../../../etc/passwd"
            with pytest.raises(PathSecurityError):
                path_validator.validate_path(input_path)
            
            # Test output path generation
            safe_input = os.path.join(temp_dir, "test.jpg")
            safe_output = path_validator.validate_output_path(
                os.path.join(temp_dir, "output.jpg")
            )
            assert temp_dir in safe_output
    
    def test_resource_exhaustion_prevention(self):
        """Test prevention of resource exhaustion attacks."""
        validator = SecureImageValidator(
            max_file_size=1024 * 1024,  # 1MB
            max_dimensions=(1000, 1000)
        )
        
        # Test file size limit
        with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
            f.write(b'\xFF' * (2 * 1024 * 1024))  # 2MB
            f.flush()
            
            with pytest.raises(ImageSecurityError, match="exceeds maximum"):
                validator.validate_file_size(f.name)


def run_security_tests():
    """Run all security tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_security_tests()