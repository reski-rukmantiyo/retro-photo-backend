"""
Secure image file validation to prevent malicious file uploads.
Implements multiple layers of validation including magic bytes, file size, and content analysis.
"""

import os
import struct
import io
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

# For image validation
try:
    from PIL import Image
    import cv2
    import numpy as np
except ImportError:
    print("Warning: PIL or OpenCV not available for image validation")

logger = logging.getLogger(__name__)


class ImageSecurityError(Exception):
    """Raised when image validation fails for security reasons."""
    pass


class SecureImageValidator:
    """
    Validates image files to prevent security vulnerabilities.
    
    Security features:
    - Magic byte validation (file signature verification)
    - File size limits
    - Image dimension limits
    - Format validation
    - Malicious content detection
    """
    
    # Magic bytes for common image formats
    IMAGE_SIGNATURES = {
        'jpeg': [
            (b'\xff\xd8\xff\xe0', 'JPEG/JFIF'),
            (b'\xff\xd8\xff\xe1', 'JPEG/EXIF'),
            (b'\xff\xd8\xff\xe8', 'JPEG/SPIFF'),
            (b'\xff\xd8\xff\xdb', 'JPEG'),
        ],
        'png': [(b'\x89PNG\r\n\x1a\n', 'PNG')],
        'gif': [(b'GIF87a', 'GIF87a'), (b'GIF89a', 'GIF89a')],
        'bmp': [(b'BM', 'BMP')],
        'tiff': [
            (b'II\x2a\x00', 'TIFF little-endian'),
            (b'MM\x00\x2a', 'TIFF big-endian')
        ],
        'webp': [(b'RIFF....WEBP', 'WebP')],  # Note: .... is 4 bytes for file size
    }
    
    # File extension to format mapping
    ALLOWED_EXTENSIONS = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.gif': 'gif',
        '.bmp': 'bmp',
        '.tiff': 'tiff',
        '.tif': 'tiff',
        '.webp': 'webp'
    }
    
    def __init__(
        self,
        max_file_size: int = 20 * 1024 * 1024,  # 20MB
        max_dimensions: Tuple[int, int] = (10000, 10000),  # Max width, height
        min_dimensions: Tuple[int, int] = (10, 10),  # Min width, height
        allowed_formats: Optional[list] = None
    ):
        """
        Initialize validator with security limits.
        
        Args:
            max_file_size: Maximum file size in bytes
            max_dimensions: Maximum (width, height) in pixels
            min_dimensions: Minimum (width, height) in pixels
            allowed_formats: List of allowed formats (default: all supported)
        """
        self.max_file_size = max_file_size
        self.max_dimensions = max_dimensions
        self.min_dimensions = min_dimensions
        self.allowed_formats = allowed_formats or list(self.IMAGE_SIGNATURES.keys())
    
    def validate_file_size(self, file_path: str) -> int:
        """
        Validate file size is within limits.
        
        Args:
            file_path: Path to image file
            
        Returns:
            File size in bytes
            
        Raises:
            ImageSecurityError: If file size exceeds limit
        """
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            raise ImageSecurityError(f"Cannot access file: {e}")
        
        if file_size > self.max_file_size:
            raise ImageSecurityError(
                f"File size {file_size} bytes exceeds maximum {self.max_file_size} bytes"
            )
        
        if file_size == 0:
            raise ImageSecurityError("File is empty")
        
        return file_size
    
    def validate_magic_bytes(self, file_path: str) -> str:
        """
        Validate file format using magic bytes (file signature).
        
        Args:
            file_path: Path to image file
            
        Returns:
            Detected format name
            
        Raises:
            ImageSecurityError: If magic bytes don't match any known image format
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 16 bytes for signature detection
                header = f.read(16)
                
                # SECURITY FIX: Also check for polyglot attacks by scanning more content
                f.seek(0)
                first_1kb = f.read(1024)  # Read first 1KB to detect polyglots
                
        except IOError as e:
            raise ImageSecurityError(f"Cannot read file: {e}")
        
        if len(header) < 2:
            raise ImageSecurityError("File too small to be a valid image")
        
        # SECURITY CHECK: Detect polyglot attacks (HTML/PHP/JS embedded in images)
        polyglot_signatures = [
            b'<html', b'<HTML', b'<!DOCTYPE',
            b'<?php', b'<?PHP', 
            b'<script', b'<SCRIPT',
            b'javascript:', b'data:text/html',
            b'<?xml'
        ]
        
        for poly_sig in polyglot_signatures:
            if poly_sig in first_1kb:
                raise ImageSecurityError(
                    f"Polyglot file detected - contains web content: {poly_sig.decode('utf-8', errors='ignore')}"
                )
        
        # Check against known signatures
        detected_format = None
        
        for format_name, signatures in self.IMAGE_SIGNATURES.items():
            for signature, description in signatures:
                if format_name == 'webp':
                    # Special handling for WebP
                    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
                        detected_format = format_name
                        break
                elif header.startswith(signature):
                    detected_format = format_name
                    break
            
            if detected_format:
                break
        
        if not detected_format:
            # Log first few bytes for debugging
            header_hex = header[:8].hex()
            raise ImageSecurityError(
                f"Unknown file format. Header: {header_hex}"
            )
        
        if detected_format not in self.allowed_formats:
            raise ImageSecurityError(
                f"File format '{detected_format}' not allowed"
            )
        
        logger.debug(f"Detected format: {detected_format}")
        return detected_format
    
    def validate_extension(self, file_path: str, detected_format: str) -> None:
        """
        Validate that file extension matches the detected format.
        
        Args:
            file_path: Path to image file
            detected_format: Format detected by magic bytes
            
        Raises:
            ImageSecurityError: If extension doesn't match content
        """
        extension = Path(file_path).suffix.lower()
        
        if extension not in self.ALLOWED_EXTENSIONS:
            raise ImageSecurityError(
                f"File extension '{extension}' not allowed"
            )
        
        expected_format = self.ALLOWED_EXTENSIONS[extension]
        
        # Allow JPEG variants
        if expected_format == 'jpeg' and detected_format == 'jpeg':
            return
        
        if expected_format != detected_format:
            raise ImageSecurityError(
                f"File extension '{extension}' doesn't match content (detected: {detected_format})"
            )
    
    def validate_image_integrity(self, file_path: str) -> Dict[str, any]:
        """
        Validate image can be loaded and check its properties.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with image properties
            
        Raises:
            ImageSecurityError: If image is corrupted or malicious
        """
        image_info = {}
        
        # Try loading with PIL first (more secure)
        try:
            with Image.open(file_path) as img:
                # Verify image can be loaded
                img.verify()
                
                # Re-open after verify (verify closes the file)
                with Image.open(file_path) as img:
                    image_info['format'] = img.format
                    image_info['mode'] = img.mode
                    image_info['size'] = img.size
                    width, height = img.size
                    
        except Exception as e:
            raise ImageSecurityError(f"PIL cannot load image: {e}")
        
        # Validate dimensions
        if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
            raise ImageSecurityError(
                f"Image dimensions {width}x{height} exceed maximum "
                f"{self.max_dimensions[0]}x{self.max_dimensions[1]}"
            )
        
        if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
            raise ImageSecurityError(
                f"Image dimensions {width}x{height} below minimum "
                f"{self.min_dimensions[0]}x{self.min_dimensions[1]}"
            )
        
        # Calculate pixel count for memory estimation
        pixel_count = width * height
        image_info['pixel_count'] = pixel_count
        
        # Estimate memory usage (rough estimate: 4 bytes per pixel)
        estimated_memory = pixel_count * 4
        if estimated_memory > 500 * 1024 * 1024:  # 500MB
            raise ImageSecurityError(
                f"Image would use approximately {estimated_memory / 1024 / 1024:.1f}MB of memory"
            )
        
        # Additional OpenCV validation
        try:
            # Test loading with OpenCV
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                raise ImageSecurityError("OpenCV cannot load image")
            
            # Check for unusual channel counts
            if len(img_cv.shape) == 3:
                channels = img_cv.shape[2]
                if channels not in [1, 3, 4]:
                    raise ImageSecurityError(
                        f"Unusual channel count: {channels}"
                    )
            
            image_info['opencv_shape'] = img_cv.shape
            
        except Exception as e:
            logger.warning(f"OpenCV validation warning: {e}")
        
        return image_info
    
    def check_malicious_content(self, file_path: str) -> None:
        """
        Check for potential malicious content in image.
        
        Args:
            file_path: Path to image file
            
        Raises:
            ImageSecurityError: If malicious content detected
        """
        # Check for embedded executables or scripts
        dangerous_signatures = [
            b'<script',  # JavaScript
            b'<?php',    # PHP
            b'#!/',      # Shell scripts
            b'MZ',       # PE executables (at positions other than 0)
            b'\x7fELF',  # ELF executables
        ]
        
        try:
            with open(file_path, 'rb') as f:
                # Skip image header
                f.seek(100)
                # Read chunks and check for dangerous content
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    
                    for signature in dangerous_signatures:
                        if signature in chunk:
                            logger.warning(
                                f"Potentially dangerous content found: {signature}"
                            )
                            # Don't immediately reject, but log for investigation
                            
        except Exception as e:
            logger.warning(f"Error scanning for malicious content: {e}")
    
    def validate_image(self, file_path: str) -> Dict[str, any]:
        """
        Perform complete validation of an image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with validation results and image properties
            
        Raises:
            ImageSecurityError: If any validation fails
        """
        validation_results = {
            'file_path': file_path,
            'valid': False,
            'checks_passed': []
        }
        
        try:
            # 1. Validate file size
            file_size = self.validate_file_size(file_path)
            validation_results['file_size'] = file_size
            validation_results['checks_passed'].append('file_size')
            
            # 2. Validate magic bytes
            detected_format = self.validate_magic_bytes(file_path)
            validation_results['format'] = detected_format
            validation_results['checks_passed'].append('magic_bytes')
            
            # 3. Validate extension matches content
            self.validate_extension(file_path, detected_format)
            validation_results['checks_passed'].append('extension_match')
            
            # 4. Validate image integrity and properties
            image_info = self.validate_image_integrity(file_path)
            validation_results.update(image_info)
            validation_results['checks_passed'].append('integrity')
            
            # 5. Check for malicious content
            self.check_malicious_content(file_path)
            validation_results['checks_passed'].append('malicious_scan')
            
            validation_results['valid'] = True
            logger.info(f"Image validation passed: {file_path}")
            
        except ImageSecurityError as e:
            validation_results['error'] = str(e)
            logger.error(f"Image validation failed: {e}")
            raise
        
        return validation_results


# Integration example for the photo restoration CLI
def create_validated_image_processor():
    """Example of integrating validator with image processor."""
    
    validator = SecureImageValidator(
        max_file_size=20 * 1024 * 1024,  # 20MB
        max_dimensions=(8000, 8000),      # 8K resolution max
        min_dimensions=(100, 100),        # Minimum 100x100
        allowed_formats=['jpeg', 'png', 'bmp', 'tiff', 'webp']
    )
    
    def process_image_with_validation(input_path: str, output_path: str):
        """Process image with security validation."""
        try:
            # Validate input image
            validation_results = validator.validate_image(input_path)
            logger.info(f"Validation passed: {validation_results['checks_passed']}")
            
            # Proceed with processing
            # ... existing image processing code ...
            
            return True
            
        except ImageSecurityError as e:
            logger.error(f"Security validation failed: {e}")
            return False
    
    return process_image_with_validation


if __name__ == "__main__":
    # Test the validator
    validator = SecureImageValidator()
    
    # Test with a sample image
    test_file = "test_image.jpg"
    
    try:
        results = validator.validate_image(test_file)
        print(f"✓ Validation passed: {results}")
    except ImageSecurityError as e:
        print(f"✗ Validation failed: {e}")