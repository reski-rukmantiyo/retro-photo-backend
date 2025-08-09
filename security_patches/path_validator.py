"""
Secure path validation utilities to prevent directory traversal attacks.
Implements defense-in-depth approach with multiple validation layers.
"""

import os
import re
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """Raised when path validation fails for security reasons."""
    pass


class SecurePathValidator:
    """
    Validates and sanitizes file paths to prevent security vulnerabilities.
    
    Security features:
    - Prevents directory traversal attacks
    - Validates against allowed directories
    - Checks for symbolic links
    - Normalizes paths to prevent bypass attempts
    """
    
    # Patterns that indicate potential security issues
    DANGEROUS_PATTERNS = [
        r'\.\.',  # Parent directory references
        r'~',      # Home directory expansion
        r'\$',     # Variable expansion
        r'%',      # Windows environment variables
        r'^/',     # Absolute paths on Unix
        r'^[A-Za-z]:',  # Absolute paths on Windows
        r'\\\\',   # UNC paths
    ]
    
    def __init__(self, allowed_base_paths: list[str] = None):
        """
        Initialize validator with allowed base paths.
        
        Args:
            allowed_base_paths: List of directories where operations are allowed
        """
        self.allowed_base_paths = []
        if allowed_base_paths:
            for path in allowed_base_paths:
                self.add_allowed_path(path)
    
    def add_allowed_path(self, path: str) -> None:
        """Add a directory to the allowed paths list."""
        abs_path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            self.allowed_base_paths.append(abs_path)
            logger.info(f"Added allowed path: {abs_path}")
        else:
            raise ValueError(f"Invalid directory path: {path}")
    
    def validate_path(self, user_path: str, base_path: Optional[str] = None) -> str:
        """
        Validate and return a safe absolute path.
        
        Args:
            user_path: Path provided by user
            base_path: Optional base directory to resolve relative paths
            
        Returns:
            Safe, validated absolute path
            
        Raises:
            PathSecurityError: If path validation fails
        """
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, user_path):
                raise PathSecurityError(
                    f"Potentially dangerous path pattern detected: {pattern}"
                )
        
        # Normalize the path
        if base_path:
            # Resolve relative to base path
            normalized = os.path.normpath(os.path.join(base_path, user_path))
        else:
            # Treat as relative to current directory
            normalized = os.path.normpath(user_path)
        
        # Convert to absolute path
        abs_path = os.path.abspath(normalized)
        
        # Check if path is within allowed directories
        if self.allowed_base_paths:
            is_allowed = False
            for allowed_path in self.allowed_base_paths:
                if abs_path.startswith(allowed_path):
                    is_allowed = True
                    break
            
            if not is_allowed:
                raise PathSecurityError(
                    f"Path outside allowed directories: {abs_path}"
                )
        
        # Check for symbolic links (could lead outside allowed paths)
        if os.path.exists(abs_path) and os.path.islink(abs_path):
            real_path = os.path.realpath(abs_path)
            # Recursively validate the real path
            return self.validate_path(real_path)
        
        return abs_path
    
    def validate_output_path(self, output_path: str, base_path: Optional[str] = None) -> str:
        """
        Validate path for output operations with additional checks.
        
        Args:
            output_path: Desired output path
            base_path: Optional base directory
            
        Returns:
            Safe, validated output path
        """
        # First perform standard validation
        safe_path = self.validate_path(output_path, base_path)
        
        # Additional checks for output paths
        parent_dir = os.path.dirname(safe_path)
        
        # Ensure parent directory exists or can be created
        if not os.path.exists(parent_dir):
            try:
                # Check if we have permission to create it
                test_path = Path(parent_dir)
                test_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise PathSecurityError(
                    f"Cannot create output directory: {parent_dir}: {e}"
                )
        
        # Check write permissions
        if os.path.exists(parent_dir) and not os.access(parent_dir, os.W_OK):
            raise PathSecurityError(
                f"No write permission for directory: {parent_dir}"
            )
        
        return safe_path
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize filename to prevent security issues.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Sanitized filename
        """
        # Remove any path components
        filename = os.path.basename(filename)
        
        # Remove potentially dangerous characters
        # Allow only alphanumeric, dash, underscore, and dot
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Prevent hidden files
        if sanitized.startswith('.'):
            sanitized = '_' + sanitized[1:]
        
        # Ensure it has a valid extension
        if '.' not in sanitized:
            sanitized += '.jpg'  # Default extension
        
        # Limit length
        max_length = 255
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:max_length - len(ext)] + ext
        
        return sanitized


def create_secure_output_path(
    input_path: str,
    output_dir: str,
    suffix: str = "_enhanced",
    validator: Optional[SecurePathValidator] = None
) -> str:
    """
    Create a secure output path based on input file.
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename
        validator: Optional path validator instance
        
    Returns:
        Secure output path
    """
    if not validator:
        validator = SecurePathValidator()
    
    # Validate input path
    safe_input = validator.validate_path(input_path)
    
    # Extract and sanitize filename
    input_filename = os.path.basename(safe_input)
    name, ext = os.path.splitext(input_filename)
    
    # Create new filename
    output_filename = validator.validate_filename(f"{name}{suffix}{ext}")
    
    # Validate output directory
    safe_output_dir = validator.validate_output_path(output_dir)
    
    # Combine to create final path
    output_path = os.path.join(safe_output_dir, output_filename)
    
    # Final validation
    return validator.validate_output_path(output_path)


# Example usage for the photo restoration CLI
def integrate_with_cli():
    """Example of how to integrate with the existing CLI."""
    
    # Initialize validator with allowed directories
    validator = SecurePathValidator([
        os.getcwd(),  # Current directory
        os.path.expanduser("~/Pictures"),  # User's pictures directory
        "/tmp/photo_restore"  # Temporary processing directory
    ])
    
    # In the CLI command handler:
    def process_with_validation(input_path: str, output_path: Optional[str] = None):
        try:
            # Validate input
            safe_input = validator.validate_path(input_path)
            
            # Generate or validate output path
            if output_path:
                safe_output = validator.validate_output_path(output_path)
            else:
                # Auto-generate safe output path
                safe_output = create_secure_output_path(
                    safe_input,
                    os.path.dirname(safe_input),
                    validator=validator
                )
            
            # Proceed with processing using validated paths
            logger.info(f"Processing: {safe_input} -> {safe_output}")
            
            return safe_input, safe_output
            
        except PathSecurityError as e:
            logger.error(f"Security violation: {e}")
            raise
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise


if __name__ == "__main__":
    # Test the validator
    validator = SecurePathValidator(["/tmp/test"])
    
    # Test cases
    test_paths = [
        "normal_file.jpg",
        "../../../etc/passwd",  # Should fail
        "~/secret_file.jpg",    # Should fail
        "/etc/shadow",          # Should fail
        "valid_dir/image.png",
        "file_with_%20spaces.jpg"
    ]
    
    for path in test_paths:
        try:
            result = validator.validate_path(path, "/tmp/test")
            print(f"✓ Valid: {path} -> {result}")
        except PathSecurityError as e:
            print(f"✗ Blocked: {path} - {e}")