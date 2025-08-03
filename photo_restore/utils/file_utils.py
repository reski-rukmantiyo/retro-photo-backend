"""File system operations and utilities."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union
import hashlib

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
OUTPUT_FORMATS = {'.jpg', '.jpeg', '.png'}


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validate image file path and format.
    
    Args:
        path: Path to image file
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: {SUPPORTED_FORMATS}")
    
    return path


def generate_output_path(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    suffix: str = "_enhanced",
    format: str = "jpg"
) -> Path:
    """
    Generate output file path based on input path.
    
    Args:
        input_path: Input image path
        output_dir: Output directory (default: same as input)
        suffix: Suffix to add to filename
        format: Output format (jpg, png)
        
    Returns:
        Generated output path
    """
    input_path = Path(input_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent
    
    # Ensure format is supported
    if f".{format.lower()}" not in OUTPUT_FORMATS:
        format = "jpg"
    
    # Generate filename
    stem = input_path.stem
    if not stem.endswith(suffix):
        stem += suffix
    
    output_path = output_dir / f"{stem}.{format.lower()}"
    
    # Handle conflicts by adding counter
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{stem}_{counter}.{format.lower()}"
        counter += 1
    
    return output_path


def find_images_in_directory(
    directory: Union[str, Path],
    recursive: bool = False
) -> List[Path]:
    """
    Find all supported image files in directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    images = []
    
    if recursive:
        for ext in SUPPORTED_FORMATS:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_FORMATS:
                images.append(item)
    
    return sorted(images)


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return Path(path).stat().st_size


def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate file hash.
    
    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Safely copy file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src, dst = Path(src), Path(dst)
    
    try:
        # Create destination directory if needed
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src, dst)
        logger.debug(f"Copied {src} to {dst}")
        
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        raise


def safe_remove(path: Union[str, Path]) -> None:
    """
    Safely remove file or directory.
    
    Args:
        path: Path to remove
    """
    path = Path(path)
    
    try:
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            logger.debug(f"Removed {path}")
            
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")


def create_temp_dir(prefix: str = "photo_restore_") -> Path:
    """
    Create temporary directory.
    
    Args:
        prefix: Directory name prefix
        
    Returns:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug(f"Created temp directory: {temp_dir}")
    return temp_dir


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_available_space(path: Union[str, Path]) -> int:
    """
    Get available disk space in bytes.
    
    Args:
        path: Path to check
        
    Returns:
        Available space in bytes
    """
    return shutil.disk_usage(path).free


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


class TempFileManager:
    """Context manager for temporary file handling."""
    
    def __init__(self, prefix: str = "photo_restore_"):
        self.prefix = prefix
        self.temp_dir: Optional[Path] = None
        self.temp_files: List[Path] = []
    
    def __enter__(self) -> 'TempFileManager':
        self.temp_dir = create_temp_dir(self.prefix)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def create_temp_file(self, suffix: str = ".tmp") -> Path:
        """Create temporary file."""
        if not self.temp_dir:
            raise RuntimeError("TempFileManager not initialized")
        
        temp_file = self.temp_dir / f"temp_{len(self.temp_files)}{suffix}"
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup(self):
        """Clean up all temporary files."""
        for temp_file in self.temp_files:
            safe_remove(temp_file)
        
        if self.temp_dir:
            safe_remove(self.temp_dir)
            self.temp_dir = None
        
        self.temp_files.clear()