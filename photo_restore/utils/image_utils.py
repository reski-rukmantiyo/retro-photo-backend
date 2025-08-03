"""Image manipulation and validation utilities."""

import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags
from pathlib import Path
from typing import Tuple, Optional, Union, Any
import io

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file with proper format handling.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array (BGR format for OpenCV compatibility)
        
    Raises:
        ValueError: If image cannot be loaded
    """
    path = Path(path)
    
    try:
        # Use PIL for better format support and EXIF handling
        with Image.open(path) as pil_img:
            # Handle EXIF orientation
            pil_img = ImageOps.exif_transpose(pil_img)
            
            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert PIL to OpenCV format (RGB -> BGR)
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        logger.debug(f"Loaded image: {path} ({img_bgr.shape})")
        return img_bgr
        
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
    format: Optional[str] = None
) -> None:
    """
    Save image to file with optimal settings.
    
    Args:
        image: Image as numpy array (BGR format)
        path: Output path
        quality: JPEG quality (1-100)
        format: Force specific format (jpg, png)
        
    Raises:
        ValueError: If image cannot be saved
    """
    path = Path(path)
    
    try:
        # Determine format
        if format:
            ext = f".{format.lower()}"
        else:
            ext = path.suffix.lower()
        
        # Ensure output directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_img = Image.fromarray(image_rgb.astype(np.uint8))
        
        # Save with appropriate settings
        if ext in ['.jpg', '.jpeg']:
            pil_img.save(
                path,
                'JPEG',
                quality=quality,
                optimize=True,
                progressive=True
            )
        elif ext == '.png':
            pil_img.save(
                path,
                'PNG',
                optimize=True
            )
        else:
            # Fallback to OpenCV
            cv2.imwrite(str(path), image)
        
        logger.debug(f"Saved image: {path}")
        
    except Exception as e:
        raise ValueError(f"Failed to save image to {path}: {e}")


def get_image_info(path: Union[str, Path]) -> dict:
    """
    Get image metadata and properties.
    
    Args:
        path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    path = Path(path)
    
    try:
        with Image.open(path) as img:
            info = {
                'path': str(path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size': path.stat().st_size,
                'has_exif': bool(img.getexif()),
            }
            
            # Add EXIF orientation if available
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                if exif and 274 in exif:  # Orientation tag
                    info['orientation'] = exif[274]
            
            return info
            
    except Exception as e:
        logger.warning(f"Failed to get image info for {path}: {e}")
        return {'path': str(path), 'error': str(e)}


def validate_image_dimensions(
    image: np.ndarray,
    min_size: int = 100,
    max_size: int = 4096
) -> None:
    """
    Validate image dimensions.
    
    Args:
        image: Image as numpy array
        min_size: Minimum dimension in pixels
        max_size: Maximum dimension in pixels
        
    Raises:
        ValueError: If dimensions are invalid
    """
    height, width = image.shape[:2]
    
    if min(width, height) < min_size:
        raise ValueError(f"Image too small: {width}x{height} (minimum: {min_size})")
    
    if max(width, height) > max_size:
        raise ValueError(f"Image too large: {width}x{height} (maximum: {max_size})")


def resize_image(
    image: np.ndarray,
    max_size: int = 2048,
    maintain_aspect: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    
    if max(width, height) <= max_size:
        return image, 1.0
    
    if maintain_aspect:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = new_height = max_size
        scale = max_size / max(width, height)
    
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )
    
    logger.debug(f"Resized image: {width}x{height} -> {new_width}x{new_height} (scale: {scale:.2f})")
    return resized, scale


def upscale_image(
    image: np.ndarray,
    scale_factor: int = 2,
    method: str = 'cubic'
) -> np.ndarray:
    """
    Upscale image using traditional interpolation.
    
    Args:
        image: Input image
        scale_factor: Upscaling factor
        method: Interpolation method (linear, cubic, lanczos)
        
    Returns:
        Upscaled image
    """
    height, width = image.shape[:2]
    new_size = (width * scale_factor, height * scale_factor)
    
    interpolation_methods = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    method_cv = interpolation_methods.get(method, cv2.INTER_CUBIC)
    upscaled = cv2.resize(image, new_size, interpolation=method_cv)
    
    logger.debug(f"Upscaled image: {width}x{height} -> {new_size[0]}x{new_size[1]}")
    return upscaled


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image
    
    # Handle different data types
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            # Assume 0-1 range
            normalized = (image * 255).astype(np.uint8)
        else:
            # Normalize to 0-255
            normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        # Convert to uint8
        normalized = cv2.convertScaleAbs(image)
    
    return normalized


def apply_sharpening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Apply unsharp masking for image sharpening.
    
    Args:
        image: Input image
        strength: Sharpening strength (0.0-2.0)
        
    Returns:
        Sharpened image
    """
    if strength <= 0:
        return image
    
    # Create gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    
    # Create unsharp mask
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return sharpened


def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """
    Enhance image contrast and brightness.
    
    Args:
        image: Input image
        alpha: Contrast multiplier (1.0 = no change)
        beta: Brightness offset
        
    Returns:
        Enhanced image
    """
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 32
) -> list:
    """
    Split image into overlapping tiles for processing.
    
    Args:
        image: Input image
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        List of (tile, position) tuples
    """
    height, width = image.shape[:2]
    tiles = []
    
    stride = tile_size - overlap
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Calculate tile boundaries
            x1 = x
            y1 = y
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)
            
            # Extract tile
            tile = image[y1:y2, x1:x2]
            
            # Pad if necessary
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            tiles.append((tile, (x1, y1, x2, y2)))
    
    return tiles


def merge_tiles(
    tiles: list,
    output_shape: Tuple[int, int, int],
    overlap: int = 32
) -> np.ndarray:
    """
    Merge processed tiles back into full image.
    
    Args:
        tiles: List of (processed_tile, position) tuples
        output_shape: Shape of output image (H, W, C)
        overlap: Overlap between original tiles
        
    Returns:
        Merged image
    """
    height, width, channels = output_shape
    result = np.zeros((height, width, channels), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    
    for tile, (x1, y1, x2, y2) in tiles:
        # Calculate actual tile size
        tile_h = y2 - y1
        tile_w = x2 - x1
        
        # Extract the relevant part of the processed tile
        processed_tile = tile[:tile_h, :tile_w]
        
        # Create weight map (higher weights in center, lower at edges)
        weight_map = np.ones((tile_h, tile_w), dtype=np.float32)
        
        if overlap > 0:
            # Apply tapering at edges
            for i in range(min(overlap, tile_h // 2)):
                weight_map[i, :] *= i / overlap
                weight_map[-(i+1), :] *= i / overlap
            
            for j in range(min(overlap, tile_w // 2)):
                weight_map[:, j] *= j / overlap
                weight_map[:, -(j+1)] *= j / overlap
        
        # Add to result with weights
        result[y1:y2, x1:x2] += processed_tile * weight_map[:, :, np.newaxis]
        weights[y1:y2, x1:x2] += weight_map
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    result = result / weights[:, :, np.newaxis]
    
    return result.astype(np.uint8)