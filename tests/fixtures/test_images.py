"""Test image fixtures and utilities."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any


class TestImageGenerator:
    """Generate test images for different scenarios."""
    
    @staticmethod
    def create_gradient_image(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Create gradient test image."""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                image[i, j, 0] = min(255, int(255 * i / height))  # Red gradient
                image[i, j, 1] = min(255, int(255 * j / width))   # Green gradient
                image[i, j, 2] = min(255, int(255 * (i + j) / (height + width)))  # Blue gradient
        
        return image
    
    @staticmethod
    def create_noise_image(size: Tuple[int, int] = (256, 256), noise_level: float = 0.2) -> np.ndarray:
        """Create noisy test image."""
        height, width = size
        
        # Create base pattern
        base_image = TestImageGenerator.create_gradient_image(size)
        
        # Add noise
        noise = np.random.normal(0, noise_level * 255, (height, width, 3))
        noisy_image = base_image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def create_face_like_image(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Create face-like test image for face enhancement testing."""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with skin-like color
        skin_color = [220, 180, 140]  # Light skin tone
        image[:, :] = skin_color
        
        # Add face-like features using simple shapes
        center_x, center_y = width // 2, height // 2
        
        # Eyes (dark circles)
        eye_y = center_y - height // 6
        eye_size = height // 20
        
        # Left eye
        cv2.circle(image, (center_x - width // 6, eye_y), eye_size, (50, 50, 50), -1)
        # Right eye
        cv2.circle(image, (center_x + width // 6, eye_y), eye_size, (50, 50, 50), -1)
        
        # Nose (small triangle)
        nose_points = np.array([
            [center_x, center_y - height // 12],
            [center_x - width // 30, center_y + height // 12],
            [center_x + width // 30, center_y + height // 12]
        ], np.int32)
        cv2.fillPoly(image, [nose_points], (180, 140, 100))
        
        # Mouth (ellipse)
        mouth_center = (center_x, center_y + height // 6)
        mouth_size = (width // 8, height // 20)
        cv2.ellipse(image, mouth_center, mouth_size, 0, 0, 360, (150, 100, 80), -1)
        
        return image
    
    @staticmethod
    def create_pattern_image(size: Tuple[int, int] = (256, 256), pattern: str = "checkerboard") -> np.ndarray:
        """Create patterned test image."""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if pattern == "checkerboard":
            square_size = 32
            for i in range(height):
                for j in range(width):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        image[i, j] = [255, 255, 255]
                    else:
                        image[i, j] = [0, 0, 0]
        
        elif pattern == "stripes":
            stripe_width = 16
            for i in range(height):
                if (i // stripe_width) % 2 == 0:
                    image[i, :] = [255, 0, 0]  # Red stripes
                else:
                    image[i, :] = [0, 0, 255]  # Blue stripes
        
        elif pattern == "circles":
            # Create concentric circles
            center_x, center_y = width // 2, height // 2
            for radius in range(10, min(height, width) // 2, 20):
                color = [radius % 255, (radius * 2) % 255, (radius * 3) % 255]
                cv2.circle(image, (center_x, center_y), radius, color, 2)
        
        return image
    
    @staticmethod
    def create_low_quality_image(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Create low quality/blurry test image."""
        # Start with gradient image
        image = TestImageGenerator.create_gradient_image(size)
        
        # Apply heavy blur
        blurred = cv2.GaussianBlur(image, (15, 15), 5.0)
        
        # Add compression artifacts by saving/loading as low quality JPEG
        # We'll simulate this by adding quantization-like effects
        quantized = (blurred // 16) * 16  # Quantize to simulate compression
        
        return quantized.astype(np.uint8)
    
    @staticmethod
    def create_extreme_size_images() -> Dict[str, np.ndarray]:
        """Create images with extreme sizes for edge case testing."""
        images = {}
        
        # Very small image
        images['tiny'] = TestImageGenerator.create_gradient_image((8, 8))
        
        # Very wide image
        images['wide'] = TestImageGenerator.create_gradient_image((100, 1000))
        
        # Very tall image
        images['tall'] = TestImageGenerator.create_gradient_image((1000, 100))
        
        # Single pixel image
        images['pixel'] = np.array([[[255, 128, 64]]], dtype=np.uint8)
        
        return images
    
    @staticmethod
    def create_color_test_images() -> Dict[str, np.ndarray]:
        """Create images for color testing."""
        images = {}
        
        # Pure colors
        size = (100, 100, 3)
        images['red'] = np.full(size, [255, 0, 0], dtype=np.uint8)
        images['green'] = np.full(size, [0, 255, 0], dtype=np.uint8)
        images['blue'] = np.full(size, [0, 0, 255], dtype=np.uint8)
        images['white'] = np.full(size, [255, 255, 255], dtype=np.uint8)
        images['black'] = np.full(size, [0, 0, 0], dtype=np.uint8)
        
        # Grayscale gradient
        gray_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gray_value = int(255 * i / 99)
            gray_image[i, :] = [gray_value, gray_value, gray_value]
        images['grayscale'] = gray_image
        
        return images
    
    @staticmethod
    def save_test_images(output_dir: Path, formats: list = None) -> Dict[str, Dict[str, Path]]:
        """Save various test images to disk in multiple formats."""
        if formats is None:
            formats = ['jpg', 'png', 'tiff', 'bmp']
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Define test images to create
        test_images = {
            'gradient': TestImageGenerator.create_gradient_image(),
            'noise': TestImageGenerator.create_noise_image(),
            'face': TestImageGenerator.create_face_like_image(),
            'checkerboard': TestImageGenerator.create_pattern_image(pattern="checkerboard"),
            'low_quality': TestImageGenerator.create_low_quality_image()
        }
        
        # Add extreme size images
        test_images.update(TestImageGenerator.create_extreme_size_images())
        
        # Add color test images
        test_images.update(TestImageGenerator.create_color_test_images())
        
        # Save each image in each format
        for image_name, image_array in test_images.items():
            saved_files[image_name] = {}
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_array)
            
            for fmt in formats:
                file_path = output_dir / f"{image_name}.{fmt}"
                
                try:
                    if fmt.lower() in ['jpg', 'jpeg']:
                        pil_image.save(file_path, 'JPEG', quality=90)
                    elif fmt.lower() == 'png':
                        pil_image.save(file_path, 'PNG')
                    elif fmt.lower() in ['tiff', 'tif']:
                        pil_image.save(file_path, 'TIFF')
                    elif fmt.lower() == 'bmp':
                        pil_image.save(file_path, 'BMP')
                    elif fmt.lower() == 'webp':
                        pil_image.save(file_path, 'WEBP', quality=90)
                    
                    saved_files[image_name][fmt] = file_path
                    
                except Exception as e:
                    print(f"Warning: Could not save {image_name}.{fmt}: {e}")
        
        return saved_files
    
    @staticmethod
    def create_corrupted_files(output_dir: Path) -> Dict[str, Path]:
        """Create corrupted files for error testing."""
        output_dir.mkdir(parents=True, exist_ok=True)
        corrupted_files = {}
        
        # Empty file
        empty_file = output_dir / "empty.jpg"
        empty_file.touch()
        corrupted_files['empty'] = empty_file
        
        # File with wrong extension
        text_file = output_dir / "text.jpg"
        text_file.write_text("This is not an image file")
        corrupted_files['text_as_image'] = text_file
        
        # Truncated JPEG
        truncated_file = output_dir / "truncated.jpg"
        truncated_file.write_bytes(b'\xFF\xD8\xFF\xE0\x00\x10JFIF')  # JPEG header only
        corrupted_files['truncated'] = truncated_file
        
        # Binary garbage
        garbage_file = output_dir / "garbage.png"
        garbage_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)  # PNG header + garbage
        corrupted_files['garbage'] = garbage_file
        
        return corrupted_files