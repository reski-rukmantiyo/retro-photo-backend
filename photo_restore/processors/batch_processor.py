"""Batch processing for multiple images."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from tqdm import tqdm

from .image_processor import ImageProcessor
from ..utils.config import Config


class BatchProcessor:
    """Handles batch processing of multiple images."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """Initialize batch processor."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.image_processor = ImageProcessor(config, logger)
        
        # Batch statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'skipped_images': 0,
            'processing_time': 0.0
        }
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        quality: str = 'balanced',
        upscale: int = 4,
        face_enhance: bool = True,
        output_format: str = 'jpg',
        recursive: bool = True
    ) -> int:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            quality: Quality level (fast/balanced/best)
            upscale: Upscale factor (2 or 4)
            face_enhance: Enable face enhancement
            output_format: Output format (jpg/png)
            recursive: Process subdirectories recursively
            
        Returns:
            Number of successfully processed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return 0
        
        # Find all image files
        image_files = self._find_image_files(input_path, recursive)
        if not image_files:
            self.logger.warning(f"No supported image files found in: {input_dir}")
            return 0
        
        self.stats['total_images'] = len(image_files)
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process images with progress bar
        success_count = 0
        
        with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
            for image_file in image_files:
                try:
                    # Generate output path maintaining directory structure
                    relative_path = image_file.relative_to(input_path)
                    output_file = output_path / relative_path.with_suffix(f'.{output_format}')
                    
                    # Skip if output already exists
                    if output_file.exists():
                        self.logger.debug(f"Skipping existing file: {output_file}")
                        self.stats['skipped_images'] += 1
                        pbar.update(1)
                        continue
                    
                    # Update progress bar description
                    pbar.set_description(f"Processing {image_file.name}")
                    
                    # Process image
                    success = self.image_processor.process_image(
                        input_path=str(image_file),
                        output_path=str(output_file),
                        quality=quality,
                        upscale=upscale,
                        face_enhance=face_enhance,
                        output_format=output_format
                    )
                    
                    if success:
                        success_count += 1
                        self.stats['processed_images'] += 1
                        self.logger.debug(f"✅ {image_file.name}")
                    else:
                        self.stats['failed_images'] += 1
                        self.logger.warning(f"❌ Failed: {image_file.name}")
                    
                except Exception as e:
                    self.stats['failed_images'] += 1
                    self.logger.error(f"Error processing {image_file}: {str(e)}")
                
                finally:
                    pbar.update(1)
        
        # Log final statistics
        self._log_batch_results()
        return success_count
    
    def _find_image_files(self, directory: Path, recursive: bool = True) -> List[Path]:
        """Find all supported image files in directory."""
        image_files = []
        supported_extensions = {f".{fmt}" for fmt in self.config.processing.supported_formats}
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Check file size
                    file_size = file_path.stat().st_size
                    if file_size > 20 * 1024 * 1024:  # 20MB limit
                        self.logger.warning(f"Skipping large file: {file_path} ({file_size/1024/1024:.1f}MB)")
                        continue
                    
                    image_files.append(file_path)
            
            # Sort for consistent processing order
            image_files.sort()
            
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {str(e)}")
        
        return image_files
    
    def _log_batch_results(self) -> None:
        """Log batch processing results."""
        total = self.stats['total_images']
        processed = self.stats['processed_images']
        failed = self.stats['failed_images']
        skipped = self.stats['skipped_images']
        
        self.logger.info("=" * 50)
        self.logger.info("BATCH PROCESSING RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total images found: {total}")
        self.logger.info(f"Successfully processed: {processed}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Skipped (already exist): {skipped}")
        
        if total > 0:
            success_rate = (processed / total) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        self.logger.info("=" * 50)
    
    def process_file_list(
        self,
        file_list: List[str],
        output_dir: str,
        quality: str = 'balanced',
        upscale: int = 4,
        face_enhance: bool = True,
        output_format: str = 'jpg'
    ) -> int:
        """
        Process a specific list of image files.
        
        Args:
            file_list: List of image file paths
            output_dir: Output directory path
            quality: Quality level
            upscale: Upscale factor
            face_enhance: Enable face enhancement
            output_format: Output format
            
        Returns:
            Number of successfully processed images
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filter valid files
        valid_files = []
        for file_path in file_list:
            path = Path(file_path)
            if path.exists() and path.is_file():
                file_ext = path.suffix.lower().lstrip('.')
                if file_ext in self.config.processing.supported_formats:
                    valid_files.append(path)
                else:
                    self.logger.warning(f"Unsupported format: {file_path}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        if not valid_files:
            self.logger.error("No valid image files to process")
            return 0
        
        self.stats['total_images'] = len(valid_files)
        success_count = 0
        
        # Process files
        with tqdm(total=len(valid_files), desc="Processing images", unit="img") as pbar:
            for image_file in valid_files:
                try:
                    # Generate output filename
                    output_file = output_path / f"{image_file.stem}_enhanced.{output_format}"
                    
                    # Process image
                    success = self.image_processor.process_image(
                        input_path=str(image_file),
                        output_path=str(output_file),
                        quality=quality,
                        upscale=upscale,
                        face_enhance=face_enhance,
                        output_format=output_format
                    )
                    
                    if success:
                        success_count += 1
                        self.stats['processed_images'] += 1
                    else:
                        self.stats['failed_images'] += 1
                    
                except Exception as e:
                    self.stats['failed_images'] += 1
                    self.logger.error(f"Error processing {image_file}: {str(e)}")
                
                finally:
                    pbar.update(1)
        
        self._log_batch_results()
        return success_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'skipped_images': 0,
            'processing_time': 0.0
        }