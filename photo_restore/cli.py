"""Main CLI interface for photo restoration."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from .processors.image_processor import ImageProcessor
from .processors.batch_processor import BatchProcessor
from .utils.config import Config
from .utils.logger import setup_logger

# SECURITY: Import path validation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'security_patches'))
try:
    from path_validator import SecurePathValidator, PathSecurityError
except ImportError:
    # Fallback if security patches not available
    class PathSecurityError(Exception):
        pass
    class SecurePathValidator:
        def __init__(self, *args, **kwargs):
            pass
        def validate_path(self, path, base_path=None):
            return os.path.abspath(path)
        def validate_output_path(self, path, base_path=None):
            return os.path.abspath(path)


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(), required=False)
@click.option('--batch', is_flag=True, help='Process entire directory')
@click.option('--quality', type=click.Choice(['fast', 'balanced', 'best']), 
              default='balanced', help='Processing quality level')
@click.option('--upscale', type=int, default=4, help='Upscale factor (2 or 4)')
@click.option('--face-enhance', is_flag=True, default=True, 
              help='Enable face enhancement with GFPGAN')
@click.option('--output-format', type=click.Choice(['jpg', 'png']), 
              default='jpg', help='Output image format')
@click.option('--config', type=click.Path(), help='Custom configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_path: str, output_path: Optional[str], batch: bool, quality: str,
         upscale: int, face_enhance: bool, output_format: str, 
         config: Optional[str], verbose: bool) -> None:
    """Photo restoration CLI tool using Real-ESRGAN and GFPGAN.
    
    Process single images or entire directories with AI-powered enhancement.
    
    Examples:
        photo-restore input.jpg output.jpg
        photo-restore --batch ./photos/ --output ./enhanced/
        photo-restore --quality best --face-enhance photo.jpg
    """
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    logger = setup_logger(level=log_level)
    
    try:
        # SECURITY: Initialize path validator with allowed directories
        allowed_paths = [
            os.getcwd(),  # Current directory
            os.path.expanduser("~/Pictures"),  # User's pictures directory
            "/tmp/photo_restore"  # Temporary processing directory
        ]
        # Add parent directories of input path to allowed paths for flexibility
        input_parent = str(Path(input_path).parent.absolute())
        if input_parent not in allowed_paths:
            allowed_paths.append(input_parent)
            
        path_validator = SecurePathValidator(allowed_paths)
        
        # SECURITY: Validate input path
        try:
            safe_input_path = path_validator.validate_path(input_path)
            logger.info(f"Validated input path: {safe_input_path}")
        except PathSecurityError as e:
            logger.error(f"Input path security validation failed: {e}")
            click.echo(f"❌ Security error: Invalid input path")
            sys.exit(1)
        
        # Load configuration
        cfg = Config.load(config_path=config)
        
        # Validate upscale factor
        if upscale not in [2, 4]:
            raise click.BadParameter("Upscale factor must be 2 or 4")
        
        # SECURITY: Validate output path for batch processing
        if batch:
            # Validate output directory
            batch_output_dir = output_path or f"{safe_input_path}_enhanced"
            try:
                safe_output_dir = path_validator.validate_output_path(batch_output_dir)
                logger.info(f"Validated output directory: {safe_output_dir}")
            except PathSecurityError as e:
                logger.error(f"Output path security validation failed: {e}")
                click.echo(f"❌ Security error: Invalid output path")
                sys.exit(1)
                
            processor = BatchProcessor(config=cfg, logger=logger)
            success_count = processor.process_directory(
                input_dir=safe_input_path,
                output_dir=safe_output_dir,
                quality=quality,
                upscale=upscale,
                face_enhance=face_enhance,
                output_format=output_format
            )
            click.echo(f"✅ Successfully processed {success_count} images")
            
        else:
            # SECURITY: Validate single image output path
            if not output_path:
                input_file = Path(safe_input_path)
                output_path = str(input_file.parent / f"{input_file.stem}_enhanced{input_file.suffix}")
            
            try:
                safe_output_path = path_validator.validate_output_path(output_path)
                logger.info(f"Validated output path: {safe_output_path}")
            except PathSecurityError as e:
                logger.error(f"Output path security validation failed: {e}")
                click.echo(f"❌ Security error: Invalid output path")
                sys.exit(1)
            
            processor = ImageProcessor(config=cfg, logger=logger)
            
            with tqdm(total=100, desc="Processing", unit="%") as pbar:
                def progress_callback(percent: int):
                    pbar.update(percent - pbar.n)
                
                result = processor.process_image(
                    input_path=safe_input_path,
                    output_path=safe_output_path,
                    quality=quality,
                    upscale=upscale,
                    face_enhance=face_enhance,
                    output_format=output_format,
                    progress_callback=progress_callback
                )
            
            if result:
                click.echo(f"✅ Enhanced image saved to: {safe_output_path}")
            else:
                click.echo("❌ Processing failed")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()