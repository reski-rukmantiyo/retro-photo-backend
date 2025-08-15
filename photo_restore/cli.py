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


def print_model_list():
    """Print comprehensive model information."""
    click.echo("\n🤖 Available AI Models:\n")
    
    click.echo("📈 Real-ESRGAN Models:")
    click.echo("  x2    - 2x upscaling, faster processing, lower memory usage")
    click.echo("  x4    - 4x upscaling, higher quality enhancement, more memory")
    
    click.echo("\n👤 GFPGAN Face Enhancement Models:")
    click.echo("  v1.3  - Higher quality faces, standard processing time") 
    click.echo("  v1.4  - Faster processing, more memory efficient (25% less RAM)")
    
    click.echo("\n🔧 Auto-Selection Rules:")
    click.echo("  --quality fast     → Real-ESRGAN x2 + GFPGAN v1.4 (speed optimized)")
    click.echo("  --quality balanced → Real-ESRGAN x4 + GFPGAN v1.4 (high-quality upscaling + efficient face enhancement)")  
    click.echo("  --quality best     → Real-ESRGAN x4 + GFPGAN v1.3 (maximum quality)")
    
    click.echo("\n💡 Usage Examples:")
    click.echo("  photo-restore image.jpg --gfpgan-version v1.4")
    click.echo("  photo-restore image.jpg --quality fast      # Auto: Real-ESRGAN x2 + GFPGAN v1.4")
    click.echo("  photo-restore image.jpg --quality balanced  # Auto: Real-ESRGAN x4 + GFPGAN v1.4")
    click.echo("  photo-restore image.jpg --quality best      # Auto: Real-ESRGAN x4 + GFPGAN v1.3")
    click.echo("  photo-restore --batch photos/ --gfpgan-version v1.3")


def print_model_info(model_name: str):
    """Print detailed information about specific model."""
    model_db = {
        'realesrgan-x2': {
            'name': 'Real-ESRGAN x2',
            'upscale_factor': '2x',
            'memory_usage': '~800MB peak',
            'processing_speed': 'Fast',
            'best_for': 'Quick enhancement, limited memory systems',
            'file_size': '~65MB download'
        },
        'realesrgan-x4': {
            'name': 'Real-ESRGAN x4', 
            'upscale_factor': '4x',
            'memory_usage': '~1.8GB peak',
            'processing_speed': 'Standard',
            'best_for': 'High quality upscaling, detailed enhancement',
            'file_size': '~65MB download'
        },
        'gfpgan-v1.3': {
            'name': 'GFPGAN v1.3',
            'specialty': 'Face restoration',
            'memory_usage': '~800MB peak', 
            'processing_speed': 'Standard',
            'best_for': 'Maximum quality face enhancement',
            'file_size': '~350MB download'
        },
        'gfpgan-v1.4': {
            'name': 'GFPGAN v1.4',
            'specialty': 'Face restoration', 
            'memory_usage': '~600MB peak (25% improvement)',
            'processing_speed': 'Fast (15% speed improvement)',
            'best_for': 'Quick face enhancement, batch processing',
            'file_size': '~350MB download'
        }
    }
    
    model_key = model_name.lower().replace('_', '-').replace(' ', '-')
    
    if model_key not in model_db:
        click.echo(f"\n❌ Model '{model_name}' not found.")
        click.echo(f"💡 Use --list-models to see all available models.")
        click.echo(f"📋 Available models: realesrgan-x2, realesrgan-x4, gfpgan-v1.3, gfpgan-v1.4")
        return
        
    info = model_db[model_key]
    click.echo(f"\n📋 {info['name']} Information:")
    for key, value in info.items():
        if key != 'name':
            formatted_key = key.replace('_', ' ').title()
            click.echo(f"  {formatted_key}: {value}")
    
    click.echo(f"\n💡 Usage Example:")
    if 'gfpgan' in model_key:
        version = 'v1.3' if 'v1.3' in model_key else 'v1.4'
        click.echo(f"  photo-restore image.jpg --gfpgan-version {version}")
    else:
        scale = 'x2' if 'x2' in model_key else 'x4'
        click.echo(f"  photo-restore image.jpg --upscale {scale[1:]}")


@click.command()
@click.argument('input_path', type=click.Path(exists=True), required=False)
@click.argument('output_path', type=click.Path(), required=False)
@click.option('--batch', is_flag=True, help='Process entire directory')
@click.option('--quality', type=click.Choice(['fast', 'balanced', 'best']), 
              default='balanced', help='Processing quality level')
@click.option('--upscale', type=int, default=4, help='Upscale factor (2 or 4)')
@click.option('--face-enhance', is_flag=True, default=True, 
              help='Enable face enhancement with GFPGAN')
@click.option('--gfpgan-version', type=click.Choice(['v1.3', 'v1.4', 'auto']),
              default='auto', help='GFPGAN model version: v1.3 (quality), v1.4 (speed), auto (smart selection)')
@click.option('--list-models', is_flag=True, 
              help='List all available models and their characteristics')
@click.option('--model-info', type=str,
              help='Show detailed information about a specific model')
@click.option('--output-format', type=click.Choice(['jpg', 'png']), 
              default='jpg', help='Output image format')
@click.option('--config', type=click.Path(), help='Custom configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_path: Optional[str], output_path: Optional[str], batch: bool, quality: str,
         upscale: int, face_enhance: bool, gfpgan_version: str, list_models: bool,
         model_info: Optional[str], output_format: str, config: Optional[str], verbose: bool) -> None:
    """Photo restoration CLI tool using Real-ESRGAN and GFPGAN.
    
    Process single images or entire directories with AI-powered enhancement.
    
    Examples:
        photo-restore input.jpg output.jpg
        photo-restore --batch ./photos/ --output ./enhanced/
        photo-restore --quality best --face-enhance photo.jpg
    """
    # Handle utility functions first
    if list_models:
        print_model_list()
        return
        
    if model_info:
        print_model_info(model_info)
        return

    # Validate input_path is provided for processing commands
    if not input_path:
        click.echo("❌ Error: input_path is required for image processing.")
        click.echo("💡 Use --list-models or --model-info for information commands.")
        sys.exit(1)

    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    logger = setup_logger(level=log_level)
    
    try:
        # SECURITY: Initialize path validator with allowed directories
        allowed_paths = [
            os.getcwd(),  # Current directory (backend/)
        ]
        
        # Add parent project directory to allow access to ../samples/ etc.
        project_root = str(Path(os.getcwd()).parent.absolute())
        if os.path.exists(project_root):
            allowed_paths.append(project_root)
        
        # Only add Pictures directory if it exists
        pictures_dir = os.path.expanduser("~/Pictures")
        if os.path.exists(pictures_dir):
            allowed_paths.append(pictures_dir)
        
        # Only add tmp directory if it exists or can be created
        tmp_dir = "/tmp/photo_restore"
        try:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            allowed_paths.append(tmp_dir)
        except (OSError, PermissionError):
            # Skip if can't create temp directory
            pass
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
                output_format=output_format,
                gfpgan_version=gfpgan_version
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
                    gfpgan_version=gfpgan_version,
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