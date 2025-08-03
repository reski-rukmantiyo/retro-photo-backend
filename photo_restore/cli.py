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
        # Load configuration
        cfg = Config.load(config_path=config)
        
        # Validate upscale factor
        if upscale not in [2, 4]:
            raise click.BadParameter("Upscale factor must be 2 or 4")
        
        # Process batch or single image
        if batch:
            processor = BatchProcessor(config=cfg, logger=logger)
            success_count = processor.process_directory(
                input_dir=input_path,
                output_dir=output_path or f"{input_path}_enhanced",
                quality=quality,
                upscale=upscale,
                face_enhance=face_enhance,
                output_format=output_format
            )
            click.echo(f"✅ Successfully processed {success_count} images")
            
        else:
            # Single image processing
            if not output_path:
                input_file = Path(input_path)
                output_path = str(input_file.parent / f"{input_file.stem}_enhanced{input_file.suffix}")
            
            processor = ImageProcessor(config=cfg, logger=logger)
            
            with tqdm(total=100, desc="Processing", unit="%") as pbar:
                def progress_callback(percent: int):
                    pbar.update(percent - pbar.n)
                
                result = processor.process_image(
                    input_path=input_path,
                    output_path=output_path,
                    quality=quality,
                    upscale=upscale,
                    face_enhance=face_enhance,
                    output_format=output_format,
                    progress_callback=progress_callback
                )
            
            if result:
                click.echo(f"✅ Enhanced image saved to: {output_path}")
            else:
                click.echo("❌ Processing failed")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()