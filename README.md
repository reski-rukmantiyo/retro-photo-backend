# Photo Restore CLI

AI-powered photo restoration using Real-ESRGAN and GFPGAN, optimized for CPU processing.

## Features

- **CPU Optimized**: No GPU required, runs on any computer
- **Batch Processing**: Process single images or entire directories
- **Face Enhancement**: Automatic face detection and restoration with GFPGAN
- **Quality Control**: Fast, balanced, and best quality modes
- **Simple CLI**: One-command photo restoration

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Process single image
photo-restore input.jpg output.jpg

# Batch process directory
photo-restore --batch ./old_photos/ --output ./enhanced/

# High quality processing with face enhancement
photo-restore --quality best --face-enhance photo.jpg
```

## Commands

### Basic Usage
```bash
photo-restore INPUT_PATH [OUTPUT_PATH]
```

### Options
- `--batch`: Process entire directory
- `--quality`: Quality level (fast/balanced/best)
- `--upscale`: Upscale factor (2 or 4) 
- `--face-enhance`: Enable face enhancement (default: true)
- `--output-format`: Output format (jpg/png)
- `--verbose`: Verbose logging

## Examples

```bash
# Single image with custom output
photo-restore old_photo.jpg enhanced_photo.jpg

# Batch processing with high quality
photo-restore --batch --quality best ./photos/ ./enhanced/

# Fast processing without face enhancement
photo-restore --quality fast --no-face-enhance image.jpg
```

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- 5GB free disk space for models

## License

Apache 2.0