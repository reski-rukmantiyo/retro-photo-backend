"""Setup configuration for photo-restore CLI tool."""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="photo-restore",
    version="0.1.0",
    author="Photo Restore Team",
    author_email="team@photo-restore.com",
    description="AI-powered photo restoration CLI tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/photo-restore",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "photo-restore=photo_restore.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "photo_restore": ["config/*.yaml"],
    },
    keywords="photo restoration, image enhancement, AI, CLI, photo repair",
    project_urls={
        "Bug Reports": "https://github.com/username/photo-restore/issues",
        "Source": "https://github.com/username/photo-restore",
        "Documentation": "https://github.com/username/photo-restore#readme",
    },
)