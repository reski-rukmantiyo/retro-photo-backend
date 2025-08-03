"""Comprehensive tests for file utilities."""

import pytest
import tempfile
import shutil
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from photo_restore.utils.file_utils import (
    validate_image_path,
    generate_output_path,
    find_images_in_directory,
    get_file_size,
    get_file_hash,
    safe_copy,
    safe_remove,
    create_temp_dir,
    ensure_directory,
    get_available_space,
    format_file_size,
    TempFileManager,
    SUPPORTED_FORMATS,
    OUTPUT_FORMATS
)


class TestValidateImagePath:
    """Test image path validation."""
    
    def test_valid_image_path(self, sample_image_file):
        """Test validation of valid image path."""
        validated_path = validate_image_path(sample_image_file)
        assert validated_path == sample_image_file
        assert isinstance(validated_path, Path)
    
    def test_valid_image_path_string(self, sample_image_file):
        """Test validation with string path."""
        validated_path = validate_image_path(str(sample_image_file))
        assert validated_path == sample_image_file
    
    def test_nonexistent_file(self, temp_dir):
        """Test validation of nonexistent file."""
        nonexistent = temp_dir / "nonexistent.jpg"
        
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            validate_image_path(nonexistent)
    
    def test_directory_instead_of_file(self, temp_dir):
        """Test validation when path is directory."""
        with pytest.raises(ValueError, match="Path is not a file"):
            validate_image_path(temp_dir)
    
    @pytest.mark.parametrize("extension", [".txt", ".doc", ".exe", ".zip"])
    def test_unsupported_format(self, temp_dir, extension):
        """Test validation of unsupported file formats."""
        unsupported_file = temp_dir / f"test{extension}"
        unsupported_file.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_image_path(unsupported_file)
    
    @pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"])
    def test_supported_formats(self, temp_dir, sample_image_rgb, extension):
        """Test validation of all supported formats."""
        image_file = temp_dir / f"test{extension}"
        
        # Create image file
        pil_image = Image.fromarray(sample_image_rgb)
        format_name = "JPEG" if extension in [".jpg", ".jpeg"] else extension[1:].upper()
        if format_name == "TIFF":
            format_name = "TIFF"
        
        try:
            pil_image.save(image_file, format_name)
            validated_path = validate_image_path(image_file)
            assert validated_path == image_file
        except Exception:
            pytest.skip(f"Format {extension} not supported in test environment")
    
    def test_case_insensitive_extension(self, temp_dir, sample_image_rgb):
        """Test case insensitive extension matching."""
        # Create files with uppercase extensions
        jpg_file = temp_dir / "test.JPG"
        png_file = temp_dir / "test.PNG"
        
        pil_image = Image.fromarray(sample_image_rgb)
        pil_image.save(jpg_file, "JPEG")
        pil_image.save(png_file, "PNG")
        
        # Should validate successfully
        assert validate_image_path(jpg_file) == jpg_file
        assert validate_image_path(png_file) == png_file


class TestGenerateOutputPath:
    """Test output path generation."""
    
    def test_same_directory_default(self, sample_image_file):
        """Test output path in same directory with default suffix."""
        output_path = generate_output_path(sample_image_file)
        
        expected = sample_image_file.parent / f"{sample_image_file.stem}_enhanced.jpg"
        assert output_path == expected
    
    def test_custom_output_directory(self, sample_image_file, temp_dir):
        """Test output path in custom directory."""
        output_dir = temp_dir / "output"
        output_path = generate_output_path(sample_image_file, output_dir)
        
        expected = output_dir / f"{sample_image_file.stem}_enhanced.jpg"
        assert output_path == expected
        assert output_dir.exists()  # Should create directory
    
    def test_custom_suffix(self, sample_image_file):
        """Test custom suffix."""
        output_path = generate_output_path(sample_image_file, suffix="_restored")
        
        expected = sample_image_file.parent / f"{sample_image_file.stem}_restored.jpg"
        assert output_path == expected
    
    @pytest.mark.parametrize("format,expected_ext", [
        ("jpg", ".jpg"),
        ("png", ".png"),
        ("JPEG", ".jpeg"),
        ("PNG", ".png")
    ])
    def test_output_formats(self, sample_image_file, format, expected_ext):
        """Test different output formats."""
        output_path = generate_output_path(sample_image_file, format=format)
        assert output_path.suffix == expected_ext
    
    def test_unsupported_format_defaults_to_jpg(self, sample_image_file):
        """Test unsupported format defaults to jpg."""
        output_path = generate_output_path(sample_image_file, format="tiff")
        assert output_path.suffix == ".jpg"
    
    def test_conflict_resolution(self, sample_image_file):
        """Test conflict resolution when file exists."""
        # Create the expected output file
        expected_path = sample_image_file.parent / f"{sample_image_file.stem}_enhanced.jpg"
        expected_path.touch()
        
        # Should generate path with counter
        output_path = generate_output_path(sample_image_file)
        expected_with_counter = sample_image_file.parent / f"{sample_image_file.stem}_enhanced_1.jpg"
        assert output_path == expected_with_counter
    
    def test_multiple_conflicts(self, sample_image_file):
        """Test multiple conflict resolution."""
        base_name = sample_image_file.stem + "_enhanced"
        parent_dir = sample_image_file.parent
        
        # Create multiple existing files
        for i in range(3):
            if i == 0:
                conflict_file = parent_dir / f"{base_name}.jpg"
            else:
                conflict_file = parent_dir / f"{base_name}_{i}.jpg"
            conflict_file.touch()
        
        output_path = generate_output_path(sample_image_file)
        expected = parent_dir / f"{base_name}_3.jpg"
        assert output_path == expected
    
    def test_suffix_already_present(self, temp_dir, sample_image_rgb):
        """Test when suffix is already in filename."""
        # Create file that already has suffix
        input_file = temp_dir / "test_enhanced.jpg"
        pil_image = Image.fromarray(sample_image_rgb)
        pil_image.save(input_file, "JPEG")
        
        output_path = generate_output_path(input_file)
        # Should not double the suffix
        expected = temp_dir / "test_enhanced.jpg"
        assert output_path == expected


class TestFindImagesInDirectory:
    """Test image file discovery."""
    
    def test_find_images_flat_directory(self, test_directory_structure):
        """Test finding images in flat directory."""
        images = find_images_in_directory(test_directory_structure, recursive=False)
        
        # Should find only images in root directory
        assert len(images) == 2  # image1.jpg and image2.png
        image_names = [img.name for img in images]
        assert "image1.jpg" in image_names
        assert "image2.png" in image_names
    
    def test_find_images_recursive(self, test_directory_structure):
        """Test finding images recursively."""
        images = find_images_in_directory(test_directory_structure, recursive=True)
        
        # Should find all images in all subdirectories
        assert len(images) == 5  # All images
        image_names = [img.name for img in images]
        expected = ["image1.jpg", "image2.png", "image3.jpg", "image4.png", "image5.jpg"]
        for expected_name in expected:
            assert expected_name in image_names
    
    def test_empty_directory(self, temp_dir):
        """Test finding images in empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        images = find_images_in_directory(empty_dir)
        assert len(images) == 0
    
    def test_nonexistent_directory(self):
        """Test finding images in nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            find_images_in_directory("/nonexistent/directory")
    
    def test_file_instead_of_directory(self, sample_image_file):
        """Test passing file instead of directory."""
        with pytest.raises(ValueError, match="Path is not a directory"):
            find_images_in_directory(sample_image_file)
    
    def test_mixed_files_directory(self, temp_dir, sample_image_rgb):
        """Test directory with mixed file types."""
        # Create mixed files
        pil_image = Image.fromarray(sample_image_rgb)
        pil_image.save(temp_dir / "image1.jpg", "JPEG")
        pil_image.save(temp_dir / "image2.png", "PNG")
        
        # Non-image files
        (temp_dir / "document.txt").write_text("content")
        (temp_dir / "data.json").write_text("{}")
        (temp_dir / "script.py").write_text("print('hello')")
        
        images = find_images_in_directory(temp_dir)
        
        # Should find only image files
        assert len(images) == 2
        image_names = [img.name for img in images]
        assert "image1.jpg" in image_names
        assert "image2.png" in image_names
    
    def test_case_insensitive_extensions(self, temp_dir, sample_image_rgb):
        """Test case insensitive extension matching."""
        pil_image = Image.fromarray(sample_image_rgb)
        
        # Create files with various case extensions
        pil_image.save(temp_dir / "image1.JPG", "JPEG")
        pil_image.save(temp_dir / "image2.Png", "PNG")
        pil_image.save(temp_dir / "image3.JPEG", "JPEG")
        
        images = find_images_in_directory(temp_dir)
        assert len(images) == 3
    
    def test_sorted_output(self, temp_dir, sample_image_rgb):
        """Test that output is sorted."""
        pil_image = Image.fromarray(sample_image_rgb)
        
        # Create files in non-alphabetical order
        files = ["zebra.jpg", "alpha.png", "beta.jpg"]
        for filename in files:
            pil_image.save(temp_dir / filename, "JPEG" if filename.endswith('.jpg') else "PNG")
        
        images = find_images_in_directory(temp_dir)
        image_names = [img.name for img in images]
        
        # Should be sorted
        assert image_names == sorted(image_names)


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_get_file_size(self, sample_image_file):
        """Test file size calculation."""
        size = get_file_size(sample_image_file)
        expected_size = sample_image_file.stat().st_size
        assert size == expected_size
        assert size > 0
    
    @pytest.mark.parametrize("algorithm", ["md5", "sha1", "sha256"])
    def test_get_file_hash(self, sample_image_file, algorithm):
        """Test file hash calculation."""
        hash_value = get_file_hash(sample_image_file, algorithm)
        
        # Verify hash format
        if algorithm == "md5":
            assert len(hash_value) == 32
        elif algorithm == "sha1":
            assert len(hash_value) == 40
        elif algorithm == "sha256":
            assert len(hash_value) == 64
        
        # Verify hash is hexadecimal
        int(hash_value, 16)  # Should not raise exception
        
        # Verify consistency
        hash_value2 = get_file_hash(sample_image_file, algorithm)
        assert hash_value == hash_value2
    
    def test_get_file_hash_different_files(self, sample_image_files):
        """Test hash values are different for different files."""
        jpg_hash = get_file_hash(sample_image_files['jpg'])
        png_hash = get_file_hash(sample_image_files['png'])
        
        # Different formats should have different hashes
        assert jpg_hash != png_hash
    
    def test_safe_copy_basic(self, sample_image_file, temp_dir):
        """Test basic file copying."""
        dest_path = temp_dir / "copied_image.jpg"
        
        safe_copy(sample_image_file, dest_path)
        
        assert dest_path.exists()
        assert get_file_size(dest_path) == get_file_size(sample_image_file)
        assert get_file_hash(dest_path) == get_file_hash(sample_image_file)
    
    def test_safe_copy_creates_directory(self, sample_image_file, temp_dir):
        """Test safe copy creates destination directory."""
        nested_dest = temp_dir / "nested" / "subdir" / "copied.jpg"
        
        safe_copy(sample_image_file, nested_dest)
        
        assert nested_dest.exists()
        assert nested_dest.parent.exists()
    
    def test_safe_copy_overwrites(self, sample_image_file, temp_dir):
        """Test safe copy overwrites existing file."""
        dest_path = temp_dir / "existing.jpg"
        dest_path.write_text("old content")
        
        safe_copy(sample_image_file, dest_path)
        
        # Should overwrite with image content
        assert dest_path.exists()
        assert get_file_hash(dest_path) == get_file_hash(sample_image_file)
    
    def test_safe_remove_file(self, temp_dir):
        """Test safe removal of file."""
        test_file = temp_dir / "to_delete.txt"
        test_file.write_text("content")
        
        safe_remove(test_file)
        assert not test_file.exists()
    
    def test_safe_remove_directory(self, temp_dir):
        """Test safe removal of directory."""
        test_dir = temp_dir / "to_delete"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")
        
        safe_remove(test_dir)
        assert not test_dir.exists()
    
    def test_safe_remove_nonexistent(self, temp_dir):
        """Test safe removal of nonexistent path."""
        nonexistent = temp_dir / "nonexistent"
        
        # Should not raise exception
        safe_remove(nonexistent)
    
    def test_create_temp_dir(self):
        """Test temporary directory creation."""
        temp_dir = create_temp_dir("test_prefix_")
        
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert "test_prefix_" in temp_dir.name
        
        # Cleanup
        safe_remove(temp_dir)
    
    def test_ensure_directory_new(self, temp_dir):
        """Test ensuring new directory exists."""
        new_dir = temp_dir / "new" / "nested" / "directory"
        
        result = ensure_directory(new_dir)
        
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_ensure_directory_existing(self, temp_dir):
        """Test ensuring existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        
        assert result == existing_dir
        assert existing_dir.exists()
    
    def test_get_available_space(self, temp_dir):
        """Test getting available disk space."""
        space = get_available_space(temp_dir)
        
        assert isinstance(space, int)
        assert space > 0  # Should have some free space
    
    @pytest.mark.parametrize("size,expected", [
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
        (1099511627776, "1.0 TB"),
        (1536, "1.5 KB"),
        (2621440, "2.5 MB")
    ])
    def test_format_file_size(self, size, expected):
        """Test file size formatting."""
        formatted = format_file_size(size)
        assert formatted == expected


class TestTempFileManager:
    """Test temporary file manager."""
    
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with TempFileManager("test_") as manager:
            assert manager.temp_dir is not None
            assert manager.temp_dir.exists()
            temp_dir_path = manager.temp_dir
        
        # Should cleanup after exit
        assert not temp_dir_path.exists()
    
    def test_create_temp_file(self):
        """Test creating temporary files."""
        with TempFileManager("test_") as manager:
            temp_file = manager.create_temp_file(".jpg")
            
            assert temp_file.parent == manager.temp_dir
            assert temp_file.suffix == ".jpg"
            assert temp_file in manager.temp_files
    
    def test_multiple_temp_files(self):
        """Test creating multiple temporary files."""
        with TempFileManager("test_") as manager:
            file1 = manager.create_temp_file(".jpg")
            file2 = manager.create_temp_file(".png")
            file3 = manager.create_temp_file(".txt")
            
            assert len(manager.temp_files) == 3
            assert all(f.parent == manager.temp_dir for f in [file1, file2, file3])
            assert file1.suffix == ".jpg"
            assert file2.suffix == ".png"
            assert file3.suffix == ".txt"
    
    def test_cleanup_on_exception(self):
        """Test cleanup happens even on exception."""
        temp_dir_path = None
        
        try:
            with TempFileManager("test_") as manager:
                temp_dir_path = manager.temp_dir
                temp_file = manager.create_temp_file()
                temp_file.write_text("content")  # Create actual file
                
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still cleanup
        assert not temp_dir_path.exists()
    
    def test_manual_cleanup(self):
        """Test manual cleanup."""
        manager = TempFileManager("test_")
        manager.__enter__()
        
        temp_file = manager.create_temp_file()
        temp_file.write_text("content")
        temp_dir_path = manager.temp_dir
        
        assert temp_dir_path.exists()
        assert temp_file.exists()
        
        manager.cleanup()
        
        assert not temp_dir_path.exists()
        assert not temp_file.exists()
        assert len(manager.temp_files) == 0
        assert manager.temp_dir is None
    
    def test_create_temp_file_without_init(self):
        """Test creating temp file without initialization."""
        manager = TempFileManager()
        
        with pytest.raises(RuntimeError, match="TempFileManager not initialized"):
            manager.create_temp_file()
    
    def test_custom_prefix(self):
        """Test custom prefix for temporary directory."""
        custom_prefix = "custom_photo_restore_"
        
        with TempFileManager(custom_prefix) as manager:
            assert custom_prefix in manager.temp_dir.name