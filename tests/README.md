# Photo Restoration CLI - Test Suite

This directory contains a comprehensive test suite for the Photo Restoration CLI project, designed to ensure production readiness with 95%+ code coverage and robust error handling.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and shared fixtures
├── pytest.ini                    # Pytest settings and markers
├── run_tests.py                   # Comprehensive test runner script
├── fixtures/                     # Test data and utilities
│   └── test_images.py            # Test image generation utilities
├── mocks/                        # Mock implementations
│   ├── __init__.py
│   └── ai_models.py              # Mock AI models for testing without downloads
└── test_*.py                     # Test modules
```

## Test Categories

### Unit Tests
- **test_config_comprehensive.py** - Configuration management tests
- **test_file_utils.py** - File system operations and utilities
- **test_logger.py** - Logging functionality tests

### Integration Tests  
- **test_model_manager.py** - AI model management with mocked models
- **test_image_processor.py** - Image processing pipeline tests
- **test_batch_processor.py** - Batch processing functionality
- **test_cli.py** - CLI interface integration tests

### Performance Tests
- **test_performance.py** - Performance benchmarks and CPU processing tests

### Error Handling Tests
- **test_error_handling.py** - Comprehensive error handling and edge cases

### Coverage Tests
- **test_coverage.py** - Coverage validation and reporting

## Running Tests

### Quick Start
```bash
# Run quick tests (recommended for development)
python run_tests.py --quick

# Run all tests with coverage
python run_tests.py --coverage-only

# Run specific test category
python run_tests.py --category unit
```

### Test Runner Options
```bash
# Run all test categories
python run_tests.py --all

# Run smoke tests (basic functionality)
python run_tests.py --smoke

# Run with verbose output
python run_tests.py --all --verbose

# Stop on first failure
python run_tests.py --all --fail-fast

# Skip slow tests (like performance benchmarks)
python run_tests.py --all --skip-slow

# Code quality checks
python run_tests.py --lint --format
```

### Direct Pytest Usage
```bash
# Run specific test file
pytest tests/test_config_comprehensive.py -v

# Run with coverage
pytest --cov=photo_restore --cov-report=html

# Run performance tests only
pytest -m performance

# Run excluding slow tests
pytest -m "not slow and not performance"
```

## Test Markers

- `unit` - Unit tests
- `integration` - Integration tests  
- `performance` - Performance benchmarks
- `slow` - Tests that take a long time
- `requires_models` - Tests requiring AI model downloads

## Mock Strategy

The test suite uses comprehensive mocking to avoid dependencies on:

- **AI Model Downloads** - Mock implementations of Real-ESRGAN and GFPGAN
- **Network Requests** - Mocked HTTP responses for model downloads
- **File System** - Controlled temporary directories and mock file operations
- **External Dependencies** - Mocked where possible to ensure deterministic tests

## Test Fixtures

### Common Fixtures (conftest.py)
- `temp_dir` - Temporary directory for test files
- `sample_config` - Test configuration instance
- `test_logger` - Test logger instance
- `sample_image_*` - Various test image formats and data
- `mock_model_manager` - Mocked model manager with AI models
- `test_directory_structure` - Directory tree with test images

### Test Image Generation
The `fixtures/test_images.py` module provides utilities for generating:
- Images with various dimensions and patterns
- Corrupted and invalid image files
- Images with different color spaces
- Performance test datasets

## Coverage Requirements

- **Overall Coverage**: 95%+ required
- **Individual Files**: 85-95% depending on complexity
- **Branch Coverage**: Tracked for complex logic paths
- **Error Paths**: All error conditions must be tested

### Coverage Reports
```bash
# Generate HTML coverage report
python run_tests.py --coverage-only
# View: htmlcov/index.html

# Generate XML coverage report (for CI)
pytest --cov=photo_restore --cov-report=xml
```

## Performance Testing

Performance tests validate:
- **Memory Efficiency** - Processing within memory constraints
- **Processing Speed** - Baseline performance benchmarks
- **Scalability** - Performance with increasing batch sizes
- **Tiling Efficiency** - Memory usage with large images

### Performance Thresholds
- Small images (256x256): < 5 seconds
- Medium images (1024x1024): < 15 seconds  
- Large images (2048x2048): < 30 seconds
- Memory increase: < 800MB for large images

## Error Handling Tests

Comprehensive error scenario coverage:
- **Invalid inputs** - Corrupted files, wrong formats
- **System errors** - Permission denied, disk full, memory errors
- **Network errors** - Download failures, timeouts
- **Edge cases** - Extreme dimensions, Unicode paths, concurrent access

## CI/CD Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions step
- name: Run Tests
  run: |
    python run_tests.py --all --skip-slow
    
- name: Generate Coverage
  run: |
    python run_tests.py --coverage-only
    
- name: Upload Coverage
  uses: codecov/codecov-action@v2
  with:
    file: ./coverage.xml
```

## Development Guidelines

### Adding New Tests
1. Place tests in appropriate category file
2. Use descriptive test names with `test_` prefix
3. Include docstrings explaining test purpose
4. Use appropriate fixtures and mocks
5. Test both success and failure paths

### Test Naming Convention
```python
def test_feature_success_case(self):
    """Test successful feature operation."""
    
def test_feature_error_handling(self):
    """Test feature handles errors gracefully."""
    
def test_feature_edge_case_large_input(self):  
    """Test feature with edge case input."""
```

### Mock Usage
```python
# Mock external dependencies
with patch('photo_restore.models.model_manager.requests.get') as mock_get:
    mock_get.return_value.status_code = 200
    # Test code here

# Use provided mock fixtures
def test_with_mock_models(self, mock_model_manager):
    processor = ImageProcessor(config, logger)
    processor.model_manager = mock_model_manager
    # Test code here
```

## Troubleshooting

### Common Issues

**Tests fail with import errors**
```bash
# Ensure project is in Python path
export PYTHONPATH=/home/reski/Github/retro-photo-2/backend:$PYTHONPATH
```

**Coverage reports not generated**
```bash
# Install coverage dependencies
pip install pytest-cov coverage
```

**Performance tests timeout**
```bash
# Skip performance tests during development
python run_tests.py --all --skip-slow
```

**Mock AI models not working**
```bash
# Ensure mock modules are imported
pytest tests/test_model_manager.py -v
```

### Debug Mode
```bash
# Run with debug output
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb

# Run single test with verbose output  
pytest tests/test_file_utils.py::TestValidateImagePath::test_valid_image_path -v -s
```

## Contributing

When contributing new features:

1. Write tests first (TDD approach)
2. Ensure 95%+ coverage for new code
3. Add performance tests for processing features
4. Include error handling tests
5. Update this README if adding new test categories

For questions or issues with the test suite, please check the test output and coverage reports first, then refer to the main project documentation.