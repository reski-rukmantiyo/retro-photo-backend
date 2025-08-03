#!/usr/bin/env python3
"""Comprehensive test runner for Photo Restoration CLI project."""

import sys
import subprocess
import argparse
from pathlib import Path
import time
from typing import List, Dict, Optional


class TestRunner:
    """Comprehensive test runner with different test categories."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_categories = {
            'unit': [
                'tests/test_config_comprehensive.py',
                'tests/test_file_utils.py',
                'tests/test_logger.py',
            ],
            'integration': [
                'tests/test_model_manager.py',
                'tests/test_image_processor.py',
                'tests/test_batch_processor.py',
                'tests/test_cli.py',
            ],
            'performance': [
                'tests/test_performance.py',
            ],
            'error_handling': [
                'tests/test_error_handling.py',
            ],
            'coverage': [
                'tests/test_coverage.py',
            ]
        }
    
    def run_test_category(self, category: str, verbose: bool = False, 
                         fail_fast: bool = False) -> bool:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            print(f"❌ Unknown test category: {category}")
            print(f"Available categories: {list(self.test_categories.keys())}")
            return False
        
        test_files = self.test_categories[category]
        print(f"🧪 Running {category} tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add test files
        cmd.extend(test_files)
        
        # Add common options
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        if fail_fast:
            cmd.append("-x")
        
        # Category-specific options
        if category == 'performance':
            cmd.extend(["-m", "performance", "--tb=short"])
        elif category == 'coverage':
            cmd.extend([
                "--cov=photo_restore",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-fail-under=95"
            ])
        elif category in ['unit', 'integration']:
            cmd.extend(["-m", "not performance and not slow"])
        
        # Run tests
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {category} tests passed ({duration:.1f}s)")
            return True
        else:
            print(f"❌ {category} tests failed ({duration:.1f}s)")
            return False
    
    def run_all_tests(self, verbose: bool = False, fail_fast: bool = False,
                     skip_slow: bool = False) -> Dict[str, bool]:
        """Run all test categories."""
        print("🚀 Running comprehensive test suite...")
        results = {}
        
        # Define test order (unit first, performance last)
        test_order = ['unit', 'integration', 'error_handling', 'coverage']
        if not skip_slow:
            test_order.append('performance')
        
        total_start = time.time()
        
        for category in test_order:
            success = self.run_test_category(category, verbose, fail_fast)
            results[category] = success
            
            if not success and fail_fast:
                print(f"🛑 Stopping due to {category} test failure")
                break
            
            print()  # Add spacing between categories
        
        total_duration = time.time() - total_start
        
        # Print summary
        self._print_test_summary(results, total_duration)
        
        return results
    
    def run_quick_tests(self, verbose: bool = False) -> bool:
        """Run quick tests (unit + basic integration)."""
        print("⚡ Running quick test suite...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_config_comprehensive.py",
            "tests/test_file_utils.py",
            "tests/test_logger.py",
            "-m", "not slow and not performance",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Quick tests passed ({duration:.1f}s)")
            return True
        else:
            print(f"❌ Quick tests failed ({duration:.1f}s)")
            return False
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests to verify basic functionality."""
        print("💨 Running smoke tests...")
        
        smoke_tests = [
            'tests/test_config_comprehensive.py::TestConfig::test_default_config_creation',
            'tests/test_file_utils.py::TestValidateImagePath::test_valid_image_path',
            'tests/test_logger.py::TestSetupLogger::test_default_logger_setup',
        ]
        
        cmd = [
            sys.executable, "-m", "pytest",
            *smoke_tests,
            "--tb=short",
            "-q"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Smoke tests passed ({duration:.1f}s)")
            return True
        else:
            print(f"❌ Smoke tests failed ({duration:.1f}s)")
            return False
    
    def run_coverage_only(self) -> bool:
        """Run all tests with coverage reporting."""
        print("📊 Running tests with coverage...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=photo_restore",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=95",
            "-m", "not performance",  # Skip performance tests for coverage
            "--tb=short"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Coverage tests passed ({duration:.1f}s)")
            print("📄 Coverage report: htmlcov/index.html")
            return True
        else:
            print(f"❌ Coverage tests failed ({duration:.1f}s)")
            return False
    
    def _print_test_summary(self, results: Dict[str, bool], duration: float):
        """Print test summary."""
        print("=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for category, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{category:15} {status}")
        
        print("-" * 60)
        print(f"Results: {passed}/{total} categories passed")
        print(f"Duration: {duration:.1f}s")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("💥 Some tests failed!")
        
        print("=" * 60)
    
    def lint_code(self) -> bool:
        """Run code linting."""
        print("🔍 Running code linting...")
        
        # Check if flake8 is available
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8",
                "photo_restore/",
                "--max-line-length=100",
                "--ignore=E203,W503",
                "--exclude=__pycache__,*.pyc"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Code linting passed")
                return True
            else:
                print("❌ Code linting failed:")
                print(result.stdout)
                return False
                
        except FileNotFoundError:
            print("⚠️  flake8 not available, skipping linting")
            return True
    
    def format_code(self) -> bool:
        """Format code with black."""
        print("🎨 Formatting code...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "black",
                "photo_restore/",
                "tests/",
                "--check",
                "--diff"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Code formatting is correct")
                return True
            else:
                print("❌ Code formatting issues found:")
                print(result.stdout)
                print("\n💡 Run 'python -m black photo_restore/ tests/' to fix")
                return False
                
        except FileNotFoundError:
            print("⚠️  black not available, skipping formatting check")
            return True


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Photo Restoration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  unit         - Unit tests for individual modules
  integration  - Integration tests for component interaction
  performance  - Performance and benchmark tests
  error        - Error handling and edge case tests
  coverage     - Coverage validation tests

Examples:
  python run_tests.py --all              # Run all tests
  python run_tests.py --quick            # Run quick tests only
  python run_tests.py --smoke            # Run smoke tests only
  python run_tests.py --category unit    # Run unit tests only
  python run_tests.py --coverage-only    # Run with coverage reporting
  python run_tests.py --lint             # Run code linting
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--all', action='store_true',
                           help='Run all test categories')
    test_group.add_argument('--quick', action='store_true',
                           help='Run quick tests only')
    test_group.add_argument('--smoke', action='store_true',
                           help='Run smoke tests only')
    test_group.add_argument('--category', choices=['unit', 'integration', 'performance', 'error_handling', 'coverage'],
                           help='Run specific test category')
    test_group.add_argument('--coverage-only', action='store_true',
                           help='Run all tests with coverage reporting')
    
    # Code quality options
    parser.add_argument('--lint', action='store_true',
                       help='Run code linting')
    parser.add_argument('--format', action='store_true',
                       help='Check code formatting')
    
    # Test execution options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--fail-fast', '-x', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow tests (like performance tests)')
    
    args = parser.parse_args()
    
    # If no options specified, run quick tests by default
    if not any([args.all, args.quick, args.smoke, args.category, 
               args.coverage_only, args.lint, args.format]):
        args.quick = True
    
    runner = TestRunner()
    success = True
    
    # Run code quality checks
    if args.lint:
        success &= runner.lint_code()
    
    if args.format:
        success &= runner.format_code()
    
    # Run tests
    if args.all:
        results = runner.run_all_tests(args.verbose, args.fail_fast, args.skip_slow)
        success &= all(results.values())
    
    elif args.quick:
        success &= runner.run_quick_tests(args.verbose)
    
    elif args.smoke:
        success &= runner.run_smoke_tests()
    
    elif args.category:
        success &= runner.run_test_category(args.category, args.verbose, args.fail_fast)
    
    elif args.coverage_only:
        success &= runner.run_coverage_only()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()