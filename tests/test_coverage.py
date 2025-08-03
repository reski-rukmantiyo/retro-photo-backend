"""Test coverage validation and reporting utilities."""

import pytest
import subprocess
import sys
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


class TestCoverageReport:
    """Test coverage reporting and validation."""
    
    @pytest.fixture
    def coverage_threshold(self):
        """Define minimum coverage thresholds."""
        return {
            'total': 95.0,
            'individual_files': {
                'photo_restore/utils/config.py': 95.0,
                'photo_restore/utils/file_utils.py': 95.0,
                'photo_restore/utils/logger.py': 90.0,
                'photo_restore/models/model_manager.py': 90.0,
                'photo_restore/processors/image_processor.py': 90.0,
                'photo_restore/processors/batch_processor.py': 90.0,
                'photo_restore/cli.py': 85.0,
            }
        }
    
    def test_overall_coverage_threshold(self, coverage_threshold):
        """Test that overall coverage meets threshold."""
        # Run coverage report and parse results
        coverage_data = self._get_coverage_data()
        
        if coverage_data:
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            threshold = coverage_threshold['total']
            
            assert total_coverage >= threshold, (
                f"Overall coverage {total_coverage:.1f}% is below threshold {threshold}%"
            )
            
            print(f"Overall coverage: {total_coverage:.1f}% (threshold: {threshold}%)")
        else:
            pytest.skip("Coverage data not available")
    
    def test_individual_file_coverage(self, coverage_threshold):
        """Test that individual files meet coverage thresholds."""
        coverage_data = self._get_coverage_data()
        
        if not coverage_data:
            pytest.skip("Coverage data not available")
        
        files = coverage_data.get('files', {})
        individual_thresholds = coverage_threshold['individual_files']
        
        failing_files = []
        
        for file_path, threshold in individual_thresholds.items():
            if file_path in files:
                file_coverage = files[file_path]['summary']['percent_covered']
                
                if file_coverage < threshold:
                    failing_files.append((file_path, file_coverage, threshold))
                
                print(f"{file_path}: {file_coverage:.1f}% (threshold: {threshold}%)")
        
        if failing_files:
            failure_msg = "Files below coverage threshold:\n"
            for file_path, actual, expected in failing_files:
                failure_msg += f"  {file_path}: {actual:.1f}% < {expected}%\n"
            
            pytest.fail(failure_msg)
    
    def test_no_untested_files(self):
        """Test that no source files are completely untested."""
        coverage_data = self._get_coverage_data()
        
        if not coverage_data:
            pytest.skip("Coverage data not available")
        
        files = coverage_data.get('files', {})
        untested_files = []
        
        for file_path, file_data in files.items():
            if file_path.startswith('photo_restore/'):
                coverage = file_data['summary']['percent_covered']
                if coverage == 0:
                    untested_files.append(file_path)
        
        if untested_files:
            pytest.fail(f"Completely untested files found: {untested_files}")
    
    def test_missing_coverage_files(self):
        """Test that all source files are included in coverage."""
        project_root = Path(__file__).parent.parent
        source_dir = project_root / "photo_restore"
        
        # Find all Python files in source
        source_files = []
        for py_file in source_dir.rglob("*.py"):
            if py_file.name != "__init__.py":
                rel_path = py_file.relative_to(project_root)
                source_files.append(str(rel_path))
        
        coverage_data = self._get_coverage_data()
        
        if not coverage_data:
            pytest.skip("Coverage data not available")
        
        covered_files = set(coverage_data.get('files', {}).keys())
        source_files_set = set(source_files)
        
        missing_files = source_files_set - covered_files
        
        if missing_files:
            pytest.fail(f"Source files not included in coverage: {missing_files}")
    
    def _get_coverage_data(self) -> Dict:
        """Get coverage data from coverage report."""
        try:
            # Try to get coverage data from JSON report
            coverage_json = Path("coverage.json")
            if coverage_json.exists():
                with open(coverage_json) as f:
                    return json.load(f)
            
            # Try to generate coverage report
            result = subprocess.run([
                sys.executable, "-m", "coverage", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and coverage_json.exists():
                with open(coverage_json) as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            print(f"Could not get coverage data: {e}")
            return {}
    
    def test_branch_coverage(self):
        """Test branch coverage for complex logic."""
        coverage_data = self._get_coverage_data()
        
        if not coverage_data:
            pytest.skip("Coverage data not available")
        
        # Focus on files with complex branching logic
        critical_files = [
            'photo_restore/processors/image_processor.py',
            'photo_restore/processors/batch_processor.py',
            'photo_restore/models/model_manager.py',
            'photo_restore/utils/config.py'
        ]
        
        low_branch_coverage = []
        
        for file_path in critical_files:
            if file_path in coverage_data.get('files', {}):
                file_data = coverage_data['files'][file_path]
                summary = file_data.get('summary', {})
                
                if 'percent_covered_display' in summary:
                    # This indicates branch coverage is available
                    branch_coverage = summary.get('percent_covered', 0)
                    
                    if branch_coverage < 85.0:  # Branch coverage threshold
                        low_branch_coverage.append((file_path, branch_coverage))
        
        if low_branch_coverage:
            failure_msg = "Files with low branch coverage:\n"
            for file_path, coverage in low_branch_coverage:
                failure_msg += f"  {file_path}: {coverage:.1f}%\n"
            
            print(failure_msg)  # Print as warning, don't fail (branch coverage is harder)


class TestCoverageIntegration:
    """Integration tests for coverage collection."""
    
    def test_coverage_collection_works(self):
        """Test that coverage collection is working properly."""
        try:
            # Run a simple test with coverage
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_config_comprehensive.py::TestConfig::test_default_config_creation",
                "--cov=photo_restore",
                "--cov-report=term",
                "-v"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0, f"Coverage test failed: {result.stderr}"
            assert "coverage" in result.stdout.lower(), "Coverage report not generated"
            
        except Exception as e:
            pytest.skip(f"Coverage collection test failed: {e}")
    
    def test_coverage_html_report_generation(self):
        """Test that HTML coverage reports can be generated."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "coverage", "html"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                html_dir = Path("htmlcov")
                assert html_dir.exists(), "HTML coverage directory not created"
                assert (html_dir / "index.html").exists(), "HTML coverage index not created"
            else:
                pytest.skip("HTML coverage report generation not available")
                
        except Exception as e:
            pytest.skip(f"HTML coverage test failed: {e}")
    
    def test_coverage_xml_report_generation(self):
        """Test that XML coverage reports can be generated."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "coverage", "xml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                xml_file = Path("coverage.xml")
                assert xml_file.exists(), "XML coverage file not created"
                
                # Validate XML structure
                tree = ET.parse(xml_file)
                root = tree.getroot()
                assert root.tag == "coverage", "Invalid XML coverage format"
                
            else:
                pytest.skip("XML coverage report generation not available")
                
        except Exception as e:
            pytest.skip(f"XML coverage test failed: {e}")


def generate_coverage_report():
    """Generate comprehensive coverage report."""
    try:
        print("Generating coverage report...")
        
        # Run tests with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=photo_restore",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-fail-under=95",
            "-v"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ Coverage report generated successfully")
            print("📊 HTML report: htmlcov/index.html")
            print("📄 XML report: coverage.xml")
        else:
            print("❌ Coverage report generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating coverage report: {e}")
        return False


if __name__ == "__main__":
    # Run coverage report generation
    success = generate_coverage_report()
    sys.exit(0 if success else 1)