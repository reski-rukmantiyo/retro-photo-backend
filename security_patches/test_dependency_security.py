"""
Test suite for dependency security module
Tests vulnerability detection, model integrity, and secure loading functions
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import hashlib

# Import the security module
from dependency_security import (
    SecureDependencyManager,
    RestrictedTorchUnpickler,
    secure_torch_load,
    ModelIntegrityError,
    VulnerablePackageError,
    generate_security_report
)


class TestSecureDependencyManager(unittest.TestCase):
    """Test cases for SecureDependencyManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.security_mgr = SecureDependencyManager(cache_dir=Path(self.temp_dir))
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_vulnerability_detection(self):
        """Test detection of vulnerable packages."""
        # Mock installed packages
        mock_packages = {
            'torch': '1.8.0',  # Vulnerable
            'Pillow': '8.0.0',  # Vulnerable
            'numpy': '1.26.4',  # Safe
            'requests': '2.31.0'  # Safe
        }
        
        with patch.object(self.security_mgr, '_get_installed_packages', return_value=mock_packages):
            scan_results = self.security_mgr.scan_dependencies()
            
            # Check results
            self.assertEqual(len(scan_results['vulnerabilities']), 2)
            
            # Verify torch vulnerability detected
            torch_vuln = next(v for v in scan_results['vulnerabilities'] if v['package'] == 'torch')
            self.assertEqual(torch_vuln['severity'], 'CRITICAL')
            self.assertIn('CVE-2022-45907', torch_vuln['cves'])
            
            # Verify Pillow vulnerability detected
            pillow_vuln = next(v for v in scan_results['vulnerabilities'] if v['package'] == 'Pillow')
            self.assertEqual(pillow_vuln['severity'], 'HIGH')
    
    def test_version_comparison(self):
        """Test version comparison logic."""
        # Test basic comparisons
        self.assertEqual(self.security_mgr._compare_versions('1.0.0', '2.0.0'), -1)
        self.assertEqual(self.security_mgr._compare_versions('2.0.0', '1.0.0'), 1)
        self.assertEqual(self.security_mgr._compare_versions('1.0.0', '1.0.0'), 0)
        
        # Test complex versions
        self.assertEqual(self.security_mgr._compare_versions('1.9.1', '2.0.0'), -1)
        self.assertEqual(self.security_mgr._compare_versions('2.2.1', '2.2.0'), 1)
        self.assertEqual(self.security_mgr._compare_versions('1.10.0', '1.9.0'), 1)
    
    def test_model_integrity_verification(self):
        """Test model integrity checking."""
        # Create a test file
        test_model_path = Path(self.temp_dir) / 'test_model.pth'
        test_content = b'This is a test model file'
        test_model_path.write_bytes(test_content)
        
        # Calculate expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        # Add to trusted hashes
        self.security_mgr.TRUSTED_MODEL_HASHES['test_model.pth'] = expected_hash
        
        # Test successful verification
        self.assertTrue(self.security_mgr.verify_model_integrity(test_model_path))
        
        # Test failed verification with wrong hash
        self.security_mgr.TRUSTED_MODEL_HASHES['test_model.pth'] = 'wrong_hash'
        with self.assertRaises(ModelIntegrityError):
            self.security_mgr.verify_model_integrity(test_model_path)
        
        # Test unknown model
        unknown_model = Path(self.temp_dir) / 'unknown_model.pth'
        unknown_model.write_bytes(b'Unknown model')
        self.assertFalse(self.security_mgr.verify_model_integrity(unknown_model))
    
    @patch('requests.get')
    def test_secure_model_download(self, mock_get):
        """Test secure model downloading."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content = lambda chunk_size: [b'chunk1', b'chunk2', b'chunk3']
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Set up expected hash
        test_content = b'chunk1chunk2chunk3'
        expected_hash = hashlib.sha256(test_content).hexdigest()
        self.security_mgr.TRUSTED_MODEL_HASHES['RealESRGAN_x4plus.pth'] = expected_hash
        
        # Test download
        destination = Path(self.temp_dir) / 'downloaded_model.pth'
        result = self.security_mgr.secure_download_model('RealESRGAN_x4plus.pth', destination)
        
        # Verify download
        self.assertTrue(result.exists())
        self.assertEqual(result.read_bytes(), test_content)
        
        # Verify SSL and timeout were used
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertTrue(call_args.kwargs['verify'])
        self.assertEqual(call_args.kwargs['timeout'], 300)
    
    def test_secure_requirements_generation(self):
        """Test generation of secure requirements.txt."""
        output_path = Path(self.temp_dir) / 'secure_requirements.txt'
        self.security_mgr.create_secure_requirements(output_path)
        
        # Verify file created
        self.assertTrue(output_path.exists())
        
        # Check content
        content = output_path.read_text()
        self.assertIn('torch==2.2.1+cpu', content)
        self.assertIn('--hash=', content)
        self.assertIn('# Security tools', content)
        self.assertIn('pip-audit==2.7.0', content)
    
    @patch('subprocess.run')
    def test_security_audit(self, mock_run):
        """Test comprehensive security audit."""
        # Mock pip-audit response
        pip_audit_result = Mock()
        pip_audit_result.returncode = 0
        pip_audit_result.stdout = json.dumps({
            'dependencies': [],
            'vulnerabilities': []
        })
        
        # Mock safety response
        safety_result = Mock()
        safety_result.returncode = 0
        safety_result.stdout = json.dumps({
            'vulnerabilities': []
        })
        
        # Mock pip list response
        pip_list_result = Mock()
        pip_list_result.returncode = 0
        pip_list_result.stdout = json.dumps([
            {'name': 'torch', 'version': '2.2.1'},
            {'name': 'numpy', 'version': '1.26.4'}
        ])
        
        # Configure mock to return different results based on command
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if 'pip_audit' in cmd:
                return pip_audit_result
            elif 'safety' in cmd:
                return safety_result
            elif 'pip' in cmd and 'list' in cmd:
                return pip_list_result
            return Mock(returncode=1, stderr='Unknown command')
        
        mock_run.side_effect = side_effect
        
        # Run audit
        audit_results = self.security_mgr.run_security_audit()
        
        # Verify results structure
        self.assertIn('timestamp', audit_results)
        self.assertIn('pip_audit', audit_results)
        self.assertIn('safety_check', audit_results)
        self.assertIn('custom_scan', audit_results)
        
        # Verify audit file created
        audit_files = list(Path(self.temp_dir).glob('security_audit_*.json'))
        self.assertEqual(len(audit_files), 1)


class TestRestrictedUnpickler(unittest.TestCase):
    """Test cases for RestrictedTorchUnpickler."""
    
    def test_allowed_modules(self):
        """Test that allowed modules can be loaded."""
        # Create a safe pickle with allowed modules
        safe_data = {'torch': 'safe', 'numpy': 'safe'}
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        try:
            with open(temp_file.name, 'wb') as f:
                pickle.dump(safe_data, f)
            
            # Test loading with restricted unpickler
            with open(temp_file.name, 'rb') as f:
                unpickler = RestrictedTorchUnpickler(f)
                loaded_data = unpickler.load()
            
            self.assertEqual(loaded_data, safe_data)
        finally:
            os.unlink(temp_file.name)
    
    def test_disallowed_modules(self):
        """Test that dangerous modules are blocked."""
        # Mock find_class to test module restrictions
        unpickler = RestrictedTorchUnpickler(None)
        
        # Test dangerous modules
        with self.assertRaises(pickle.UnpicklingError):
            unpickler.find_class('os', 'system')
        
        with self.assertRaises(pickle.UnpicklingError):
            unpickler.find_class('subprocess', 'Popen')
        
        with self.assertRaises(pickle.UnpicklingError):
            unpickler.find_class('builtins', 'eval')
    
    def test_dangerous_classes(self):
        """Test that dangerous classes are blocked even in allowed modules."""
        unpickler = RestrictedTorchUnpickler(None)
        
        # Test dangerous classes
        dangerous_classes = ['eval', 'exec', 'compile', '__import__', 'open']
        
        for cls in dangerous_classes:
            with self.assertRaises(pickle.UnpicklingError):
                unpickler.find_class('torch', cls)


class TestSecureTorchLoad(unittest.TestCase):
    """Test cases for secure_torch_load function."""
    
    @patch('torch.load')
    @patch('torch.__version__', '2.2.1')
    def test_weights_only_loading(self, mock_load):
        """Test loading with weights_only parameter for new PyTorch."""
        # Mock successful load
        mock_load.return_value = {'model': 'weights'}
        
        # Test secure load
        model_path = Path('test_model.pth')
        result = secure_torch_load(model_path)
        
        # Verify weights_only was used
        mock_load.assert_called_once_with(
            model_path,
            map_location='cpu',
            weights_only=True
        )
        self.assertEqual(result, {'model': 'weights'})
    
    @patch('torch.__version__', '1.9.0')
    def test_fallback_to_restricted_unpickler(self):
        """Test fallback to restricted unpickler for old PyTorch."""
        # Create a test model file
        test_data = {'layer': 'conv', 'weights': [1, 2, 3]}
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        try:
            with open(temp_file.name, 'wb') as f:
                pickle.dump(test_data, f)
            
            # Mock torch.serialization._legacy_load
            with patch('torch.serialization._legacy_load', return_value=None):
                # Test secure load
                result = secure_torch_load(Path(temp_file.name))
                self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_file.name)


class TestSecurityReport(unittest.TestCase):
    """Test security report generation."""
    
    def test_report_generation(self):
        """Test comprehensive security report generation."""
        # Create mock security manager
        security_mgr = Mock()
        security_mgr.run_security_audit.return_value = {
            'timestamp': '2025-01-01T00:00:00',
            'custom_scan': {
                'vulnerabilities': [
                    {
                        'package': 'torch',
                        'installed_version': '1.8.0',
                        'severity': 'CRITICAL',
                        'cves': ['CVE-2022-45907'],
                        'vulnerable_versions': ['<2.2.0']
                    }
                ]
            },
            'pip_audit': {},
            'safety_check': {}
        }
        
        # Generate report
        report = generate_security_report(security_mgr)
        
        # Verify report content
        self.assertIn('Dependency Security Report', report)
        self.assertIn('Critical Vulnerabilities Found: 1', report)
        self.assertIn('torch', report)
        self.assertIn('CVE-2022-45907', report)
        self.assertIn('Recommendations', report)
        self.assertIn('Security Checklist', report)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete security system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.security_mgr = SecureDependencyManager(cache_dir=Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_security_check(self):
        """Test complete security check workflow."""
        # Create mock model file
        model_path = Path(self.temp_dir) / 'test_model.pth'
        model_content = b'Mock PyTorch model data'
        model_path.write_bytes(model_content)
        
        # Add to trusted models
        model_hash = hashlib.sha256(model_content).hexdigest()
        self.security_mgr.TRUSTED_MODEL_HASHES['test_model.pth'] = model_hash
        
        # Test complete workflow
        # 1. Verify model integrity
        self.assertTrue(self.security_mgr.verify_model_integrity(model_path))
        
        # 2. Scan dependencies (mocked)
        with patch.object(self.security_mgr, '_get_installed_packages', 
                         return_value={'torch': '2.2.1', 'numpy': '1.26.4'}):
            scan_results = self.security_mgr.scan_dependencies()
            self.assertEqual(len(scan_results['vulnerabilities']), 0)
        
        # 3. Generate secure requirements
        req_path = Path(self.temp_dir) / 'requirements.txt'
        self.security_mgr.create_secure_requirements(req_path)
        self.assertTrue(req_path.exists())
    
    def test_malicious_model_detection(self):
        """Test detection of tampered model files."""
        # Create legitimate model
        model_path = Path(self.temp_dir) / 'legitimate_model.pth'
        legitimate_content = b'Legitimate model content'
        model_path.write_bytes(legitimate_content)
        
        # Add correct hash
        correct_hash = hashlib.sha256(legitimate_content).hexdigest()
        self.security_mgr.TRUSTED_MODEL_HASHES['legitimate_model.pth'] = correct_hash
        
        # Verify legitimate model passes
        self.assertTrue(self.security_mgr.verify_model_integrity(model_path))
        
        # Tamper with model
        model_path.write_bytes(b'Malicious content injected')
        
        # Verify tampered model fails
        with self.assertRaises(ModelIntegrityError) as context:
            self.security_mgr.verify_model_integrity(model_path)
        
        self.assertIn('Model integrity check failed', str(context.exception))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)