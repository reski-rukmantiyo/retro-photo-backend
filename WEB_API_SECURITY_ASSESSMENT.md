# Web API Security Assessment & Implementation Guide

## Executive Summary

This comprehensive security assessment addresses the transition from a CLI-based photo restoration tool to a web-accessible API. Based on the critical vulnerabilities discovered and patches implemented, this guide provides defense-in-depth security controls for the web interface, which will become the primary attack vector.

**Risk Level**: **CRITICAL** - Web APIs expose significantly more attack surface than CLI tools

## 1. Threat Model & Attack Vectors

### Primary Attack Vectors
1. **File Upload Attacks** (CRITICAL)
   - Malicious image files (polyglot attacks, embedded executables)
   - Path traversal via filename manipulation
   - DoS through large/complex files
   - Image bombs (decompression attacks)

2. **Model Poisoning** (CRITICAL)
   - Malicious model file uploads (if allowed)
   - PyTorch deserialization RCE
   - Model substitution attacks

3. **API Abuse** (HIGH)
   - Resource exhaustion (CPU/Memory DoS)
   - Rate limit bypass
   - Concurrent request flooding

4. **Authentication/Authorization** (HIGH)
   - Unauthorized access to processing resources
   - Cross-tenant data access
   - Session hijacking

5. **Injection Attacks** (MEDIUM)
   - Command injection via metadata
   - SQL injection in job tracking
   - NoSQL injection in Redis

## 2. Security Architecture

### Layered Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                    WAF / DDoS Protection                     │
├─────────────────────────────────────────────────────────────┤
│                    Rate Limiting Layer                       │
├─────────────────────────────────────────────────────────────┤
│              Authentication & Authorization                  │
├─────────────────────────────────────────────────────────────┤
│                    Input Validation                          │
├─────────────────────────────────────────────────────────────┤
│                   Secure Processing                          │
├─────────────────────────────────────────────────────────────┤
│                    Output Sanitization                       │
└─────────────────────────────────────────────────────────────┘
```

## 3. Web-Specific Security Controls

### 3.1 Secure File Upload Implementation

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import aiofiles
import tempfile
import os
from typing import Optional
import magic
import hashlib
import uuid

# Import existing security modules
from security_patches.secure_image_validator import SecureImageValidator
from security_patches.path_validator import SecurePathValidator

class SecureFileUploadHandler:
    """
    Secure file upload handler with multiple validation layers.
    """
    
    def __init__(self):
        self.image_validator = SecureImageValidator(
            max_file_size=20 * 1024 * 1024,  # 20MB
            max_dimensions=(8000, 8000),
            min_dimensions=(100, 100)
        )
        self.path_validator = SecurePathValidator()
        self.upload_dir = tempfile.mkdtemp(prefix="secure_upload_")
        self.quarantine_dir = tempfile.mkdtemp(prefix="quarantine_")
        
    async def handle_upload(
        self, 
        file: UploadFile,
        user_id: str,
        request_id: str
    ) -> dict:
        """
        Handle file upload with comprehensive security checks.
        """
        # 1. Generate secure filename
        file_id = str(uuid.uuid4())
        extension = self._get_safe_extension(file.filename)
        secure_filename = f"{user_id}_{request_id}_{file_id}{extension}"
        
        # 2. Create isolated user directory
        user_dir = os.path.join(self.upload_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # 3. Save to quarantine first
        quarantine_path = os.path.join(self.quarantine_dir, secure_filename)
        
        try:
            # Stream file to disk with size limit
            bytes_written = 0
            async with aiofiles.open(quarantine_path, 'wb') as f:
                while chunk := await file.read(8192):  # 8KB chunks
                    bytes_written += len(chunk)
                    if bytes_written > self.image_validator.max_file_size:
                        raise HTTPException(
                            status_code=413,
                            detail="File too large"
                        )
                    await f.write(chunk)
            
            # 4. Validate file in quarantine
            validation_result = await self._validate_quarantined_file(
                quarantine_path, 
                file.content_type
            )
            
            # 5. Move to processing directory if valid
            final_path = os.path.join(user_dir, secure_filename)
            os.rename(quarantine_path, final_path)
            
            return {
                "file_id": file_id,
                "path": final_path,
                "validation": validation_result
            }
            
        except Exception as e:
            # Clean up on any error
            if os.path.exists(quarantine_path):
                os.remove(quarantine_path)
            raise
    
    async def _validate_quarantined_file(
        self, 
        file_path: str, 
        content_type: str
    ) -> dict:
        """
        Comprehensive file validation in quarantine.
        """
        # 1. Content-Type validation
        if content_type not in [
            'image/jpeg', 'image/png', 'image/gif', 
            'image/bmp', 'image/tiff', 'image/webp'
        ]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported content type: {content_type}"
            )
        
        # 2. Magic byte validation
        file_magic = magic.from_file(file_path, mime=True)
        if file_magic != content_type:
            raise HTTPException(
                status_code=400,
                detail="File content doesn't match declared type"
            )
        
        # 3. Use existing image validator
        try:
            validation_result = self.image_validator.validate_image(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Image validation failed: {str(e)}"
            )
        
        # 4. Additional web-specific checks
        await self._check_exif_data(file_path)
        await self._scan_for_malware(file_path)
        
        return validation_result
    
    async def _check_exif_data(self, file_path: str):
        """Check for malicious EXIF data."""
        # Implementation would check for:
        # - Suspicious EXIF tags
        # - Embedded scripts
        # - GPS data (privacy)
        pass
    
    async def _scan_for_malware(self, file_path: str):
        """Integrate with antivirus scanning."""
        # Implementation would use ClamAV or similar
        pass
    
    def _get_safe_extension(self, filename: str) -> str:
        """Extract and validate file extension."""
        if not filename:
            return ".jpg"
        
        # Only allow last extension to prevent double extension attacks
        parts = filename.rsplit('.', 1)
        if len(parts) != 2:
            return ".jpg"
        
        ext = f".{parts[1].lower()}"
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
            return ".jpg"
        
        return ext
```

### 3.2 Authentication & Authorization

```python
from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from sqlalchemy.orm import Session

class SecureAuthSystem:
    """
    JWT-based authentication with rate limiting and session management.
    """
    
    SECRET_KEY = os.environ.get("JWT_SECRET_KEY")  # Must be set in environment
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client = redis.Redis(decode_responses=True)
        self.security = HTTPBearer()
    
    def create_access_token(
        self, 
        user_id: str, 
        scopes: List[str] = None
    ) -> str:
        """Create JWT access token with scopes."""
        expire = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        claims = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "scopes": scopes or []
        }
        
        # Add additional security claims
        claims["jti"] = str(uuid.uuid4())  # JWT ID for revocation
        
        return jwt.encode(claims, self.SECRET_KEY, algorithm=self.ALGORITHM)
    
    async def verify_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> dict:
        """Verify JWT token and check revocation."""
        token = credentials.credentials
        
        try:
            # Decode token
            payload = jwt.decode(
                token, 
                self.SECRET_KEY, 
                algorithms=[self.ALGORITHM]
            )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if self.redis_client.get(f"revoked_token:{jti}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            return payload
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    
    def require_scopes(self, required_scopes: List[str]):
        """Decorator to check required scopes."""
        async def scope_checker(
            token_data: dict = Depends(self.verify_token)
        ):
            token_scopes = token_data.get("scopes", [])
            for scope in required_scopes:
                if scope not in token_scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required scope: {scope}"
                    )
            return token_data
        return scope_checker
```

### 3.3 Rate Limiting Implementation

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import time

class AdaptiveRateLimiter:
    """
    Adaptive rate limiting with different tiers and DoS protection.
    """
    
    def __init__(self, app: FastAPI):
        # Create limiter with Redis backend
        self.limiter = Limiter(
            key_func=self._get_rate_limit_key,
            default_limits=["100 per hour"],
            storage_uri="redis://localhost:6379"
        )
        
        # Add to FastAPI
        app.state.limiter = self.limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
        
        # Define rate limit tiers
        self.tiers = {
            "anonymous": ["10 per hour", "2 per minute"],
            "free": ["100 per hour", "10 per minute"],
            "premium": ["1000 per hour", "50 per minute"],
            "enterprise": ["10000 per hour", "200 per minute"]
        }
        
    def _get_rate_limit_key(self, request):
        """Generate rate limit key based on user authentication."""
        # Try to get authenticated user
        user_id = self._get_user_from_request(request)
        
        if user_id:
            # Use user ID for authenticated users
            return f"user:{user_id}"
        else:
            # Use IP for anonymous users
            return get_remote_address(request)
    
    def get_tier_limits(self, user_tier: str):
        """Get rate limits for user tier."""
        return self.tiers.get(user_tier, self.tiers["anonymous"])
    
    def apply_endpoint_limit(self, limits: List[str]):
        """Decorator for endpoint-specific rate limits."""
        def decorator(func):
            # Apply multiple rate limits
            for limit in limits:
                func = self.limiter.limit(limit)(func)
            return func
        return decorator
    
    async def check_resource_limits(self, user_id: str, resource_type: str):
        """Check resource-based rate limits (CPU time, storage, etc)."""
        key = f"resource:{user_id}:{resource_type}"
        current = self.redis_client.get(key)
        
        limits = {
            "cpu_minutes": 60,  # 60 CPU minutes per day
            "storage_mb": 1000,  # 1GB storage per day
            "api_calls": 1000    # 1000 API calls per day
        }
        
        if current and int(current) >= limits.get(resource_type, 0):
            raise HTTPException(
                status_code=429,
                detail=f"Resource limit exceeded for {resource_type}"
            )
```

### 3.4 API Security Headers

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import base64

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add comprehensive security headers to all responses.
    """
    
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        
        # Content Security Policy
        csp = {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline'",  # Adjust as needed
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self'",
            "connect-src": "'self'",
            "media-src": "'none'",
            "object-src": "'none'",
            "frame-ancestors": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "upgrade-insecure-requests": ""
        }
        
        csp_string = "; ".join([f"{k} {v}" for k, v in csp.items()])
        response.headers["Content-Security-Policy"] = csp_string
        
        # HSTS (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        return response

def configure_security_middleware(app: FastAPI):
    """Configure all security middleware."""
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],  # Specific origins only
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=3600
    )
    
    # Trusted host validation
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
```

## 4. Processing Security

### 4.1 Secure Job Processing

```python
from celery import Celery, Task
from celery.exceptions import SoftTimeLimitExceeded
import resource
import os
import psutil

class SecureProcessingTask(Task):
    """
    Secure Celery task with resource limits and isolation.
    """
    
    # Resource limits
    time_limit = 300  # 5 minutes hard limit
    soft_time_limit = 240  # 4 minutes soft limit
    max_memory_mb = 2048  # 2GB memory limit
    
    def __call__(self, *args, **kwargs):
        """Execute task with resource limits."""
        # Set memory limit
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.max_memory_mb * 1024 * 1024, -1)
        )
        
        # Set CPU priority
        os.nice(10)  # Lower priority
        
        # Monitor resources during execution
        return super().__call__(*args, **kwargs)

@celery_app.task(
    base=SecureProcessingTask,
    bind=True,
    max_retries=3
)
def process_image_secure(self, job_id: str, file_path: str, options: dict):
    """
    Process image with security constraints.
    """
    try:
        # 1. Validate inputs again (defense in depth)
        validator = SecurePathValidator()
        safe_path = validator.validate_path(file_path)
        
        # 2. Create isolated processing environment
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy input to temp directory
            temp_input = os.path.join(temp_dir, "input.jpg")
            shutil.copy2(safe_path, temp_input)
            
            # 3. Process with monitoring
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss
            
            # Use existing secure processors
            processor = ImageProcessor(config)
            result_path = processor.process_single(temp_input)
            
            # 4. Validate output
            output_validator = SecureImageValidator()
            output_validator.validate_image(result_path)
            
            # 5. Move to secure output location
            output_path = create_secure_output_path(job_id)
            shutil.move(result_path, output_path)
            
            # 6. Log metrics
            processing_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - memory_before
            
            return {
                "status": "completed",
                "output_path": output_path,
                "metrics": {
                    "processing_time": processing_time,
                    "memory_used_mb": memory_used / 1024 / 1024
                }
            }
            
    except SoftTimeLimitExceeded:
        # Handle timeout gracefully
        self.update_state(
            state='FAILURE',
            meta={'error': 'Processing timeout'}
        )
        raise
    except Exception as e:
        # Log error securely (no sensitive data)
        logger.error(f"Processing failed for job {job_id}: {type(e).__name__}")
        raise
```

## 5. Data Security

### 5.1 Secure Storage

```python
import boto3
from cryptography.fernet import Fernet
import base64

class SecureStorageManager:
    """
    Encrypted storage with access controls.
    """
    
    def __init__(self):
        # Use environment variable for encryption key
        key = os.environ.get("STORAGE_ENCRYPTION_KEY")
        self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
        
        # S3 client with encryption
        self.s3 = boto3.client(
            's3',
            config=boto3.session.Config(signature_version='s3v4')
        )
        self.bucket_name = os.environ.get("S3_BUCKET_NAME")
    
    async def store_processed_image(
        self, 
        file_path: str, 
        user_id: str, 
        job_id: str
    ) -> str:
        """Store processed image with encryption."""
        # Generate secure S3 key
        s3_key = f"processed/{user_id}/{job_id}/{uuid.uuid4()}.enc"
        
        # Encrypt file
        with open(file_path, 'rb') as f:
            encrypted_data = self.cipher.encrypt(f.read())
        
        # Upload with server-side encryption
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=encrypted_data,
            ServerSideEncryption='AES256',
            Metadata={
                'user_id': user_id,
                'job_id': job_id,
                'encrypted': 'true'
            }
        )
        
        # Generate time-limited presigned URL
        url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=3600  # 1 hour
        )
        
        return url
    
    async def cleanup_expired_files(self):
        """Remove expired files based on retention policy."""
        # Implementation would check file age and remove old files
        pass
```

## 6. Monitoring & Logging

### 6.1 Security Monitoring

```python
import structlog
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk

# Metrics
security_events = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method', 'status']
)

active_jobs = Gauge(
    'active_processing_jobs',
    'Number of active processing jobs'
)

class SecurityLogger:
    """
    Structured logging for security events.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger()
        
        # Initialize Sentry for error tracking
        sentry_sdk.init(
            dsn=os.environ.get("SENTRY_DSN"),
            traces_sample_rate=0.1
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: dict,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log security event with context."""
        # Increment metrics
        security_events.labels(
            event_type=event_type,
            severity=severity
        ).inc()
        
        # Structure log entry
        self.logger.warning(
            "security_event",
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            **details
        )
        
        # Alert on critical events
        if severity == "critical":
            self._send_security_alert(event_type, details)
    
    def _send_security_alert(self, event_type: str, details: dict):
        """Send immediate alert for critical security events."""
        # Implementation would send to PagerDuty, Slack, etc.
        pass
```

## 7. API Security Checklist

### Pre-Deployment

- [ ] **Authentication & Authorization**
  - [ ] JWT implementation with secure key management
  - [ ] Role-based access control (RBAC)
  - [ ] API key management for service accounts
  - [ ] Session management and timeout policies

- [ ] **Input Validation**
  - [ ] File upload validation (size, type, content)
  - [ ] Request body validation with Pydantic
  - [ ] Query parameter sanitization
  - [ ] Header validation

- [ ] **Rate Limiting**
  - [ ] Per-user rate limits
  - [ ] Endpoint-specific limits
  - [ ] Resource-based quotas
  - [ ] DDoS protection

- [ ] **Secure Communication**
  - [ ] TLS 1.3 only
  - [ ] Certificate pinning for mobile apps
  - [ ] Secure WebSocket implementation

### Runtime Security

- [ ] **Monitoring**
  - [ ] Security event logging
  - [ ] Anomaly detection
  - [ ] Performance monitoring
  - [ ] Resource usage tracking

- [ ] **Incident Response**
  - [ ] Security runbooks
  - [ ] Automated response to threats
  - [ ] Data breach procedures
  - [ ] Regular security drills

### Compliance

- [ ] **Data Protection**
  - [ ] GDPR compliance (if applicable)
  - [ ] Data retention policies
  - [ ] Right to deletion implementation
  - [ ] Privacy policy enforcement

- [ ] **Audit Trail**
  - [ ] Comprehensive logging
  - [ ] Log retention and rotation
  - [ ] Tamper-proof audit logs
  - [ ] Regular security audits

## 8. Security Testing

### 8.1 Security Test Suite

```python
import pytest
from httpx import AsyncClient
import asyncio

class TestAPISecurity:
    """
    Comprehensive security test suite.
    """
    
    @pytest.mark.asyncio
    async def test_sql_injection(self, client: AsyncClient):
        """Test SQL injection protection."""
        payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--"
        ]
        
        for payload in payloads:
            response = await client.get(f"/api/v1/jobs/{payload}")
            assert response.status_code in [400, 404]
            assert "error" not in response.text.lower()
    
    @pytest.mark.asyncio
    async def test_path_traversal(self, client: AsyncClient):
        """Test path traversal protection."""
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]
        
        for payload in payloads:
            response = await client.post(
                "/api/v1/enhance",
                files={"file": (payload, b"fake_content", "image/jpeg")}
            )
            assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client: AsyncClient):
        """Test rate limit enforcement."""
        # Make requests up to limit
        for i in range(10):
            response = await client.get("/api/v1/health")
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = await client.get("/api/v1/health")
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_authentication_bypass(self, client: AsyncClient):
        """Test authentication cannot be bypassed."""
        # Test missing token
        response = await client.post("/api/v1/enhance")
        assert response.status_code == 401
        
        # Test invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = await client.post("/api/v1/enhance", headers=headers)
        assert response.status_code == 401
        
        # Test expired token
        expired_token = create_expired_token()
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = await client.post("/api/v1/enhance", headers=headers)
        assert response.status_code == 401
```

## 9. Deployment Security

### Container Security

```dockerfile
# Secure Dockerfile
FROM python:3.11-slim-bookworm

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/
WORKDIR /app

# Install Python dependencies as root
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . /app/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run with limited permissions
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

### Kubernetes Security

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: photo-restore-api
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  
  containers:
  - name: api
    image: photo-restore-api:latest
    
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    
    resources:
      limits:
        memory: "2Gi"
        cpu: "1000m"
      requests:
        memory: "1Gi"
        cpu: "500m"
    
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/.cache
    
  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
```

## 10. Conclusion

This comprehensive security assessment provides defense-in-depth controls for the web API implementation. Key priorities:

1. **Immediate Implementation**:
   - Secure file upload with validation
   - Authentication and authorization
   - Rate limiting
   - Security headers

2. **Critical Security Features**:
   - Input validation at every layer
   - Resource isolation for processing
   - Encrypted storage
   - Comprehensive monitoring

3. **Ongoing Security**:
   - Regular security audits
   - Dependency scanning
   - Penetration testing
   - Incident response planning

Remember: The web interface significantly increases attack surface. Every feature must be designed with security-first principles.