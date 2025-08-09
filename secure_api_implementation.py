"""
Secure Web API Implementation for Photo Restoration Service
Implements all security controls from the security assessment
"""

import os
import uuid
import tempfile
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import aiofiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import JWTError, jwt
import redis
from celery import Celery
import structlog

# Import existing security modules
from security_patches.secure_image_validator import SecureImageValidator, ImageSecurityError
from security_patches.path_validator import SecurePathValidator, PathSecurityError
from security_patches.secure_model_loader import SecureModelLoader

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ===== CONFIGURATION =====

class SecurityConfig:
    """Security configuration with environment variables"""
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-this-in-production")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    MAX_IMAGE_DIMENSIONS = (8000, 8000)
    MIN_IMAGE_DIMENSIONS = (100, 100)
    
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
    ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost").split(",")
    
    UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/secure_uploads")
    PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/tmp/secure_processed")
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = "100 per hour"
    UPLOAD_RATE_LIMIT = "10 per hour"
    

# ===== SECURITY MODELS =====

class JobStatus(str):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingOptions(BaseModel):
    """Validated processing options"""
    quality: str = Field(default="balanced", regex="^(fast|balanced|best)$")
    face_enhance: bool = Field(default=True)
    upscale_factor: int = Field(default=2, ge=1, le=4)
    
    @validator('upscale_factor')
    def validate_upscale(cls, v):
        if v not in [1, 2, 4]:
            raise ValueError("Upscale factor must be 1, 2, or 4")
        return v


class JobResponse(BaseModel):
    """Job response model"""
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    options: ProcessingOptions
    result_url: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# ===== SECURITY MIDDLEWARE =====

class SecurityHeadersMiddleware:
    """Add comprehensive security headers"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Security headers
                    security_headers = {
                        b"x-content-type-options": b"nosniff",
                        b"x-frame-options": b"DENY",
                        b"x-xss-protection": b"1; mode=block",
                        b"referrer-policy": b"strict-origin-when-cross-origin",
                        b"permissions-policy": b"geolocation=(), microphone=(), camera=()",
                        b"content-security-policy": (
                            b"default-src 'self'; "
                            b"script-src 'self'; "
                            b"style-src 'self' 'unsafe-inline'; "
                            b"img-src 'self' data: https:; "
                            b"font-src 'self'; "
                            b"connect-src 'self'; "
                            b"media-src 'none'; "
                            b"object-src 'none'; "
                            b"frame-ancestors 'none'; "
                            b"base-uri 'self'; "
                            b"form-action 'self'"
                        )
                    }
                    
                    # Add HSTS for HTTPS
                    if scope.get("scheme") == "https":
                        security_headers[b"strict-transport-security"] = (
                            b"max-age=31536000; includeSubDomains; preload"
                        )
                    
                    # Update headers
                    for name, value in security_headers.items():
                        headers[name] = value
                    
                    # Convert back to list
                    message["headers"] = [(k, v) for k, v in headers.items()]
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# ===== AUTHENTICATION =====

class AuthHandler:
    """JWT-based authentication handler"""
    
    def __init__(self):
        self.secret = SecurityConfig.JWT_SECRET_KEY
        self.algorithm = SecurityConfig.JWT_ALGORITHM
        self.redis_client = redis.from_url(SecurityConfig.REDIS_URL, decode_responses=True)
    
    def create_access_token(self, user_id: str, scopes: List[str] = None) -> str:
        """Create JWT access token"""
        payload = {
            "sub": user_id,
            "exp": datetime.utcnow() + timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": str(uuid.uuid4()),
            "scopes": scopes or []
        }
        
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> dict:
        """Verify JWT token"""
        token = credentials.credentials
        
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if self.redis_client.exists(f"revoked_token:{jti}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
            
        except JWTError as e:
            logger.warning("jwt_verification_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )


# ===== FILE UPLOAD SECURITY =====

class SecureFileHandler:
    """Secure file upload and validation handler"""
    
    def __init__(self):
        self.image_validator = SecureImageValidator(
            max_file_size=SecurityConfig.MAX_FILE_SIZE,
            max_dimensions=SecurityConfig.MAX_IMAGE_DIMENSIONS,
            min_dimensions=SecurityConfig.MIN_IMAGE_DIMENSIONS
        )
        self.path_validator = SecurePathValidator([
            SecurityConfig.UPLOAD_DIR,
            SecurityConfig.PROCESSED_DIR
        ])
        
        # Create secure directories
        os.makedirs(SecurityConfig.UPLOAD_DIR, mode=0o750, exist_ok=True)
        os.makedirs(SecurityConfig.PROCESSED_DIR, mode=0o750, exist_ok=True)
    
    async def handle_upload(
        self, 
        file: UploadFile, 
        user_id: str,
        request_id: str
    ) -> Dict[str, Any]:
        """Handle file upload with comprehensive security"""
        
        # Generate secure filename
        file_id = str(uuid.uuid4())
        extension = self._get_safe_extension(file.filename)
        secure_filename = f"{user_id}_{request_id}_{file_id}{extension}"
        
        # Create user-specific directory
        user_dir = os.path.join(SecurityConfig.UPLOAD_DIR, user_id)
        os.makedirs(user_dir, mode=0o750, exist_ok=True)
        
        # Quarantine path for initial validation
        quarantine_path = os.path.join(user_dir, f"quarantine_{secure_filename}")
        final_path = os.path.join(user_dir, secure_filename)
        
        try:
            # Stream file to disk with size limit
            bytes_written = 0
            async with aiofiles.open(quarantine_path, 'wb') as f:
                while chunk := await file.read(8192):
                    bytes_written += len(chunk)
                    if bytes_written > SecurityConfig.MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File size exceeds {SecurityConfig.MAX_FILE_SIZE} bytes"
                        )
                    await f.write(chunk)
            
            # Validate file
            validation_result = self.image_validator.validate_image(quarantine_path)
            
            # Move to final location if valid
            os.rename(quarantine_path, final_path)
            
            logger.info(
                "file_upload_successful",
                user_id=user_id,
                file_id=file_id,
                size=bytes_written,
                format=validation_result.get('format')
            )
            
            return {
                "file_id": file_id,
                "path": final_path,
                "size": bytes_written,
                "validation": validation_result
            }
            
        except ImageSecurityError as e:
            # Clean up and log security issue
            if os.path.exists(quarantine_path):
                os.remove(quarantine_path)
            
            logger.warning(
                "file_validation_failed",
                user_id=user_id,
                error=str(e),
                filename=file.filename
            )
            
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {str(e)}"
            )
        except Exception as e:
            # Clean up on any error
            if os.path.exists(quarantine_path):
                os.remove(quarantine_path)
            
            logger.error(
                "file_upload_error",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _get_safe_extension(self, filename: str) -> str:
        """Extract and validate file extension"""
        if not filename:
            return ".jpg"
        
        parts = filename.rsplit('.', 1)
        if len(parts) != 2:
            return ".jpg"
        
        ext = f".{parts[1].lower()}"
        allowed = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        
        return ext if ext in allowed else ".jpg"


# ===== JOB MANAGEMENT =====

class JobManager:
    """Secure job management with Redis backend"""
    
    def __init__(self):
        self.redis_client = redis.from_url(SecurityConfig.REDIS_URL, decode_responses=True)
        self.file_handler = SecureFileHandler()
    
    async def create_job(
        self, 
        user_id: str, 
        file_path: str, 
        options: ProcessingOptions
    ) -> str:
        """Create processing job"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "file_path": file_path,
            "status": JobStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "options": options.dict()
        }
        
        # Store in Redis with expiration
        self.redis_client.hset(
            f"job:{job_id}",
            mapping={k: str(v) if not isinstance(v, dict) else str(v) for k, v in job_data.items()}
        )
        self.redis_client.expire(f"job:{job_id}", 86400)  # 24 hours
        
        # Add to user's job list
        self.redis_client.lpush(f"user_jobs:{user_id}", job_id)
        self.redis_client.ltrim(f"user_jobs:{user_id}", 0, 99)  # Keep last 100 jobs
        
        return job_id
    
    async def get_job(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get job details with authorization check"""
        job_data = self.redis_client.hgetall(f"job:{job_id}")
        
        if not job_data:
            return None
        
        # Authorization check
        if job_data.get("user_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job"
            )
        
        return job_data
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: JobStatus, 
        result_path: Optional[str] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Update job status"""
        updates = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if result_path:
            updates["result_path"] = result_path
        if error:
            updates["error"] = error
        if metrics:
            updates["metrics"] = str(metrics)
        
        self.redis_client.hset(
            f"job:{job_id}",
            mapping={k: str(v) for k, v in updates.items()}
        )


# ===== API APPLICATION =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("api_startup", version="1.0.0")
    
    # Initialize model loader
    app.state.model_loader = SecureModelLoader("/models")
    
    yield
    
    # Shutdown
    logger.info("api_shutdown")


# Create FastAPI app
app = FastAPI(
    title="Secure Photo Restoration API",
    version="1.0.0",
    docs_url=None,  # Disable in production
    redoc_url=None,  # Disable in production
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SecurityConfig.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=SecurityConfig.ALLOWED_HOSTS
)

# Initialize components
auth_handler = AuthHandler()
job_manager = JobManager()
file_handler = SecureFileHandler()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ===== API ENDPOINTS =====

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/v1/auth/token")
@limiter.limit("5 per minute")
async def login(request: Request, username: str, password: str):
    """Login endpoint (simplified for demo)"""
    # In production, verify against database with hashed passwords
    if username == "demo" and password == "secure_password":
        token = auth_handler.create_access_token(
            user_id="demo_user",
            scopes=["enhance:create", "enhance:read"]
        )
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@app.post("/api/v1/enhance", response_model=JobResponse)
@limiter.limit(SecurityConfig.UPLOAD_RATE_LIMIT)
async def enhance_image(
    request: Request,
    file: UploadFile = File(...),
    options: ProcessingOptions = ProcessingOptions(),
    token_data: dict = Depends(auth_handler.verify_token)
):
    """Upload and enhance a single image"""
    user_id = token_data["sub"]
    request_id = str(uuid.uuid4())
    
    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}"
        )
    
    try:
        # Handle file upload
        upload_result = await file_handler.handle_upload(file, user_id, request_id)
        
        # Create processing job
        job_id = await job_manager.create_job(
            user_id=user_id,
            file_path=upload_result["path"],
            options=options
        )
        
        # Queue for processing (would use Celery in production)
        # celery_app.send_task("process_image", args=[job_id])
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            options=options
        )
        
    except Exception as e:
        logger.error(
            "enhance_image_error",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process image"
        )


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
@limiter.limit(SecurityConfig.DEFAULT_RATE_LIMIT)
async def get_job_status(
    request: Request,
    job_id: str,
    token_data: dict = Depends(auth_handler.verify_token)
):
    """Get job status"""
    user_id = token_data["sub"]
    
    # Validate job_id format
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID format"
        )
    
    job_data = await job_manager.get_job(job_id, user_id)
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    return JobResponse(
        job_id=job_id,
        status=job_data["status"],
        created_at=datetime.fromisoformat(job_data["created_at"]),
        updated_at=datetime.fromisoformat(job_data["updated_at"]),
        options=ProcessingOptions(**eval(job_data["options"])),
        result_url=job_data.get("result_url"),
        error=job_data.get("error"),
        metrics=eval(job_data["metrics"]) if job_data.get("metrics") else None
    )


@app.get("/api/v1/jobs/{job_id}/result")
@limiter.limit(SecurityConfig.DEFAULT_RATE_LIMIT)
async def download_result(
    request: Request,
    job_id: str,
    token_data: dict = Depends(auth_handler.verify_token)
):
    """Download processed image"""
    user_id = token_data["sub"]
    
    job_data = await job_manager.get_job(job_id, user_id)
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    if job_data["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job_data['status']}, not completed"
        )
    
    result_path = job_data.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(
            status_code=404,
            detail="Result file not found"
        )
    
    # Validate path is within allowed directory
    try:
        file_handler.path_validator.validate_path(result_path)
    except PathSecurityError:
        logger.error(
            "path_traversal_attempt",
            user_id=user_id,
            job_id=job_id,
            path=result_path
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    return FileResponse(
        result_path,
        media_type="image/jpeg",
        filename=f"enhanced_{job_id}.jpg",
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff"
        }
    )


@app.delete("/api/v1/jobs/{job_id}")
@limiter.limit(SecurityConfig.DEFAULT_RATE_LIMIT)
async def cancel_job(
    request: Request,
    job_id: str,
    token_data: dict = Depends(auth_handler.verify_token)
):
    """Cancel a processing job"""
    user_id = token_data["sub"]
    
    job_data = await job_manager.get_job(job_id, user_id)
    
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    
    if job_data["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status {job_data['status']}"
        )
    
    # Update job status
    await job_manager.update_job_status(job_id, JobStatus.CANCELLED)
    
    # Clean up files
    for key in ["file_path", "result_path"]:
        if path := job_data.get(key):
            try:
                os.remove(path)
            except:
                pass
    
    return {"message": "Job cancelled successfully"}


# ===== ERROR HANDLERS =====

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "secure_api_implementation:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )