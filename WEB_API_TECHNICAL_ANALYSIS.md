# Web API Technical Requirements Analysis

## Executive Summary

Based on analysis of the current image processing pipeline, I provide technical recommendations for exposing the photo restoration CLI as a REST API with focus on production scalability, security, and performance.

## 1. Current Architecture Analysis ✅

### Existing Components
- **ImageProcessor**: Core single-image processing engine
- **BatchProcessor**: Multi-image processing with progress tracking  
- **ModelManager**: AI model loading and memory management
- **Config System**: Comprehensive configuration management
- **Security Layer**: Path validation and secure model loading implemented

### Processing Pipeline
```
Input → Validation → Model Loading → ESRGAN Enhancement → Face Enhancement → Output
```

### Key Characteristics
- **Memory-aware processing**: Adaptive tile sizing based on available memory
- **CPU-optimized**: Designed for CPU inference with thread control
- **Progress tracking**: Built-in callback system for progress reporting
- **Quality levels**: Fast/Balanced/Best with different tile sizes
- **Error handling**: Comprehensive with fallback mechanisms

---

## 2. REST API Architecture Design 🏗️

### Recommended Technology Stack

```python
# Core Framework
FastAPI          # High-performance async framework
Pydantic         # Data validation and serialization  
Celery          # Distributed task queue for async processing
Redis           # Message broker and result backend
SQLite/PostgreSQL # Job metadata and status tracking

# File Handling
aiofiles        # Async file operations
python-multipart # File upload handling

# Security & Monitoring  
python-jose[cryptography]  # JWT tokens
slowapi         # Rate limiting
prometheus-client          # Metrics collection
structlog       # Structured logging
```

### API Endpoint Structure

```python
# Single Image Processing
POST   /api/v1/enhance                    # Upload and process single image
GET    /api/v1/jobs/{job_id}             # Get job status
GET    /api/v1/jobs/{job_id}/result      # Download result
DELETE /api/v1/jobs/{job_id}             # Cancel/cleanup job

# Batch Processing
POST   /api/v1/enhance/batch             # Upload multiple images
GET    /api/v1/batch/{batch_id}          # Get batch status
GET    /api/v1/batch/{batch_id}/results  # Download batch results

# System Management
GET    /api/v1/health                    # Health check
GET    /api/v1/status                    # System resource status
GET    /api/v1/config                    # Get available configurations
POST   /api/v1/models/warmup             # Warm up models
```

---

## 3. File Upload/Download Handling 📁

### Upload Requirements

```python
# Maximum file constraints
MAX_FILE_SIZE = 20 * 1024 * 1024      # 20MB per file
MAX_BATCH_SIZE = 50                    # Max files per batch
MAX_CONCURRENT_UPLOADS = 10            # Per user limit

# Supported formats
SUPPORTED_FORMATS = [
    'image/jpeg', 'image/png', 'image/tiff', 
    'image/bmp', 'image/webp'
]

# Upload validation pipeline
async def validate_upload(file):
    # 1. Size validation
    # 2. Magic byte verification (security)
    # 3. Image dimension validation  
    # 4. Format validation
    # 5. Virus scanning (optional)
```

### Storage Architecture

```python
# Temporary file structure
/tmp/photo-restore-api/
├── uploads/
│   ├── {user_id}/
│   │   └── {job_id}/
│   │       ├── input.jpg
│   │       └── metadata.json
├── processing/
│   └── {job_id}/
│       ├── input.jpg
│       ├── output.jpg
│       └── progress.json
└── results/
    └── {job_id}/
        ├── enhanced.jpg
        ├── thumbnail.jpg
        └── metadata.json

# Cleanup policy
- Upload files: Deleted after 1 hour if not processed
- Processing files: Deleted after job completion
- Result files: Retained for 24 hours, then deleted
```

### Download Implementation

```python
# Streaming download for large files
@app.get("/api/v1/jobs/{job_id}/result")
async def download_result(job_id: str):
    return StreamingResponse(
        file_generator(result_path),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"attachment; filename=enhanced_{job_id}.jpg",
            "Cache-Control": "no-cache"
        }
    )

# Thumbnail preview
@app.get("/api/v1/jobs/{job_id}/preview")
async def get_preview(job_id: str):
    # Return 400x400 thumbnail for quick preview
    pass
```

---

## 4. Async Processing Architecture ⚡

### Task Queue Design

```python
# Celery task structure
@celery_app.task(bind=True)
def process_image_task(self, job_id: str, input_path: str, params: dict):
    """
    Async image processing task with progress updates
    """
    try:
        # Initialize progress tracking
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'stage': 'Loading models'}
        )
        
        # Load configuration and models
        config = Config.load()
        processor = ImageProcessor(config)
        
        # Define progress callback
        def progress_callback(percent: int):
            self.update_state(
                state='PROCESSING',
                meta={'progress': percent, 'stage': get_stage(percent)}
            )
        
        # Process image with progress tracking
        result = processor.process_image(
            input_path=input_path,
            output_path=get_output_path(job_id),
            progress_callback=progress_callback,
            **params
        )
        
        if result:
            return {
                'status': 'SUCCESS',
                'result_path': get_output_path(job_id),
                'metadata': extract_metadata(result)
            }
        else:
            raise Exception("Processing failed")
            
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'progress': 0}
        )
        raise
```

### Job Status Tracking

```python
# Job status database schema
class JobStatus(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str
    status: str  # QUEUED, PROCESSING, SUCCESS, FAILURE
    progress: int = 0
    stage: str = ""
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    
    # Processing parameters
    quality: str = "balanced"
    upscale: int = 4
    face_enhance: bool = True
    output_format: str = "jpg"

# Real-time updates via WebSocket
@app.websocket("/api/v1/jobs/{job_id}/status")
async def job_status_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    while True:
        status = await get_job_status(job_id)
        await websocket.send_json(status)
        if status['status'] in ['SUCCESS', 'FAILURE']:
            break
        await asyncio.sleep(1)
```

### Batch Processing Architecture

```python
# Batch job coordination
@celery_app.task
def process_batch_task(batch_id: str, job_ids: list[str]):
    """
    Coordinate multiple image processing jobs
    """
    # Use Celery group for parallel processing
    job_group = group(
        process_image_task.s(job_id, input_path, params)
        for job_id, input_path, params in get_batch_jobs(batch_id)
    )
    
    result = job_group.apply_async()
    
    # Monitor progress
    while not result.ready():
        progress = calculate_batch_progress(result)
        update_batch_status(batch_id, progress)
        time.sleep(2)
    
    return aggregate_batch_results(result.get())
```

---

## 5. Memory Management Strategy 🧠

### Request-Level Memory Management

```python
# Memory-aware request handling
class MemoryManager:
    def __init__(self):
        self.active_jobs = {}
        self.memory_threshold = 0.8  # 80% memory usage limit
        
    async def can_accept_job(self, estimated_memory_mb: int) -> bool:
        current_usage = psutil.virtual_memory().percent / 100
        estimated_usage = current_usage + (estimated_memory_mb / 1024)
        
        return estimated_usage < self.memory_threshold
    
    def estimate_memory_usage(self, image_dimensions: tuple, quality: str) -> int:
        """Estimate memory usage based on image size and quality"""
        width, height = image_dimensions
        pixels = width * height
        
        base_memory = {
            'fast': 800,    # MB
            'balanced': 1200,
            'best': 1800
        }
        
        # Scale by image size
        scale_factor = pixels / (1024 * 1024)  # Normalize to 1MP
        return int(base_memory[quality] * scale_factor)

# Request middleware for memory checking
@app.middleware("http")
async def memory_check_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/v1/enhance"):
        if not await memory_manager.can_accept_job(1200):  # Estimated memory
            raise HTTPException(
                status_code=503,
                detail="Server overloaded, please try again later"
            )
    
    response = await call_next(request)
    return response
```

### Model Loading Strategy

```python
# Singleton model manager for API
class APIModelManager:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self.config = Config.load()
            self.model_manager = ModelManager(self.config)
            self._warmup_models()
            self._models_loaded = True
    
    def _warmup_models(self):
        """Pre-load models on startup"""
        self.model_manager.load_esrgan_model(scale=4)
        self.model_manager.load_gfpgan_model()
    
    def get_processor(self) -> ImageProcessor:
        """Get configured image processor"""
        return ImageProcessor(self.config, self.model_manager)

# Memory cleanup after each job
def cleanup_job_memory(job_id: str):
    """Clean up memory after job completion"""
    import gc
    import torch
    
    # Clear temporary files
    cleanup_temp_files(job_id)
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Concurrent Request Handling

```python
# Rate limiting and concurrency control
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/enhance")
@limiter.limit("3/minute")  # Max 3 requests per minute per IP
async def enhance_image(request: Request, file: UploadFile):
    # Semaphore for concurrent processing
    async with processing_semaphore:  # Limit to N concurrent jobs
        return await process_single_image(file)

# Global settings
MAX_CONCURRENT_JOBS = 5
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
```

---

## 6. Integration Patterns with CLI Codebase 🔗

### Shared Core Architecture

```python
# photo_restore/
├── core/                    # Shared components
│   ├── processors/         # Image processing engines
│   ├── models/            # Model management
│   ├── utils/             # Utilities and config
│   └── security/          # Security patches
├── cli/                   # CLI-specific code
│   └── main.py           # Click CLI interface
├── api/                  # API-specific code
│   ├── main.py          # FastAPI application
│   ├── routers/         # API route handlers
│   ├── models/          # Pydantic models
│   ├── tasks/           # Celery tasks
│   └── middleware/      # API middleware
└── common/              # Common constants and types
```

### Configuration Abstraction

```python
# Unified configuration system
class APIConfig(Config):
    """Extended config for API usage"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__(config_dict)
        
        # API-specific settings
        self.api = APISettings()
        if config_dict and 'api' in config_dict:
            for key, value in config_dict['api'].items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)

@dataclass
class APISettings:
    """API-specific configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_workers: int = 4
    max_file_size: int = 20 * 1024 * 1024
    max_batch_size: int = 50
    result_retention_hours: int = 24
    enable_auth: bool = True
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])
    
    # Celery settings
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_routes: dict = field(default_factory=lambda: {
        'api.tasks.*': {'queue': 'processing'}
    })
```

### Service Abstraction Layer

```python
# Bridge between CLI and API
class ProcessingService:
    """
    Service layer abstracting image processing
    Used by both CLI and API interfaces
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = ImageProcessor(config)
        self.batch_processor = BatchProcessor(config)
    
    async def process_single_image(
        self,
        input_data: Union[str, bytes],  # File path or binary data
        output_path: Optional[str] = None,
        quality: str = 'balanced',
        upscale: int = 4,
        face_enhance: bool = True,
        output_format: str = 'jpg',
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Unified single image processing"""
        
        # Handle different input types
        if isinstance(input_data, bytes):
            # Save bytes to temp file for processing
            temp_path = await save_temp_file(input_data)
            input_path = temp_path
        else:
            input_path = input_data
        
        # Process using existing CLI logic
        success = self.processor.process_image(
            input_path=input_path,
            output_path=output_path,
            quality=quality,
            upscale=upscale,
            face_enhance=face_enhance,
            output_format=output_format,
            progress_callback=progress_callback
        )
        
        return ProcessingResult(
            success=success,
            output_path=output_path,
            metadata=self.processor.get_stats()
        )
    
    async def process_batch(
        self,
        input_files: list[Union[str, bytes]],
        output_dir: str,
        **kwargs
    ) -> BatchProcessingResult:
        """Unified batch processing"""
        # Implementation leveraging existing BatchProcessor
        pass

# Data models for API responses
class ProcessingResult:
    success: bool
    output_path: Optional[str]
    metadata: dict
    error_message: Optional[str] = None

class BatchProcessingResult:
    total_images: int
    processed_images: int
    failed_images: int
    results: list[ProcessingResult]
```

---

## 7. Implementation Roadmap 🛣️

### Phase 1: Core API (2-3 weeks)
1. **FastAPI application setup**
   - Basic project structure
   - Configuration management
   - Security middleware

2. **Single image endpoint**
   - File upload handling
   - Sync processing (blocking)
   - Response with result URL

3. **Basic monitoring**
   - Health checks
   - Resource monitoring
   - Basic logging

### Phase 2: Async Processing (2-3 weeks)
1. **Celery integration**
   - Task queue setup
   - Redis configuration
   - Job status tracking

2. **Async endpoints**
   - Job submission
   - Status polling
   - Result retrieval

3. **WebSocket support**
   - Real-time progress updates
   - Connection management

### Phase 3: Production Features (2-3 weeks)
1. **Batch processing**
   - Multi-file uploads
   - Batch job coordination
   - ZIP download results

2. **Advanced monitoring**
   - Prometheus metrics
   - Performance tracking
   - Error alerting

3. **Optimization**
   - Memory management
   - Model warming
   - Caching strategies

### Phase 4: Scale & Security (1-2 weeks)
1. **Authentication & authorization**
   - JWT tokens
   - Rate limiting
   - User quotas

2. **Production deployment**
   - Docker containers
   - Load balancer config
   - Database migration

---

## 8. Technical Considerations 🔧

### Security Requirements
- **Input validation**: All uploads validated for type, size, content
- **Path security**: Reuse existing SecurePathValidator  
- **Rate limiting**: Prevent API abuse
- **Authentication**: JWT-based user authentication
- **CORS**: Configurable cross-origin policies

### Performance Optimization
- **Model pre-loading**: Warm up models on startup
- **Connection pooling**: Database and Redis connections
- **Async I/O**: Non-blocking file operations
- **Caching**: Result caching for repeated requests
- **Compression**: Response compression for large images

### Monitoring & Observability
- **Health checks**: Deep health validation
- **Metrics collection**: Processing time, success rates, resource usage
- **Distributed tracing**: Request tracking across services
- **Log aggregation**: Structured logging with correlation IDs

### Error Handling
- **Circuit breaker**: Prevent cascade failures
- **Retry logic**: Automatic retry for transient failures
- **Graceful degradation**: Fallback to lower quality processing
- **Error reporting**: Detailed error information for debugging

---

## 9. Deployment Architecture 🚀

### Containerized Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:pass@postgres:5432/photoapi
    depends_on:
      - redis
      - postgres
    volumes:
      - ./models:/app/models
      - /tmp/photo-restore:/tmp/photo-restore

  worker:
    build: .
    command: celery -A api.tasks worker --loglevel=info --concurrency=2
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
      - /tmp/photo-restore:/tmp/photo-restore

  beat:
    build: .
    command: celery -A api.tasks beat --loglevel=info
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    
  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=photoapi
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
```

### Load Balancer Configuration

```nginx
upstream photo_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    client_max_body_size 25M;
    
    location /api/ {
        proxy_pass http://photo_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
    
    location /ws/ {
        proxy_pass http://photo_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 10. Cost & Resource Analysis 💰

### Hardware Requirements (per API server)
- **CPU**: 4+ cores (AI inference intensive)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 100GB for models and temp files
- **Network**: 1Gbps for file transfers

### Scaling Estimates
- **Single server capacity**: ~5-10 concurrent processing jobs
- **Processing time**: 30-60 seconds per 4MP image (4x upscale)
- **Memory per job**: 1-2GB peak usage
- **Storage per job**: 50-100MB temporary space

### Cost Factors
- **Compute cost**: High due to AI processing requirements
- **Storage cost**: Moderate for temporary file storage
- **Network cost**: Variable based on upload/download volume
- **Infrastructure**: Redis, PostgreSQL, load balancer

---

## Technical Feasibility: ✅ HIGH

The existing CLI codebase is well-architected for API integration with:
- **Clean separation** of processing logic
- **Comprehensive configuration** system  
- **Memory management** already implemented
- **Security patches** in place
- **Progress tracking** infrastructure ready

**Recommended approach**: Gradual migration with shared core components, starting with synchronous API and evolving to full async architecture.