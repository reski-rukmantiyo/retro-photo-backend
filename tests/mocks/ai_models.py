"""Mock AI model classes for testing without actual model downloads."""

import numpy as np
import cv2
from typing import Optional, Union
from unittest.mock import MagicMock


class MockRealESRGANer:
    """Mock Real-ESRGAN model for testing."""
    
    def __init__(self, scale: int = 4, model_path: str = "", device: str = "cpu", 
                 tile: int = 512, tile_pad: int = 32, pre_pad: int = 0, half: bool = False,
                 gpu_id=None, dni_weight=None):
        """Initialize mock Real-ESRGAN model."""
        self.scale = scale
        self.model_path = model_path
        self.device = device
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.gpu_id = gpu_id
        self.dni_weight = dni_weight
        
        # Track calls for testing
        self.enhance_calls = []
        self.last_input_shape = None
        
        # Simulate model loading time
        self._loaded = True
    
    def enhance(self, img: np.ndarray, outscale: Optional[int] = None) -> np.ndarray:
        """
        Mock enhance method that simulates upscaling.
        
        Args:
            img: Input image array (BGR format)
            outscale: Output scale factor
            
        Returns:
            Upscaled image array
        """
        if outscale is None:
            outscale = self.scale
        
        self.enhance_calls.append({
            'input_shape': img.shape,
            'outscale': outscale,
            'device': self.device
        })
        self.last_input_shape = img.shape
        
        # Simulate processing by upscaling with interpolation
        height, width = img.shape[:2]
        new_height, new_width = height * outscale, width * outscale
        
        # Use cubic interpolation for better quality simulation
        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Add slight noise reduction effect simulation
        upscaled = cv2.bilateralFilter(upscaled, 5, 75, 75)
        
        # Enhance contrast slightly to simulate AI enhancement
        upscaled = cv2.convertScaleAbs(upscaled, alpha=1.1, beta=5)
        
        return upscaled
    
    def get_call_count(self) -> int:
        """Get number of times enhance was called."""
        return len(self.enhance_calls)
    
    def get_last_call_info(self) -> Optional[dict]:
        """Get information about last enhance call."""
        return self.enhance_calls[-1] if self.enhance_calls else None
    
    def reset_call_history(self) -> None:
        """Reset call history for testing."""
        self.enhance_calls = []
        self.last_input_shape = None


class MockGFPGANer:
    """Mock GFPGAN model for testing face restoration."""
    
    def __init__(self, model_path: str = "", upscale: int = 1, arch: str = "clean",
                 channel_multiplier: int = 2, bg_upsampler=None, device: str = "cpu"):
        """Initialize mock GFPGAN model."""
        self.model_path = model_path
        self.upscale = upscale
        self.arch = arch
        self.channel_multiplier = channel_multiplier
        self.bg_upsampler = bg_upsampler
        self.device = device
        
        # Track calls for testing
        self.enhance_calls = []
        self.face_detection_calls = []
        
        # Simulate face detection threshold
        self.face_threshold = 0.5
        
        self._loaded = True
        
        # Add gfpgan attribute to simulate the actual GFPGAN model structure
        # This is needed for the JIT compilation tests in model_manager.py
        self.gfpgan = MagicMock()
        self.gfpgan.eval = MagicMock()
        self.gfpgan.parameters = MagicMock(return_value=[])  # Empty parameters for CPU optimization tests
    
    def enhance(self, img: np.ndarray, has_aligned: bool = False, 
                only_center_face: bool = False, paste_back: bool = True,
                weight: float = 0.5) -> tuple:
        """
        Mock enhance method that simulates face restoration.
        
        Args:
            img: Input image array (BGR format)
            has_aligned: Whether faces are pre-aligned
            only_center_face: Only process center face
            paste_back: Paste enhanced faces back to original image
            weight: Blending weight
            
        Returns:
            Tuple of (cropped_faces, restored_faces, restored_img)
        """
        self.enhance_calls.append({
            'input_shape': img.shape,
            'has_aligned': has_aligned,
            'only_center_face': only_center_face,
            'paste_back': paste_back,
            'weight': weight
        })
        
        # Simulate face detection
        faces = self._detect_faces(img)
        
        if not faces:
            # No faces detected, return original image
            return [], [], img.copy()
        
        # Simulate face restoration
        cropped_faces = []
        restored_faces = []
        
        enhanced_img = img.copy()
        
        for face_bbox in faces:
            x1, y1, x2, y2 = face_bbox
            
            # Extract face region
            face_region = img[y1:y2, x1:x2].copy()
            cropped_faces.append(face_region)
            
            # Simulate face enhancement
            enhanced_face = self._enhance_face(face_region)
            restored_faces.append(enhanced_face)
            
            if paste_back:
                # Resize enhanced face to original size and paste back
                face_resized = cv2.resize(enhanced_face, (x2-x1, y2-y1))
                
                # Blend with original
                alpha = weight
                blended = cv2.addWeighted(face_resized, alpha, face_region, 1-alpha, 0)
                enhanced_img[y1:y2, x1:x2] = blended
        
        return cropped_faces, restored_faces, enhanced_img
    
    def _detect_faces(self, img: np.ndarray) -> list:
        """
        Mock face detection.
        
        Returns:
            List of face bounding boxes [x1, y1, x2, y2]
        """
        height, width = img.shape[:2]
        
        self.face_detection_calls.append({
            'image_shape': img.shape,
            'threshold': self.face_threshold
        })
        
        # Simulate finding faces in common locations
        faces = []
        
        # Center face (most common case)
        center_x, center_y = width // 2, height // 2
        face_size = min(width, height) // 3
        
        if face_size > 64:  # Minimum face size
            x1 = max(0, center_x - face_size // 2)
            y1 = max(0, center_y - face_size // 2)
            x2 = min(width, center_x + face_size // 2)
            y2 = min(height, center_y + face_size // 2)
            faces.append([x1, y1, x2, y2])
        
        # Simulate additional faces based on image size
        if width > 800 and height > 600:
            # Left face
            left_x = width // 4
            if left_x > face_size:
                x1 = max(0, left_x - face_size // 3)
                y1 = max(0, center_y - face_size // 3)
                x2 = min(width, left_x + face_size // 3)
                y2 = min(height, center_y + face_size // 3)
                faces.append([x1, y1, x2, y2])
        
        return faces
    
    def _enhance_face(self, face: np.ndarray) -> np.ndarray:
        """
        Mock face enhancement processing.
        
        Args:
            face: Face region image
            
        Returns:
            Enhanced face image
        """
        # Simulate face enhancement effects
        enhanced = face.copy()
        
        # Skin smoothing simulation
        enhanced = cv2.bilateralFilter(enhanced, 15, 80, 80)
        
        # Slight sharpening for details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = cv2.addWeighted(face, 0.7, enhanced, 0.3, 0)
        
        # Color enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=3)
        
        # Simulate detail enhancement
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(enhanced, 0.95, edges_colored, 0.05, 0)
        
        return enhanced
    
    def get_call_count(self) -> int:
        """Get number of times enhance was called."""
        return len(self.enhance_calls)
    
    def get_face_detection_count(self) -> int:
        """Get number of face detection calls."""
        return len(self.face_detection_calls)
    
    def set_face_threshold(self, threshold: float) -> None:
        """Set face detection threshold for testing."""
        self.face_threshold = threshold
    
    def reset_call_history(self) -> None:
        """Reset call history for testing."""
        self.enhance_calls = []
        self.face_detection_calls = []


class MockBasicSRModel:
    """Mock BasicSR model for testing."""
    
    def __init__(self, model_path: str = "", device: str = "cpu"):
        """Initialize mock BasicSR model."""
        self.model_path = model_path
        self.device = device
        self.enhance_calls = []
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Mock enhance method."""
        self.enhance_calls.append({'input_shape': img.shape})
        
        # Simple upscaling for testing
        height, width = img.shape[:2]
        upscaled = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        return upscaled


class MockModelFactory:
    """Factory for creating mock models."""
    
    @staticmethod
    def create_esrgan_model(scale: int = 4, **kwargs) -> MockRealESRGANer:
        """Create mock Real-ESRGAN model."""
        return MockRealESRGANer(scale=scale, **kwargs)
    
    @staticmethod
    def create_gfpgan_model(**kwargs) -> MockGFPGANer:
        """Create mock GFPGAN model."""
        return MockGFPGANer(**kwargs)
    
    @staticmethod
    def create_failing_model() -> MagicMock:
        """Create a model that fails on enhance."""
        mock_model = MagicMock()
        mock_model.enhance.side_effect = RuntimeError("Model enhancement failed")
        return mock_model
    
    @staticmethod
    def create_slow_model(delay: float = 1.0) -> MagicMock:
        """Create a model that simulates slow processing."""
        import time
        
        def slow_enhance(img):
            time.sleep(delay)
            height, width = img.shape[:2]
            return cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        mock_model = MagicMock()
        mock_model.enhance.side_effect = slow_enhance
        return mock_model
    
    @staticmethod
    def create_memory_intensive_model() -> MagicMock:
        """Create a model that simulates high memory usage."""
        def memory_intensive_enhance(img):
            # Simulate memory allocation
            height, width = img.shape[:2]
            # Create temporary large arrays to simulate memory usage
            temp_arrays = []
            for i in range(5):
                temp_array = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
                temp_arrays.append(temp_array)
            
            # Return upscaled image
            result = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            return result
        
        mock_model = MagicMock()
        mock_model.enhance.side_effect = memory_intensive_enhance
        return mock_model


# Helper functions for test setup
def patch_realesrgan_import():
    """Patch Real-ESRGAN import to use mock."""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock module
    mock_realesrgan = MagicMock()
    mock_realesrgan.RealESRGANer = MockRealESRGANer
    
    # Replace in sys.modules
    sys.modules['realesrgan'] = mock_realesrgan
    return mock_realesrgan


def patch_gfpgan_import():
    """Patch GFPGAN import to use mock."""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock module
    mock_gfpgan = MagicMock()
    mock_gfpgan.GFPGANer = MockGFPGANer
    
    # Replace in sys.modules
    sys.modules['gfpgan'] = mock_gfpgan
    return mock_gfpgan


def patch_basicsr_import():
    """Patch BasicSR import to use mock."""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock module
    mock_basicsr = MagicMock()
    mock_basicsr.BasicSRModel = MockBasicSRModel
    
    # Replace in sys.modules
    sys.modules['basicsr'] = mock_basicsr
    return mock_basicsr