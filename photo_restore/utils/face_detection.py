"""Optimized face detection utilities for CPU inference."""

import logging
import time
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch

try:
    from facexlib.detection import RetinaFaceDetection
    FACEXLIB_AVAILABLE = True
except ImportError:
    FACEXLIB_AVAILABLE = False


class CPUFaceDetector:
    """CPU-optimized face detection for GFPGAN integration."""
    
    def __init__(self, 
                 model_type: str = 'retinaface_resnet50',
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize face detector.
        
        Args:
            model_type: Detection model type
            confidence_threshold: Face detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize detector
        self.detector = None
        self._init_detector(model_type)
    
    def _init_detector(self, model_type: str) -> None:
        """Initialize the appropriate face detector."""
        try:
            if FACEXLIB_AVAILABLE and model_type.startswith('retinaface'):
                # Use RetinaFace for better accuracy
                self.detector = RetinaFaceDetection(
                    model_name='retinaface_resnet50',
                    device='cpu'
                )
                self.detector_type = 'retinaface'
                self.logger.info("Initialized RetinaFace detector")
            else:
                # Fallback to OpenCV Haar cascades (very fast on CPU)
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.detector_type = 'opencv'
                self.logger.info("Initialized OpenCV Haar cascade detector")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize {model_type}: {str(e)}")
            # Ultimate fallback to OpenCV
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.detector_type = 'opencv'
            self.logger.info("Using OpenCV Haar cascade as fallback")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of face detection results with bounding boxes
        """
        start_time = time.time()
        
        try:
            if self.detector_type == 'retinaface':
                faces = self._detect_retinaface(image)
            else:
                faces = self._detect_opencv(image)
            
            detection_time = time.time() - start_time
            self.logger.debug(f"Face detection took {detection_time:.3f}s, found {len(faces)} faces")
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def _detect_retinaface(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFace."""
        # Convert BGR to RGB for RetinaFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            bboxes, landmarks = self.detector.detect_faces(
                rgb_image,
                confidence_threshold=self.confidence_threshold
            )
        
        faces = []
        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                face_info = {
                    'bbox': bbox[:4].astype(int),  # x1, y1, x2, y2
                    'confidence': float(bbox[4]),
                    'landmarks': landmarks[i] if landmarks is not None else None,
                    'detector': 'retinaface'
                }
                faces.append(face_info)
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV Haar cascades."""
        # Convert to grayscale for OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_rect = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in faces_rect:
            face_info = {
                'bbox': [x, y, x + w, y + h],  # x1, y1, x2, y2
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'landmarks': None,
                'detector': 'opencv'
            }
            faces.append(face_info)
        
        return faces
    
    def filter_faces_by_size(self, 
                           faces: List[Dict[str, Any]], 
                           min_face_size: int = 64,
                           max_face_size: int = 512) -> List[Dict[str, Any]]:
        """Filter faces by size constraints."""
        filtered_faces = []
        
        for face in faces:
            bbox = face['bbox']
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_size = max(face_width, face_height)
            
            if min_face_size <= face_size <= max_face_size:
                filtered_faces.append(face)
            else:
                self.logger.debug(f"Filtered out face with size {face_size}")
        
        return filtered_faces
    
    def get_largest_face(self, faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the largest detected face."""
        if not faces:
            return None
        
        largest_face = None
        largest_area = 0
        
        for face in faces:
            bbox = face['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > largest_area:
                largest_area = area
                largest_face = face
        
        return largest_face
    
    def crop_face_region(self, 
                        image: np.ndarray, 
                        face: Dict[str, Any], 
                        padding: float = 0.3) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Crop face region with padding.
        
        Args:
            image: Input image
            face: Face detection result
            padding: Padding ratio around face
            
        Returns:
            Cropped face image and crop coordinates
        """
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Add padding
        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)
        
        # Calculate crop coordinates
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(image.shape[1], x2 + pad_x)
        crop_y2 = min(image.shape[0], y2 + pad_y)
        
        # Crop image
        cropped_face = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        crop_info = {
            'x1': crop_x1,
            'y1': crop_y1,
            'x2': crop_x2,
            'y2': crop_y2
        }
        
        return cropped_face, crop_info


def create_face_detector(config: Dict[str, Any], 
                        logger: Optional[logging.Logger] = None) -> CPUFaceDetector:
    """Factory function to create face detector based on config."""
    detector_config = config.get('face_detection', {})
    
    return CPUFaceDetector(
        model_type=detector_config.get('model_type', 'retinaface_resnet50'),
        confidence_threshold=detector_config.get('threshold', 0.5),
        nms_threshold=detector_config.get('nms_threshold', 0.4),
        logger=logger
    )