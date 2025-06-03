"""
Face detection using MTCNN and dlib.
"""
import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
from typing import List, Tuple, Optional
from loguru import logger

from ..core.config import settings


class FaceDetector:
    """Face detection using multiple methods."""
    
    def __init__(self, method: str = "mtcnn"):
        """
        Initialize face detector.
        
        Args:
            method: Detection method - 'mtcnn', 'dlib', or 'opencv'
        """
        self.method = method
        self.confidence_threshold = settings.face_detection_confidence
        
        if method == "mtcnn":
            self.detector = MTCNN()
        elif method == "dlib":
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(
                str(settings.model_path / "shape_predictor_68_face_landmarks.dat")
            )
        elif method == "opencv":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            raise ValueError(f"Unsupported detection method: {method}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face detection results with bounding boxes and confidence
        """
        if self.method == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.method == "dlib":
            return self._detect_dlib(image)
        elif self.method == "opencv":
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[dict]:
        """Detect faces using MTCNN."""
        results = self.detector.detect_faces(image)
        faces = []
        
        for result in results:
            if result['confidence'] >= self.confidence_threshold:
                box = result['box']
                faces.append({
                    'bbox': (box[0], box[1], box[0] + box[2], box[1] + box[3]),
                    'confidence': result['confidence'],
                    'landmarks': result.get('keypoints', {})
                })
        
        return faces
    
    def _detect_dlib(self, image: np.ndarray) -> List[dict]:
        """Detect faces using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        results = []
        for face in faces:
            bbox = (face.left(), face.top(), face.right(), face.bottom())
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            landmark_points = {}
            for i in range(68):
                point = landmarks.part(i)
                landmark_points[f'point_{i}'] = (point.x, point.y)
            
            results.append({
                'bbox': bbox,
                'confidence': 1.0,  # dlib doesn't provide confidence
                'landmarks': landmark_points
            })
        
        return results
    
    def _detect_opencv(self, image: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'landmarks': {}
            })
        
        return results
    
    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                    size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Extract and preprocess face from image.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            size: Output face size
            
        Returns:
            Preprocessed face image
        """
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            logger.warning("Empty face region detected")
            return np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Resize face
        face = cv2.resize(face, size)
        
        # Normalize
        face = face.astype(np.float32) / 255.0
        
        return face