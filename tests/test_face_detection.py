"""
Tests for face detection functionality.
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.face_detection import FaceDetector


class TestFaceDetector:
    """Test face detection functionality."""
    
    @pytest.fixture
    def face_detector(self):
        """Create face detector instance."""
        return FaceDetector(method="mtcnn")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple synthetic image with face-like features
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face-like structure
        cv2.circle(image, (320, 240), 80, (255, 255, 255), -1)  # Face
        cv2.circle(image, (300, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (340, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(image, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        return image
    
    def test_detector_initialization(self):
        """Test detector initialization with different methods."""
        # Test MTCNN
        detector_mtcnn = FaceDetector(method="mtcnn")
        assert detector_mtcnn.method == "mtcnn"
        
        # Test dlib (if available)
        try:
            detector_dlib = FaceDetector(method="dlib")
            assert detector_dlib.method == "dlib"
        except:
            pytest.skip("dlib model files not available")
        
        # Test OpenCV
        detector_opencv = FaceDetector(method="opencv")
        assert detector_opencv.method == "opencv"
        
        # Test invalid method
        with pytest.raises(ValueError):
            FaceDetector(method="invalid")
    
    def test_face_detection(self, face_detector, sample_image):
        """Test face detection functionality."""
        faces = face_detector.detect_faces(sample_image)
        
        # Check return type
        assert isinstance(faces, list)
        
        # Each face should have required keys
        for face in faces:
            assert "bbox" in face
            assert "confidence" in face
            assert "landmarks" in face
            
            # Check bbox format
            bbox = face["bbox"]
            assert len(bbox) == 4
            assert all(isinstance(coord, (int, float)) for coord in bbox)
            
            # Check confidence range
            assert 0.0 <= face["confidence"] <= 1.0
    
    def test_face_extraction(self, face_detector, sample_image):
        """Test face extraction functionality."""
        # Define a test bounding box
        bbox = (240, 160, 400, 320)  # (x1, y1, x2, y2)
        
        # Extract face
        face = face_detector.extract_face(sample_image, bbox)
        
        # Check output
        assert isinstance(face, np.ndarray)
        assert face.shape == (224, 224, 3)  # Default output size
        assert face.dtype == np.float32
        assert 0.0 <= face.max() <= 1.0  # Should be normalized
    
    def test_face_extraction_custom_size(self, face_detector, sample_image):
        """Test face extraction with custom size."""
        bbox = (240, 160, 400, 320)
        custom_size = (128, 128)
        
        face = face_detector.extract_face(sample_image, bbox, size=custom_size)
        
        assert face.shape == (128, 128, 3)
    
    def test_empty_image(self, face_detector):
        """Test detection on empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = face_detector.detect_faces(empty_image)
        
        # Should return empty list for empty image
        assert isinstance(faces, list)
    
    def test_invalid_bbox(self, face_detector, sample_image):
        """Test face extraction with invalid bbox."""
        # Invalid bbox (out of bounds)
        invalid_bbox = (1000, 1000, 1100, 1100)
        
        face = face_detector.extract_face(sample_image, invalid_bbox)
        
        # Should return zeros for invalid bbox
        assert isinstance(face, np.ndarray)
        assert face.shape == (224, 224, 3)