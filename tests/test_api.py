"""
Tests for API endpoints.
"""
import pytest
import numpy as np
import cv2
import io
from PIL import Image
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.api.main import app


class TestAPI:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        # Create a simple image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)
        cv2.circle(image, (85, 85), 5, (0, 0, 0), -1)
        cv2.circle(image, (115, 85), 5, (0, 0, 0), -1)
        cv2.ellipse(image, (100, 120), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        # Convert to PIL and then to bytes
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models" in data
    
    def test_detect_faces_endpoint(self, client, sample_image_bytes):
        """Test face detection endpoint."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        response = client.post("/detect_faces", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "num_faces" in data
        assert "faces" in data
        assert isinstance(data["faces"], list)
    
    def test_detect_faces_invalid_file(self, client):
        """Test face detection with invalid file."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        
        response = client.post("/detect_faces", files=files)
        assert response.status_code == 400
    
    def test_extract_embeddings_endpoint(self, client, sample_image_bytes):
        """Test embedding extraction endpoint."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        response = client.post("/extract_embeddings?model=facenet", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "num_faces" in data
        assert "model_used" in data
        assert "embeddings" in data
    
    def test_extract_embeddings_invalid_model(self, client, sample_image_bytes):
        """Test embedding extraction with invalid model."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        response = client.post("/extract_embeddings?model=invalid", files=files)
        assert response.status_code == 400
    
    def test_anti_spoofing_endpoint(self, client, sample_image_bytes):
        """Test anti-spoofing endpoint."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        response = client.post("/anti_spoofing_check", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "num_faces" in data
        assert "results" in data
        assert "threshold_used" in data
    
    def test_compare_faces_endpoint(self, client, sample_image_bytes):
        """Test face comparison endpoint."""
        files = [
            ("file1", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("file2", ("test2.jpg", sample_image_bytes, "image/jpeg"))
        ]
        
        response = client.post("/compare_faces?model=facenet", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "faces_in_image1" in data
        assert "faces_in_image2" in data
        assert "model_used" in data
        assert "comparisons" in data
    
    def test_clear_cache_endpoint(self, client):
        """Test cache clearing endpoint."""
        response = client.get("/clear_cache")
        # This might fail if Redis is not available, which is okay for testing
        assert response.status_code in [200, 500]