"""
Anti-spoofing detection for face recognition system.
"""
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from loguru import logger

from ..core.config import settings


class LBPFeatureExtractor:
    """Local Binary Pattern feature extractor for anti-spoofing."""
    
    def __init__(self, radius: int = 1, n_points: int = 8):
        """
        Initialize LBP extractor.
        
        Args:
            radius: Radius of the LBP
            n_points: Number of points in the LBP
        """
        self.radius = radius
        self.n_points = n_points
    
    def extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            LBP histogram features
        """
        from skimage import feature
        
        # Calculate LBP
        lbp = feature.local_binary_pattern(
            image, self.n_points, self.radius, method='uniform'
        )
        
        # Calculate histogram
        n_bins = self.n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        
        return hist


class CNNAntiSpoofing(nn.Module):
    """CNN-based anti-spoofing detection model."""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        """
        Initialize CNN anti-spoofing model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes (real/fake)
        """
        super(CNNAntiSpoofing, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ColorSpaceAnalyzer:
    """Color space analysis for anti-spoofing."""
    
    def __init__(self):
        """Initialize color space analyzer."""
        pass
    
    def analyze_color_distribution(self, image: np.ndarray) -> dict:
        """
        Analyze color distribution in different color spaces.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary of color statistics
        """
        stats = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Calculate statistics for each channel
        for name, img in [('BGR', image), ('HSV', hsv), ('LAB', lab), ('YUV', yuv)]:
            for i, channel in enumerate(['ch0', 'ch1', 'ch2']):
                channel_data = img[:, :, i].flatten()
                stats[f'{name}_{channel}_mean'] = np.mean(channel_data)
                stats[f'{name}_{channel}_std'] = np.std(channel_data)
                stats[f'{name}_{channel}_skew'] = self._calculate_skewness(channel_data)
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)


class TextureAnalyzer:
    """Texture analysis for anti-spoofing."""
    
    def __init__(self):
        """Initialize texture analyzer."""
        pass
    
    def calculate_glcm_features(self, image: np.ndarray) -> dict:
        """
        Calculate GLCM (Gray-Level Co-occurrence Matrix) features.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary of texture features
        """
        from skimage.feature import graycomatrix, graycoprops
        
        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        features = {}
        
        for distance in distances:
            glcm = graycomatrix(
                image, distances=[distance], angles=np.radians(angles),
                levels=256, symmetric=True, normed=True
            )
            
            # Calculate properties
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                values = graycoprops(glcm, prop).flatten()
                features[f'{prop}_d{distance}_mean'] = np.mean(values)
                features[f'{prop}_d{distance}_std'] = np.std(values)
        
        return features


class AntiSpoofingDetector:
    """Complete anti-spoofing detection system."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize anti-spoofing detector.
        
        Args:
            model_path: Path to pretrained model
            device: Device to use
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize models and analyzers
        self.cnn_model = CNNAntiSpoofing()
        self.cnn_model.to(device)
        self.cnn_model.eval()
        
        self.lbp_extractor = LBPFeatureExtractor()
        self.color_analyzer = ColorSpaceAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        
        # Load pretrained weights if available
        if model_path:
            self._load_model(model_path)
        
        logger.info(f"Anti-spoofing detector initialized on {device}")
    
    def _load_model(self, model_path: str):
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded anti-spoofing model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
    
    def detect_spoofing(self, face_image: np.ndarray) -> Tuple[bool, float, dict]:
        """
        Detect if a face image is spoofed.
        
        Args:
            face_image: Input face image
            
        Returns:
            Tuple of (is_real, confidence, features)
        """
        features = {}
        
        # Ensure image is in correct format
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = face_image
        
        # CNN-based detection
        cnn_score = self._cnn_detection(face_image)
        features['cnn_score'] = cnn_score
        
        # LBP features
        lbp_features = self.lbp_extractor.extract_lbp(gray_image)
        features['lbp_uniformity'] = np.std(lbp_features)
        
        # Color space analysis
        if len(face_image.shape) == 3:
            color_stats = self.color_analyzer.analyze_color_distribution(face_image)
            features.update(color_stats)
        
        # Texture analysis
        texture_features = self.texture_analyzer.calculate_glcm_features(gray_image)
        features.update(texture_features)
        
        # Combine scores (simple weighted average)
        combined_score = (
            cnn_score * 0.6 +
            (1.0 - features['lbp_uniformity']) * 0.2 +
            self._calculate_texture_score(texture_features) * 0.2
        )
        
        is_real = combined_score > settings.anti_spoofing_threshold
        
        return is_real, combined_score, features
    
    def _cnn_detection(self, face_image: np.ndarray) -> float:
        """CNN-based spoofing detection."""
        # Preprocess image
        if len(face_image.shape) == 3:
            # Resize and normalize
            face_resized = cv2.resize(face_image, (224, 224))
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
            face_tensor = face_tensor / 255.0
        else:
            # Convert grayscale to RGB
            face_resized = cv2.resize(face_image, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
            face_tensor = face_tensor / 255.0
        
        face_batch = face_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.cnn_model(face_batch)
            probabilities = F.softmax(outputs, dim=1)
            real_probability = probabilities[0, 1].item()  # Class 1 is real
        
        return real_probability
    
    def _calculate_texture_score(self, texture_features: dict) -> float:
        """Calculate texture-based liveness score."""
        # Simple heuristic based on texture contrast
        contrast_features = [v for k, v in texture_features.items() if 'contrast' in k]
        if contrast_features:
            avg_contrast = np.mean(contrast_features)
            # Higher contrast typically indicates real faces
            return min(avg_contrast / 100.0, 1.0)
        return 0.5
    
    def train_mode(self):
        """Set model to training mode."""
        self.cnn_model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.cnn_model.eval()