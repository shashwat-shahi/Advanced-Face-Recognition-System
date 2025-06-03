"""
SphereFace model implementation for face recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from loguru import logger


class AngleLinear(nn.Module):
    """Angular margin linear layer for SphereFace."""
    
    def __init__(self, in_features: int, out_features: int, m: int = 4):
        """
        Initialize AngleLinear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features (number of classes)
            m: Angular margin multiplier
        """
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute coefficients for angle calculation
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]
    
    def forward(self, input_features: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Normalize input features and weights
        x = F.normalize(input_features, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        wf = torch.mm(x, w.t())
        
        if target is None:
            # Inference mode
            return wf
        
        # Training mode with angular margin
        numerator = self.mlambda[self.m](wf)
        
        # Create one-hot encoding for target
        one_hot = torch.zeros_like(wf)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        
        # Apply angular margin
        output = one_hot * numerator + (1.0 - one_hot) * wf
        
        return output


class SphereFaceResNet(nn.Module):
    """ResNet backbone for SphereFace."""
    
    def __init__(self, num_layers: int = 20, feature_dim: int = 512):
        """
        Initialize SphereFace ResNet.
        
        Args:
            num_layers: Number of layers in ResNet
            feature_dim: Dimension of output features
        """
        super(SphereFaceResNet, self).__init__()
        
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else:
            raise ValueError(f"Unsupported number of layers: {num_layers}")
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        
        # Feature layer
        self.fc = nn.Linear(512 * 7 * 7, feature_dim)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a layer with multiple blocks."""
        layers = []
        
        # First block with potential stride
        layers.append(SphereFaceBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(SphereFaceBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class SphereFaceBlock(nn.Module):
    """Basic block for SphereFace ResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize SphereFace block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
        """
        super(SphereFaceBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu2(out)
        
        return out


class SphereFace(nn.Module):
    """Complete SphereFace model."""
    
    def __init__(self, num_classes: int, feature_dim: int = 512, 
                 angular_margin: int = 4, num_layers: int = 20):
        """
        Initialize SphereFace model.
        
        Args:
            num_classes: Number of identity classes
            feature_dim: Dimension of feature embeddings
            angular_margin: Angular margin multiplier
            num_layers: Number of layers in ResNet backbone
        """
        super(SphereFace, self).__init__()
        
        self.feature_dim = feature_dim
        self.angular_margin = angular_margin
        
        # Backbone network
        self.backbone = SphereFaceResNet(num_layers, feature_dim)
        
        # Angular margin classifier
        self.classifier = AngleLinear(feature_dim, num_classes, angular_margin)
    
    def forward(self, x, target=None):
        """Forward pass."""
        # Extract features
        features = self.backbone(x)
        
        if target is not None:
            # Training mode
            output = self.classifier(features, target)
            return output, features
        else:
            # Inference mode - return normalized features
            return F.normalize(features, p=2, dim=1)


class SphereFaceModel:
    """SphereFace wrapper for face recognition."""
    
    def __init__(self, num_classes: int = 10000, feature_dim: int = 512,
                 angular_margin: int = 4, num_layers: int = 20,
                 pretrained: bool = False, device: Optional[str] = None):
        """
        Initialize SphereFace model.
        
        Args:
            num_classes: Number of identity classes
            feature_dim: Feature dimension
            angular_margin: Angular margin multiplier
            num_layers: Number of layers in backbone
            pretrained: Whether to load pretrained weights
            device: Device to use
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.model = SphereFace(
            num_classes=num_classes,
            feature_dim=feature_dim,
            angular_margin=angular_margin,
            num_layers=num_layers
        )
        self.model.to(device)
        self.model.eval()
        
        if pretrained:
            self._load_pretrained_weights()
        
        logger.info(f"SphereFace model loaded on {device}")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        # Placeholder for loading pretrained weights
        # In practice, you would load from a checkpoint file
        logger.info("Loading pretrained SphereFace weights...")
    
    def extract_embeddings(self, faces: torch.Tensor) -> np.ndarray:
        """
        Extract face embeddings.
        
        Args:
            faces: Batch of face images [B, C, H, W]
            
        Returns:
            Face embeddings [B, feature_dim]
        """
        with torch.no_grad():
            faces = faces.to(self.device)
            embeddings = self.model(faces)
            return embeddings.cpu().numpy()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.model.feature_dim
    
    def get_angular_margin(self) -> int:
        """Get angular margin."""
        return self.model.angular_margin