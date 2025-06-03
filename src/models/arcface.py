"""
ArcFace model implementation for face recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from loguru import logger


class ArcMarginProduct(nn.Module):
    """Implementation of ArcFace margin product."""
    
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, 
                 m: float = 0.50, easy_margin: bool = False):
        """
        Initialize ArcMarginProduct.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features (number of classes)
            s: Scale factor
            m: Margin
            easy_margin: Whether to use easy margin
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input_features: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Normalize features and weights
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate phi
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ArcFaceResNet(nn.Module):
    """ResNet backbone for ArcFace."""
    
    def __init__(self, block_layers: list = [2, 2, 2, 2], num_features: int = 512):
        """
        Initialize ArcFace ResNet.
        
        Args:
            block_layers: Number of blocks in each layer
            num_features: Number of output features
        """
        super(ArcFaceResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(ResNetBlock, 64, block_layers[0], stride=1)
        self.layer2 = self._make_layer(ResNetBlock, 128, block_layers[1], stride=2)
        self.layer3 = self._make_layer(ResNetBlock, 256, block_layers[2], stride=2)
        self.layer4 = self._make_layer(ResNetBlock, 512, block_layers[3], stride=2)
        
        # Feature layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_features)
        self.bn_fc = nn.BatchNorm1d(num_features)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)
        
        return x


class ArcFace(nn.Module):
    """Complete ArcFace model."""
    
    def __init__(self, num_classes: int, feature_dim: int = 512, 
                 s: float = 30.0, m: float = 0.50):
        """
        Initialize ArcFace model.
        
        Args:
            num_classes: Number of identity classes
            feature_dim: Dimension of feature embeddings
            s: Scale factor
            m: Margin
        """
        super(ArcFace, self).__init__()
        
        self.backbone = ArcFaceResNet(num_features=feature_dim)
        self.margin = ArcMarginProduct(feature_dim, num_classes, s, m)
        
    def forward(self, x, labels=None):
        """Forward pass."""
        features = self.backbone(x)
        
        if labels is not None:
            # Training mode
            output = self.margin(features, labels)
            return output, features
        else:
            # Inference mode
            return F.normalize(features, p=2, dim=1)


class ArcFaceModel:
    """ArcFace wrapper for face recognition."""
    
    def __init__(self, num_classes: int = 10000, feature_dim: int = 512,
                 pretrained: bool = False, device: Optional[str] = None):
        """
        Initialize ArcFace model.
        
        Args:
            num_classes: Number of identity classes
            feature_dim: Feature dimension
            pretrained: Whether to load pretrained weights
            device: Device to use
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.model = ArcFace(num_classes, feature_dim)
        self.model.to(device)
        self.model.eval()
        
        if pretrained:
            self._load_pretrained_weights()
        
        logger.info(f"ArcFace model loaded on {device}")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        # Placeholder for loading pretrained weights
        # In practice, you would load from a checkpoint file
        logger.info("Loading pretrained ArcFace weights...")
    
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