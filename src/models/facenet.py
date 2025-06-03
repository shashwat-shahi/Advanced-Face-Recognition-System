"""
FaceNet model implementation for face recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class InceptionResnetV1(nn.Module):
    """Inception ResNet V1 model for FaceNet."""
    
    def __init__(self, classify: bool = False, num_classes: Optional[int] = None, 
                 dropout_prob: float = 0.6, device: Optional[str] = None):
        """
        Initialize InceptionResnetV1.
        
        Args:
            classify: Whether to include classification layer
            num_classes: Number of output classes for classification
            dropout_prob: Dropout probability
            device: Device to use
        """
        super(InceptionResnetV1, self).__init__()
        
        self.classify = classify
        self.num_classes = num_classes
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        
        # Mixed 5b (Inception-A block)
        self.mixed_5b = Block35(scale=1.0, noReLU=False)
        
        # Mixed 6a (Reduction-A block)
        self.mixed_6a = Mixed_6a()
        
        # Mixed 6b (Inception-B block)
        self.mixed_6b = Block17(scale=1.0, noReLU=False)
        
        # Mixed 7a (Reduction-B block)
        self.mixed_7a = Mixed_7a()
        
        # Mixed 8a (Inception-C block)
        self.mixed_8a = Block8(scale=1.0, noReLU=False)
        self.mixed_8b = Block8(scale=1.0, noReLU=True)
        
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        
        if self.classify:
            self.logits = nn.Linear(512, self.num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        
        x = self.mixed_5b(x)
        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_7a(x)
        x = self.mixed_8a(x)
        x = self.mixed_8b(x)
        
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        
        if self.classify:
            x = self.logits(x)
        
        return x


class BasicConv2d(nn.Module):
    """Basic convolutional layer."""
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):
    """Inception-A block."""
    
    def __init__(self, scale=1.0, noReLU=False):
        super(Block35, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """Reduction-A block."""
    
    def __init__(self):
        super(Mixed_6a, self).__init__()
        
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )
        
        self.branch2 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    """Inception-B block."""
    
    def __init__(self, scale=1.0, noReLU=False):
        super(Block17, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    """Reduction-B block."""
    
    def __init__(self):
        super(Mixed_7a, self).__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        
        self.branch3 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    """Inception-C block."""
    
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class FaceNet:
    """FaceNet wrapper for face recognition."""
    
    def __init__(self, pretrained: bool = True, device: Optional[str] = None):
        """
        Initialize FaceNet model.
        
        Args:
            pretrained: Whether to load pretrained weights
            device: Device to use
        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.model = InceptionResnetV1(pretrained=pretrained, device=device)
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"FaceNet model loaded on {device}")
    
    def extract_embeddings(self, faces: torch.Tensor) -> np.ndarray:
        """
        Extract face embeddings.
        
        Args:
            faces: Batch of face images [B, C, H, W]
            
        Returns:
            Face embeddings [B, 512]
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