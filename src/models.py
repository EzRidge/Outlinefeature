"""
Neural network models for roof feature detection and segmentation.
Implements hybrid architecture combining RID's segmentation with Roofline-Extraction's features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .config import MODEL_CONFIG

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction module."""
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        
        # Encoder blocks at different scales
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, feature_channels),
            ConvBlock(feature_channels, feature_channels)
        )
        
        self.enc2 = nn.Sequential(
            ConvBlock(feature_channels, feature_channels*2, stride=2),
            ConvBlock(feature_channels*2, feature_channels*2)
        )
        
        self.enc3 = nn.Sequential(
            ConvBlock(feature_channels*2, feature_channels*4, stride=2),
            ConvBlock(feature_channels*4, feature_channels*4)
        )
        
        # Bridge
        self.bridge = nn.Sequential(
            ConvBlock(feature_channels*4, feature_channels*8, stride=2),
            ConvBlock(feature_channels*8, feature_channels*8)
        )
        
        # Decoder blocks with skip connections
        self.dec3 = nn.Sequential(
            ConvBlock(feature_channels*12, feature_channels*4),
            ConvBlock(feature_channels*4, feature_channels*4)
        )
        
        self.dec2 = nn.Sequential(
            ConvBlock(feature_channels*6, feature_channels*2),
            ConvBlock(feature_channels*2, feature_channels*2)
        )
        
        self.dec1 = nn.Sequential(
            ConvBlock(feature_channels*3, feature_channels),
            ConvBlock(feature_channels, feature_channels)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bridge
        b = self.bridge(e3)
        
        # Decoder path with skip connections
        d3 = self.dec3(torch.cat([
            F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1))
        
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False),
            e2
        ], dim=1))
        
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False),
            e1
        ], dim=1))
        
        return d1, d2, d3, b

class HybridRoofDetector(nn.Module):
    """
    Hybrid model combining semantic segmentation with feature detection.
    Uses RID's segmentation approach with Roofline-Extraction's feature heads.
    """
    def __init__(self, config=MODEL_CONFIG):
        super().__init__()
        
        # Load backbone from segmentation-models-pytorch
        self.backbone = smp.Unet(
            encoder_name=config['backbone'],
            encoder_weights='imagenet',
            in_channels=config['input_channels'],
            classes=1  # Single channel output for each feature
        ).encoder
        
        # Multi-scale feature extraction
        feature_channels = config['feature_channels']
        self.feature_extractor = MultiScaleFeatureExtractor(
            in_channels=config['input_channels'],
            feature_channels=feature_channels
        )
        
        # Feature detection heads
        self.outline_head = nn.Sequential(
            ConvBlock(feature_channels, feature_channels//2),
            nn.Conv2d(feature_channels//2, 1, 1)
        )
        
        self.ridge_head = nn.Sequential(
            ConvBlock(feature_channels, feature_channels//2),
            nn.Conv2d(feature_channels//2, 1, 1)
        )
        
        self.hip_head = nn.Sequential(
            ConvBlock(feature_channels, feature_channels//2),
            nn.Conv2d(feature_channels//2, 1, 1)
        )
        
        self.valley_head = nn.Sequential(
            ConvBlock(feature_channels, feature_channels//2),
            nn.Conv2d(feature_channels//2, 1, 1)
        )
        
        # Angle prediction head for geometric constraints
        self.angle_head = nn.Sequential(
            ConvBlock(feature_channels, feature_channels//2),
            nn.Conv2d(feature_channels//2, 1, 1)  # Predict angle in radians
        )
    
    def forward(self, x):
        # Extract multi-scale features
        features_ms = self.feature_extractor(x)
        base_features = features_ms[0]  # Use first scale features for detection
        
        # Apply feature detection heads
        outline = torch.sigmoid(self.outline_head(base_features))
        ridge = torch.sigmoid(self.ridge_head(base_features))
        hip = torch.sigmoid(self.hip_head(base_features))
        valley = torch.sigmoid(self.valley_head(base_features))
        angles = self.angle_head(base_features)
        
        return {
            'outline': outline,
            'ridge': ridge,
            'hip': hip,
            'valley': valley,
            'angles': angles,
            'features': features_ms  # Return all features for potential auxiliary tasks
        }

def create_model(config=MODEL_CONFIG):
    """Create a HybridRoofDetector model."""
    model = HybridRoofDetector(config)
    return model
