"""
Neural network models for roof feature detection and segmentation.
Implements multi-scale CNN architecture from the Roofline-Extraction paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN for roof feature detection and segmentation.
    Architecture based on the Roofline-Extraction paper with modifications
    for improved feature detection.
    """
    
    def __init__(self, in_channels=3, num_classes=4):
        """
        Initialize the MultiScaleCNN model.
        
        Args:
            in_channels (int): Number of input image channels (default: 3 for RGB)
            num_classes (int): Number of output classes (default: 4)
                - Class 0: Background
                - Class 1: Roof outline
                - Class 2: Ridge lines
                - Class 3: Valley lines
        """
        super(MultiScaleCNN, self).__init__()
        
        # Encoder blocks
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Multi-scale feature fusion
        self.msf = MultiScaleFeatureFusion(512)
        
        # Decoder blocks
        self.dec4 = self._make_decoder_block(512, 256)
        self.dec3 = self._make_decoder_block(256, 128)
        self.dec2 = self._make_decoder_block(128, 64)
        self.dec1 = self._make_decoder_block(64, 32)
        
        # Final convolution
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Create an encoder block with double convolution and max pooling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with upsampling and double convolution."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Multi-scale feature fusion
        msf = self.msf(e4)
        
        # Decoder path with skip connections
        d4 = self.dec4(msf) + e3
        d3 = self.dec3(d4) + e2
        d2 = self.dec2(d3) + e1
        d1 = self.dec1(d2)
        
        # Final convolution
        out = self.final_conv(d1)
        
        return out

class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module for combining features at different scales.
    """
    
    def __init__(self, channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        fused = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(fused)
