import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class HybridRoofModel(nn.Module):
    def __init__(self, num_segment_classes=12):  # 12 classes (0-11)
        """
        Initialize the hybrid roof model.
        Args:
            num_segment_classes: Number of segment classes (default: 12)
                - 0: Background
                - 1: Roof
                - 2: Ridge
                - 3: Valley
                - 4: Eave
                - 5: Dormer
                - 6: Chimney
                - 7: Window
                - 8: PV Module
                - 9: Shadow
                - 10: Tree
                - 11: Unknown
        """
        super().__init__()
        
        # Load pretrained ResNet50 backbone
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove final layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Common layers
        self.conv1x1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
        # Task-specific heads
        self.segment_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_segment_classes, kernel_size=1)
        )
        
        self.line_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1),  # 4 types: ridge, valley, eave, outline
            nn.Sigmoid()
        )
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        # New attention module for multi-scale feature enhancement
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.conv1x1(features)
        
        # Apply attention
        attention = self.attention(features)
        features = features * attention
        
        # Upsample features
        features = self.upsample(features)
        
        # Generate predictions for each task
        return {
            'segments': self.segment_head(features),
            'lines': self.line_head(features),
            'depth': self.depth_head(features)
        }

class RoofLoss(nn.Module):
    def __init__(self, weights: dict = None):
        super().__init__()
        
        # Initialize task weights
        self.task_weights = weights or {
            'segments': 1.0,
            'lines': 1.0,
            'depth': 1.0
        }
        
        # Class weights for handling imbalanced classes
        self.segment_weights = torch.ones(12)  # Adjust based on class distribution
        self.segment_weights[0] = 0.5  # Reduce weight for background
        
        # Initialize loss functions
        self.segment_loss = nn.CrossEntropyLoss(weight=self.segment_weights)
        self.line_loss = nn.BCEWithLogitsLoss()
        self.depth_loss = nn.L1Loss()
    
    def forward(self, predictions: dict, targets: dict) -> tuple:
        losses = {}
        
        # Segment loss (predictions: [B, C, H, W], targets: [B, H, W])
        losses['segments'] = self.segment_loss(
            predictions['segments'],
            targets['segments']
        ) * self.task_weights['segments']
        
        # Line loss (4 types of lines)
        losses['lines'] = self.line_loss(
            predictions['lines'],  # [B, 4, H, W]
            targets['lines']      # [B, 4, H, W]
        ) * self.task_weights['lines']
        
        # Depth loss
        losses['depth'] = self.depth_loss(
            predictions['depth'].squeeze(1),  # [B, H, W]
            targets['depth']  # [B, H, W]
        ) * self.task_weights['depth']
        
        return sum(losses.values()), losses

def create_model(num_classes=12, pretrained=True):
    """
    Factory function to create a new HybridRoofModel instance.
    
    Args:
        num_classes (int): Number of segmentation classes
        pretrained (bool): Whether to use pretrained backbone
        
    Returns:
        model: Initialized HybridRoofModel
        criterion: Loss function
    """
    model = HybridRoofModel(num_segment_classes=num_classes)
    criterion = RoofLoss()
    
    return model, criterion
