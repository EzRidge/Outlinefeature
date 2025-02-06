import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class HybridRoofModel(nn.Module):
    def __init__(self, num_segment_classes=9):  # 9 classes (0-8)
        """
        Initialize the hybrid roof model.
        Args:
            num_segment_classes: Number of segment classes (default: 9)
                - 0: Background
                - 1: PV module
                - 2: Dormer
                - 3: Window
                - 4: Ladder
                - 5: Chimney
                - 6: Shadow
                - 7: Tree
                - 8: Unknown
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
        
        self.superstructure_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.line_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1),  # 3 types of lines
            nn.Sigmoid()
        )
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.conv1x1(features)
        features = self.upsample(features)
        
        # Generate predictions for each task
        return {
            'segments': self.segment_head(features),
            'superstructures': self.superstructure_head(features),
            'lines': self.line_head(features),
            'depth': self.depth_head(features)
        }

class RoofLoss(nn.Module):
    def __init__(self, weights: dict = None):
        super().__init__()
        
        # Initialize task weights
        self.task_weights = weights or {
            'segments': 1.0,
            'superstructures': 1.0,
            'lines': 1.0,
            'depth': 1.0
        }
        
        # Initialize loss functions
        # For segments, we don't use class weights initially
        # This avoids the "weight tensor should be defined either for all or no classes" error
        self.segment_loss = nn.CrossEntropyLoss()
        self.superstructure_loss = nn.BCEWithLogitsLoss()
        self.line_loss = nn.BCEWithLogitsLoss()
        self.depth_loss = nn.L1Loss()
    
    def forward(self, predictions: dict, targets: dict) -> tuple:
        losses = {}
        
        # Segment loss (predictions: [B, C, H, W], targets: [B, H, W])
        losses['segments'] = self.segment_loss(
            predictions['segments'],
            targets['segments']
        ) * self.task_weights['segments']
        
        # Superstructure loss
        losses['superstructures'] = self.superstructure_loss(
            predictions['superstructures'].squeeze(1),  # [B, H, W]
            targets['superstructures']  # [B, H, W]
        ) * self.task_weights['superstructures']
        
        # Line loss
        losses['lines'] = self.line_loss(
            predictions['lines'],  # [B, 3, H, W]
            targets['lines']      # [B, 3, H, W]
        ) * self.task_weights['lines']
        
        # Depth loss
        losses['depth'] = self.depth_loss(
            predictions['depth'].squeeze(1),  # [B, H, W]
            targets['depth']  # [B, H, W]
        ) * self.task_weights['depth']
        
        return sum(losses.values()), losses
