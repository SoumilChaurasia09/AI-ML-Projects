import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceDetectorCNN(nn.Module):
    """
    Custom CNN for face detection.
    Predicts:
    1. is_face_logit (binary classification: presence of a face)
    2. bbox_coords (4 coordinates: [x_min, y_min, x_max, y_max] normalized in [0, 1])
    """
    def __init__(self):
        super(FaceDetectorCNN, self).__init__()
        
        # Convolutional Backbone (input size: 3 x 128 x 128)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 16 x 128 x 128
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 32 x 64 x 64 (after pool)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 x 32 x 32 (after pool)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# 128 x 16 x 16 (after pool)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)# 128 x 8 x 8 (after pool)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully Connected Layer for feature extraction
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        
        # Branch 1: Face Classification (binary classifier)
        self.class_out = nn.Linear(256, 1)
        
        # Branch 2: Bounding Box Regression (4 coords: [xmin, ymin, xmax, ymax])
        self.bbox_out = nn.Linear(256, 4)
        
    def forward(self, x):
        # x is [B, 3, 128, 128]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 64, 64]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 16, 16]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # [B, 128, 8, 8]
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # [B, 128, 4, 4]
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        
        # Classification output (logit)
        cls_logits = self.class_out(x)
        
        # Bounding box output (normalized coordinates in [0,1])
        # Use sigmoid to clamp predictions to valid [0, 1] range
        bbox = torch.sigmoid(self.bbox_out(x))
        
        return cls_logits, bbox


class FaceEmbeddingCNN(nn.Module):
    """
    Custom CNN for face embedding extraction (Siamese Network backbone).
    Takes a 64x64 cropped face and outputs a 128-dimensional L2-normalized embedding.
    """
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingCNN, self).__init__()
        
        # Input size: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32 x 64 x 64
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 x 32 x 32 (after pool)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# 128 x 16 x 16 (after pool)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# 256 x 8 x 8 (after pool)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, embedding_dim)
        
    def forward(self, x):
        # x is [B, 3, 64, 64]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 16, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 128, 8, 8]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # [B, 256, 4, 4]
        
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 normalize embeddings so cosine similarity corresponds to Euclidean distance
        x = F.normalize(x, p=2, dim=1)
        return x
