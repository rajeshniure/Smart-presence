import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetectionCNN(nn.Module):
    """Custom CNN for face detection (binary classification: face/no-face)."""

    def __init__(self, input_size: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        conv_out = input_size // 16
        self.fc1 = nn.Linear(256 * conv_out * conv_out, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FaceRecognitionCNN(nn.Module):
    """Custom CNN for face recognition (multi-class classification) with BatchNorm."""

    def __init__(self, num_classes: int, input_size: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.35)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def preprocess_image_to_tensor(img_np: np.ndarray, size: int = 64) -> torch.Tensor:
    from PIL import Image
    img = Image.fromarray(img_np)
    img = img.resize((size, size))
    tensor = torch.from_numpy(np.array(img)).float()
    tensor = tensor.permute(2, 0, 1) / 255.0
    return tensor


def sliding_window_detect(
    image_np: np.ndarray,
    model: FaceDetectionCNN,
    device: torch.device,
    window: int = 64,
    stride: int = 16,
    prob_threshold: float = 0.7,
) -> List[List[int]]:
    """Very simple sliding-window detector using the custom detection CNN.
    Returns list of [x, y, w, h] boxes.
    """
    h, w = image_np.shape[:2]
    boxes: List[List[int]] = []
    with torch.no_grad():
        for y in range(0, max(1, h - window), stride):
            for x in range(0, max(1, w - window), stride):
                crop = image_np[y : y + window, x : x + window]
                if crop.shape[0] != window or crop.shape[1] != window:
                    continue
                inp = preprocess_image_to_tensor(crop).unsqueeze(0).to(device)
                logits = model(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if float(probs[1]) >= prob_threshold:
                    boxes.append([x, y, window, window])
    return boxes


