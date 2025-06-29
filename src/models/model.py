# -----------------------------------------
# File: src/models/model.py
# -----------------------------------------

import torch
import torch.nn as nn

from src.utils.logger import setup_logger
from src.utils.config_loader import load_params  # ✅ For loading params.yaml

# Setup logger
logger = setup_logger("ModelLogger", log_file="logs/model.log")

# Load parameters from params.yaml
params = load_params()
model_params = params["model"]
training_params = params["training"]

num_classes = model_params["num_classes"]
use_residual = model_params.get("use_residual", True)
use_attention = model_params.get("use_attention", True)
dropout_rate = training_params.get("dropout_rate", 0.3)

logger.info(f"Model Configuration: use_residual={use_residual}, use_attention={use_attention}, dropout_rate={dropout_rate}, num_classes={num_classes}")

# -------------------------------
# Residual Block
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        try:
            identity = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            out = self.relu(out)
            return out
        except Exception as e:
            logger.error(f"Error in ResidualBlock forward pass: {e}")
            raise

# -------------------------------
# Attention Block
# -------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            b, c, _, _ = x.size()
            y = self.global_pool(x).view(b, c)
            y = self.relu(self.fc1(y))
            y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
            return x * y
        except Exception as e:
            logger.error(f"Error in AttentionBlock forward pass: {e}")
            raise

# -------------------------------
# CNN Model
# -------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block1 = ResidualBlock(32, 64, stride=2) if use_residual else nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.attn1 = AttentionBlock(64) if use_attention else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.res_block2 = ResidualBlock(64, 128, stride=2) if use_residual else nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attn2 = AttentionBlock(128) if use_attention else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        try:
            x = self.layer1(x)
            x = self.dropout1(self.attn1(self.res_block1(x)))
            x = self.dropout2(self.attn2(self.res_block2(x)))
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        except Exception as e:
            logger.error(f"Error in CNNModel forward pass: {e}")
            raise
