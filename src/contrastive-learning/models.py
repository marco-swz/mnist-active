"""
This file contains the model definitions required for contrastive learning
"""

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Architecture Definition (only convolutional part)
class ActiveMnistCnn(nn.Module):
    def __init__(self):
        super(ActiveMnistCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # All layers after the flattening are removed --> Representation learning!

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        # All layers after the flattening are removed --> Representation learning!
        return x


# Projection Head for Contrastive Learning
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# SimCLR Model Definition
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return x


# Definition of Classifier Model
class Classifier(nn.Module):
    def __init__(self, base_encoder, num_classes=10):
        super(Classifier, self).__init__()

        # Use a larger batch size for dummy input to avoid batch norm error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_encoder = base_encoder.to(device)

        # Use a larger batch size for dummy input to avoid batch norm error
        dummy_input = torch.zeros(2, 1, 28, 28, device=device)  # Batch size of 2
        with torch.no_grad():  # Ensure no gradients are computed
            output_size = base_encoder(dummy_input).view(-1).shape[0] // 2

        self.fc1 = nn.Linear(output_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.base_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the base encoder
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


