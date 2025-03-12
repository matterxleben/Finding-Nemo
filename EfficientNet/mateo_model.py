import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from image_dataset import MultiLabelImageDataset

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()

        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, num_classes)  # Replace classifier
        )

    def forward(self, x):
        return self.model(x)