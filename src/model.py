import torch.nn as nn
import torch.nn.functional as F

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10 * 10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 30)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x