import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, (4, 4))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, (5, 5))
        self.conv3 = nn.Conv2d(3, 1, (3, 3))
        self.dropout = nn.Dropout(p=0.5)
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(29 * 29, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.linear_relu_stack(x)
        return x
