import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3),  # 3,448,448 -> 64,224,224
            nn.MaxPool2d(2, stride=2),  # -> 64,112,112

            nn.Conv2d(64, 192, 3, padding=1),  # -> 192,112,112
            nn.MaxPool2d(2, stride=2),  # -> 192,56,56

            nn.Conv2d(192, 128, 1),  # -> 128,56,56
            nn.Conv2d(128, 256, 3, padding=1),  # -> 256,56,56
            nn.Conv2d(256, 256, 1),  # -> 256,56,56
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,56,56
            nn.MaxPool2d(2, stride=2),  # -> 512,28,28

            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.Conv2d(512, 512, 1),  # -> 512,28,28
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 1024,28,28
            nn.MaxPool2d(2, stride=2),  # -> 1024,14,14

            nn.Conv2d(1024, 512, 1),  # -> 512,14,14
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 1024,14,14
            nn.Conv2d(1024, 512, 1, padding=1),  # -> 512,14,14
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 1024,14,14
            nn.Conv2d(1024, 1024, 3, padding=1),  # -> 1024,14,14
            nn.Conv2d(1024, 1024, 3, 2),  # -> 1024,7,7

            nn.Conv2d(1024, 1024, 3, padding=1),  # -> 1024,7,7
            nn.Conv2d(1024, 1024, 3, padding=1)  # -> 1024,7,7
        )

    def forward(self, x):
        y = self.layers(x)
        return y
