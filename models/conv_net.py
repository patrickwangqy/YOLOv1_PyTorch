import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, cell_size, box_num, class_num, alpha=0.1):
        super().__init__()
        self.cell_size = cell_size
        self.box_num = box_num
        self.class_num = class_num
        self.cell_num = self.cell_size[0] * self.cell_size[1]
        self.feature_size = self.box_num * 5 + self.class_num
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3),  # 3,448,448 -> 64,224,224
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2, stride=2),  # -> 64,112,112

            nn.Conv2d(64, 192, 3, padding=1),  # -> 192,112,112
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2, stride=2),  # -> 192,56,56

            nn.Conv2d(192, 128, 1),  # -> 128,56,56
            nn.LeakyReLU(alpha),
            nn.Conv2d(128, 256, 3, padding=1),  # -> 256,56,56
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 256, 1),  # -> 256,56,56
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,56,56
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2, stride=2),  # -> 512,28,28

            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 256, 1),  # -> 256,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 1),  # -> 512,28,28
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 1024,28,28
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2, stride=2),  # -> 1024,14,14

            nn.Conv2d(1024, 512, 1),  # -> 512,14,14
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 1024,14,14
            nn.LeakyReLU(alpha),
            nn.Conv2d(1024, 512, 1, padding=1),  # -> 512,14,14
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 1024,14,14
            nn.LeakyReLU(alpha),
            nn.Conv2d(1024, 1024, 3, padding=1),  # -> 1024,14,14
            nn.LeakyReLU(alpha),
            nn.Conv2d(1024, 1024, 3, 2),  # -> 1024,7,7
            nn.LeakyReLU(alpha),

            nn.Conv2d(1024, 1024, 3, padding=1),  # -> 1024,7,7
            nn.LeakyReLU(alpha),
            nn.Conv2d(1024, 1024, 3, padding=1),  # -> 1024,7,7
            nn.LeakyReLU(alpha)
        )
        self.out = nn.Sequential(
            nn.Linear(1024 * self.cell_size[0] * self.cell_size[1], 4096),
            nn.LeakyReLU(alpha),
            nn.Linear(4096, cell_size[0] * cell_size[1] * self.feature_size)
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.cell_size[0], self.cell_size[1], self.feature_size)
        return x
