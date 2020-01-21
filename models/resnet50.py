import torch
import torch.nn as nn
import torchvision


class ResNet50(nn.Module):
    def __init__(self, grid_size, boxes_per_cell: int, categories: int, pretrained=True):
        super(ResNet50, self).__init__()
        self.grid_size = grid_size

        self.feature = torchvision.models.resnet50(pretrained=pretrained)
        self.feature.fc = nn.Identity()
        self.fc_1 = nn.Linear(2048, 4096)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(4096, grid_size[0] * grid_size[1] * (boxes_per_cell * 5 + categories))

    def forward(self, x):
        x = self.feature(x)
        x = torch.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        x = torch.sigmoid(x)
        x = x.contiguous().view(x.size(0), self.grid_size[0], self.grid_size[1], -1)
        return x
