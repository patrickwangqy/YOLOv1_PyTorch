import torch
import torch.nn as nn
import torchvision


class ResNet50(nn.Module):
    def __init__(self, grid_size, boxes_per_cell: int, categories: int, pretrained=True):
        super(ResNet50, self).__init__()
        self.grid_size = grid_size

        self.feature = torchvision.models.resnet50(pretrained=pretrained)
        self.feature.avgpool = nn.Sequential()
        self.feature.fc = nn.Sequential()
        self.conv = nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc_1 = nn.Linear(512*grid_size[0]*grid_size[1], 4096)
        self.relu_2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(4096, grid_size[0]*grid_size[1]*(boxes_per_cell*5+categories))

    def forward(self, x):
        x = self.feature.conv1(x)
        x = self.feature.bn1(x)
        x = self.feature.relu(x)
        x = self.feature.maxpool(x)
        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        x = self.feature.layer4(x)
        x = self.relu_1(self.bn(self.conv(x)))
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.relu_2(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = torch.sigmoid(x)
        x = x.contiguous().view(x.size(0), self.grid_size[0], self.grid_size[1], -1)
        return x
