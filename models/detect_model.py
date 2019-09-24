import torch
import torch.nn as nn
import torchvision.models as models
from models.conv_net import ConvNet


class DetectModel(nn.Module):
    def __init__(self):
        super(DetectModel, self).__init__()
        # self.feature = torchvision.models.vgg19_bn(pretrained=True)
        # self.feature.classifier = torch.nn.Identity()
        # self.feature = models.resnet50(pretrained=True)
        # self.feature.fc = nn.Identity()
        self. feature = ConvNet()
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.leaky_relu = nn.LeakyReLU()
        self.out = nn.Linear(4096, 7 * 7 * 30)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 7, 7, 30)
        return x
