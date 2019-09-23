import torch
import torch.nn as nn
import torchvision.models as models
import torchvision


class DetectModel(nn.Module):
    def __init__(self):
        super(DetectModel, self).__init__()
        # self.feature = torchvision.models.vgg19_bn(pretrained=True)
        # self.feature.classifier = torch.nn.Identity()
        self.feature = models.resnet50(pretrained=True)
        self.feature.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 4096)
        self.leaky_relu = nn.LeakyReLU()
        self.out = nn.Linear(4096, 7 * 7 * 30)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 7, 7, 30)
        return x
