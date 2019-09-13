import torch
import torchvision


class DetectModel(torch.nn.Module):
    def __init__(self):
        super(DetectModel, self).__init__()
        self.feature = torchvision.models.resnet50(pretrained=False)
        self.feature.fc = torch.nn.Identity()
        self.fc1 = torch.nn.Linear(2048, 4096)
        self.out = torch.nn.Linear(4096, 7 * 7 * 30)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 7, 7, 30)
        return x
