import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.conv2,
            self.bn2,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        if self.conv3 is not None:
            shortcut = self.conv3(x)
        return self.relu(self.net(x) + shortcut)

if __name__ == "__main__":
    myRes = Residual(1,32,use_1x1conv=True,strides=1)
    x = torch.randn(32,1,28,28)
    y = myRes(x)
    print(y.shape)