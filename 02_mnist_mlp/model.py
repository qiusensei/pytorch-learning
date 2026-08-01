import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self,x):
        return self.net(x)

if __name__ == "__main__":
    model = MyMLP()
    x = torch.randn(64,784)
    out = model(x)
    print(out.shape)