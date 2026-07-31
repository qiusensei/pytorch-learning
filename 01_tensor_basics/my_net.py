import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()                                          #What is "super" doing?
        self.layer1 = nn.Linear(4,8)          #What is Linear?
        self.relu = nn.ReLU()                                       #I create a self.relu here,am I?
        self.layer2 = nn.Linear(8,2)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    model = MyNet()
    x = torch.randn(5, 4)
    x_output = model(x)

    print(list(model.parameters()))  # How can I print every parameter in the list?
    print("输出形状:", x_output.shape)

    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}")