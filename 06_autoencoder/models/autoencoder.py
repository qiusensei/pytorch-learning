import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
           nn.Flatten(),
           nn.Linear(784,64),
           nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64,784),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)