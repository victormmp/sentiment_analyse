import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, features: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)

        return x
