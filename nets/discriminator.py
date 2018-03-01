import torch
import numpy as np

from torch import nn


class Avg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(torch.mean(x, -1), -1)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 64, (4, 4), 2, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(64, 128, (4, 4), 2, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(128, 256, (4, 4), 2, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(256, 512, (4, 4), 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(512, 1, (4, 4), 1, 1))
        
    def forward(self, x):
        
        output = self.conv(x)
        output = nn.Sigmoid()(Avg()(output))
        return output
    
    