# Deep learning libraries: torch and torchvision
import torch
from torch import nn
from torch.utils import data

from torchvision import datasets, transforms # pip install torchvision
from torchvision.utils import save_image


class LinearAutoEncoder(nn.Module):

    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 1024),  # 3*64*64   ====> 1024 
            nn.ReLU(True),
            nn.Linear(1024, 256),  # 1024 ===> 256
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))  #
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(True),
                                     nn.Linear(128, 256), nn.ReLU(True),
                                     nn.Linear(256, 1024), nn.ReLU(True),
                                     nn.Linear(1024, image_size), nn.Tanh())  #

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # loss = \hat{x} - x  <== outputs - images
