import torch.nn as nn
import numpy as np
import torch.nn.functional as F

img_shape = (3, 128, 128)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 64 1 28 28 =>64 784
        validity = self.model(img_flat)  # 64 784 =>64 1

        return validity

