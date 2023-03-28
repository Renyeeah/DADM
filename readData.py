import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


device = torch.device('cuda:0')


class DADDPMdataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.images = os.listdir(self.path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.path, image_name)

        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name





