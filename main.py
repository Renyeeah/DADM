import os
import random

import torch
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from tqdm import trange

from models import Discriminator
from dataset import ImageDataset
from readData import DADDPMdataset

parser = argparse.ArgumentParser()

# input
parser.add_argument('--img_size', type=int, default=128, help='image size')
parser.add_argument('--dataset', type=str, default='facades', help='name of dataset')
parser.add_argument('--input_channel', type=int, default=3, help='color channel of input image')

# train
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--parallel', type=bool, default=False, help='multiple GPUs train')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0")


def warmup_lr(step):
    return min(step, opt.epoch / 2) / (opt.epoch / 2)

def pre_train_D():

    target_real = Variable(torch.Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(torch.Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

    trainDataset_A = DADDPMdataset(
        path=os.path.join('dataset', 'facades', 'train', 'A'),
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
        ])
    )

    trainDataloader_A = torch.utils.data.DataLoader(
        dataset=trainDataset_A,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )

    trainDataset_B = DADDPMdataset(
        path=os.path.join('dataset', 'facades', 'train', 'B'),
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
        ])
    )

    trainDataloader_B = torch.utils.data.DataLoader(
        dataset=trainDataset_B,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )

    DomainDiscriminator = Discriminator().to(device)
    optimizer_D = torch.optim.Adam(DomainDiscriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=warmup_lr)

    if not os.path.exists(os.path.join('ckpts', opt.dataset)):
        os.makedirs(os.path.join('ckpts', opt.dataset))

    with trange(opt.epoch, dynamic_ncols=True) as pbar:
        for epoch in pbar:
            imgs_x_0_A = []
            imgs_x_0_B = []
            for x_0_A, _ in iter(trainDataloader_A):
                imgs_x_0_A.append(x_0_A)
            for x_0_B, _ in iter(trainDataloader_B):
                imgs_x_0_B.append(x_0_B)

            for x_0_A in imgs_x_0_A:
                x_0_A = x_0_A.to(device)
                prev_Domain_x_0_A = DomainDiscriminator(x_0_A)
                loss_D_Fake = F.mse_loss(prev_Domain_x_0_A, target_fake.to(x_0_A.device))
                x_0_B = imgs_x_0_B[random.randint(0, len(imgs_x_0_B)-1)].to(x_0_A.device)
                prev_Domain_x_0_B = DomainDiscriminator(x_0_B)
                loss_D_Real = F.mse_loss(prev_Domain_x_0_B, target_real.to(x_0_A.device))
                loss_D = (loss_D_Fake + loss_D_Real) / 2

                loss_D.backward()
                optimizer_D.step()

            lr_scheduler_D_A.step()
            torch.save(DomainDiscriminator.state_dict(), os.path.join('ckpts', opt.dataset, 'DomainDiscriminator.pth'))

def train():
    pass

if __name__ == '__main__':

    testDataset_A = DADDPMdataset(
        path=os.path.join('dataset', 'facades', 'test', 'A'),
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
        ])
    )

    testDataloader_A = torch.utils.data.DataLoader(
        dataset=testDataset_A,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )

    testDataset_B = DADDPMdataset(
        path=os.path.join('dataset', 'facades', 'train', 'B'),
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
        ])
    )

    testDataloader_B = torch.utils.data.DataLoader(
        dataset=testDataset_B,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )

    ckpt_DomainDiscriminator = torch.load(os.path.join('ckpts', 'facades', 'DomainDiscriminator.pth'))
    DomainDiscrimiator = Discriminator()
    DomainDiscrimiator.load_state_dict(ckpt_DomainDiscriminator)

    for i, _ in testDataloader_A:
        a = DomainDiscrimiator(i)
        print(a)



