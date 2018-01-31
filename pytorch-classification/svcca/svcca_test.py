import sys

sys.path.append("../models/cifar/")


import os

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from dft_ccas import *

from scatwave.scattering import Scattering
import torch.utils.data as data
import torch.nn.parallel


import torchvision.datasets as datasets
import torchvision.transforms as transforms



class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model



class ScatFirst_FNum(nn.Module):

    def __init__(self, num_classes=10):
        super(ScatFirst_FNum, self).__init__()
        self.J = 2
        self.N = 32
        self.L = 4
        self.scat = Scattering(M=32,N=32,J=self.J, L = self.L).cuda()
        print(len(self.scat.Psi))
        self.nfscat = (1 + self.L * self.J + self.L * self.L * self.J * (self.J - 1) / 2)
        print(self.nfscat*3)
        self.nspace = self.N / (2 ** self.J)

    def forward(self, x):
        x = torch.autograd.Variable(self.scat(x.data), requires_grad = False)
        x = x.view(x.size(0), self.nfscat*3, self.nspace, self.nspace)
        #print("X2SHAPE")
        #print(x2.shape)
        #x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x


def scatfirst_fnum(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = ScatFirst_FNum(**kwargs)
    return model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


dataloader = datasets.CIFAR100
num_classes = 100
trainset = dataloader(root='.././data', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

best_alexnet = torch.load('../checkpoints/cifar100/alexnet/model_best.pth.tar')
a = AlexNet(num_classes = 100)
a = torch.nn.DataParallel(a).cuda()
a.load_state_dict(best_alexnet['state_dict'])
first_layer = nn.Sequential(*list(a.module.features.children())[0:2])

#best_scatfirst_75 = torch.load('../checkpoints/cifar100/scatfirst_75/model_best.pth.tar')
scat = ScatFirst_FNum(num_classes = 100)

for batch_idx, (inputs, targets) in enumerate(trainloader):

    if batch_idx > 0:
    	exit()

    if True:
        inputs, targets = inputs.cuda(), targets.cuda(async=True)
    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

    # compute output
    hmm1 = first_layer(inputs)

    hmm2 = scat(inputs)

    hmm1, hmm2 = hmm1.data.cpu().numpy(), hmm2.data.cpu().numpy()

    print(type(hmm1))
    print(type(hmm2))
    hmm11 = np.swapaxes(hmm1, 1, 3)
    hmm22 = np.swapaxes(hmm2, 1, 3)

    asdf = fourier_ccas(hmm11, hmm22, verbose = True)
    print(asdf)
