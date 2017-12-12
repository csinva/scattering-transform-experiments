'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
from scatwave.scattering import Scattering



__all__ = ['scat2only']


class Scat2Only(nn.Module):

    def __init__(self, num_classes=10):
        super(Scat2Only, self).__init__()
        self.J1 = 2
        self.N1 = 32
        self.scat1 = Scattering(M=32,N=32,J=self.J1).cuda()
        self.nfscat1 = (1 + 8 * self.J1 + 8 * 8 * self.J1 * (self.J1 - 1) / 2)
        self.nspace1 = self.N1 / (2 ** self.J1)


        self.J2 = 2
        self.N2 = 8
        self.scat2 = Scattering(M=8,N=8,J=self.J2).cuda()

        self.nfscat2 = (1 + 8 * self.J2 + 8 * 8 * self.J2 * (self.J2 - 1) / 2)
        self.nspace2 = self.N2 / (2 ** self.J2)

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(243, num_classes)

    def forward(self, x):
        x = torch.autograd.Variable(self.scat1(x.data), requires_grad = False)
        x = x.view(x.size(0), self.nfscat1*3, self.nspace1, self.nspace1)
        x = torch.autograd.Variable(self.scat2(x.data), requires_grad = False)
        x = x.view(x.size(0), self.nfscat2*3, self.nspace2, self.nspace2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def scat2only(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = Scat2Only(**kwargs)
    return model
