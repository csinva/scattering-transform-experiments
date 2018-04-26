'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
from .scatwave.scattering import Scattering



__all__ = ['scatfirst2']


class ScatFirst2(nn.Module):

    def __init__(self, num_classes=10):
        super(ScatFirst2, self).__init__()
        self.J = 2
        self.N = 32
        self.scat = Scattering(M=32,N=32,J=self.J).cuda()
        self.nfscat = (1 + 8 * self.J + 8 * 8 * self.J * (self.J - 1) / 2)
        self.nspace = self.N / (2 ** self.J)

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.nfscat*3, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.autograd.Variable(self.scat(x.data), requires_grad = False)
        x = x.view(x.size(0), self.nfscat*3, self.nspace, self.nspace)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def scatfirst2(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = ScatFirst2(**kwargs)
    return model
