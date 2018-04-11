'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
from .scatwave.scattering import Scattering



#__all__ = ['alexscat_fnum']


class AlexScat_FNum(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexScat_FNum, self).__init__()
        self.J = 2
        self.N = 32
        self.L = 2
        self.scat = Scattering(M=32,N=32,J=self.J, L = self.L).cuda()
        self.nfscat = (1 + self.L * self.J + self.L * self.L * self.J * (self.J - 1) / 2)
        print(self.nfscat*3)
        self.nspace = self.N / (2 ** self.J)

        #assert self.nfscat*3 <= 64
        #Combinations to try:
        #J L TOTAL_FILTERS
        #2 2 27
        #2 3 48
        #1 10 33
        #3 1 21 ???



        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64 - self.nfscat*3, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = self.first_layer(x)
        #print("X1SHAPE")
        #print(x1.shape)
        x2 = torch.autograd.Variable(self.scat(x.data), requires_grad = False)
        x2 = x2.view(x.size(0), self.nfscat*3, self.nspace, self.nspace)
        #print("X2SHAPE")
        #print(x2.shape)
        x = torch.cat([x1,x2], 1)
        x = self.features(x)
        #print("X2SHAPE")
        #print(x2.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexscat_fnum(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexScat_FNum(**kwargs)
    return model
