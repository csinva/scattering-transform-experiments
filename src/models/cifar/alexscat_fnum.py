'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
from .scatwave.scattering import Scattering



__all__ = ['alexscat_fnum']


class AlexScat_FNum(nn.Module):

    def __init__(self, num_classes=10, n = 32, j = 2, l = 2):
        super(AlexScat_FNum, self).__init__()
        self.J = j
        self.N = n
        self.L = l
        self.scat = Scattering(M=self.N,N=self.N,J=self.J, L = self.L).cuda()
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

        self.n_flayer = 64 #NORMALLY 64 FOR NORMAL ALEXNET

        if self.nfscat*3 < self.n_flayer:        
            self.first_layer = nn.Sequential(
                nn.Conv2d(3, self.n_flayer - self.nfscat*3, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True)
                )
        else:
            self.n_flayer = self.nfscat*3

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.n_flayer, 192, kernel_size=5, padding=2),
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

        x2 = torch.autograd.Variable(self.scat(x.data), requires_grad = False)
        x2 = x2.view(x.size(0), self.nfscat*3, self.nspace, self.nspace)

        if self.nfscat*3 < self.n_flayer:
            x1 = self.first_layer(x)
            x = torch.cat([x1,x2], 1)
        else:
            x = x2

        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexscat_fnum(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexScat_FNum(**kwargs)
    return model




#apython run_cifar100.py --arch alexnet --checkpoint checkpoint/alexnet; apython run_cifar100.py --ascat_j 2 --ascat_l 2 --checkpoint checkpoint/j2l2; apython run_cifar100.py --ascat_j 2 --ascat_l 3 --checkpoint checkpoint/j2l3; apython run_cifar100.py --ascat_j 2 --ascat_l 4 --checkpoint checkpoint/j2l4
