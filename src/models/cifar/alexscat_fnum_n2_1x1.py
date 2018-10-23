'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch
import torch.nn as nn
from .scatwave.scattering import Scattering



__all__ = ['alexscat_fnum_n2_1x1']


class AlexScat_FNum_n2_1x1(nn.Module):

    def __init__(self, num_classes=10, n = 32, j = 2, l = 2, extra_conv = 0):
        super(AlexScat_FNum_n2_1x1, self).__init__()
        self.J = j
        self.N = n
        self.L = l
        self.extra_conv = extra_conv
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

        self.post_scat = nn.Sequential(
            nn.Conv2d(self.nfscat * 3, self.n_flayer, kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True)
            )

        #if self.nfscat*3 < self.n_flayer:
        #    self.first_layer = nn.Sequential(
        #        nn.Conv2d(3, self.n_flayer - self.nfscat*3, kernel_size=11, stride=4, padding=5),
        #        nn.ReLU(inplace=True)
        #        )
        if self.extra_conv:
            self.first_layer = nn.Sequential(
                nn.Conv2d(3, self.extra_conv, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True)
                )
            self.n_flayer = self.n_flayer + self.extra_conv
        #else:
        #    self.n_flayer = self.nfscat*3

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.n_flayer, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(384, num_classes)
        self.remove = None

    def forward(self, x):

        x2 = torch.autograd.Variable(self.scat(x.data), requires_grad = False)
        x2 = x2.view(x.size(0), self.nfscat*3, self.nspace, self.nspace)
        x2 = self.post_scat(x2)
        if self.remove is not None:
            x2[:,self.remove,:,:] = 0
        if self.extra_conv:
            x1 = self.first_layer(x)
            x = torch.cat([x1,x2], 1)
        else:
            x = x2

        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexscat_fnum_n2_1x1(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexScat_FNum_n2_1x1(**kwargs)
    return model
