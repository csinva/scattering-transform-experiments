'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
from .scatwave.scattering import Scattering



__all__ = ['alexscat2_sep']


class AlexScat2_Sep(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexScat2_Sep, self).__init__()

        self.J1 = 2
        self.N1 = 32
        self.L1 = 2
        self.scat1 = Scattering(M=self.N1,N=self.N1,J=self.J1, L = self.L1).cuda()
        self.nfscat1 = (1 + self.L1 * self.J1 + self.L1 * self.L1 * self.J1 * (self.J1 - 1) / 2)
       #print(self.nfscat1*3)
        self.nspace1 = self.N1 / (2 ** self.J1)

        #Combinations to try:
        #J L TOTAL_FILTERS
        #2 2 27
        #2 3 48
        #1 10 33
        #3 1 21 ???


        self.J2 = 1
        self.N2 = 8
        self.L2 = 2
        self.scat2 = Scattering(M=self.N2,N=self.N2,J=self.J2, L=self.L2).cuda()

        self.nfscat2 = (1 + self.L2 * self.J2 + self.L2 * self.L2 * self.J2 * (self.J2 - 1) / 2)
        self.nspace2 = self.N2 / (2 ** self.J2)
        print(self.nfscat2*3)

        self.n_flayer1 = 64
        self.n_flayer2 = 192

        self.first_two_layers = nn.Sequential(
            nn.Conv2d(3, self.n_flayer1 - self.nfscat1*3, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.n_flayer1 - self.nfscat1*3, self.n_flayer2 - self.nfscat2*self.nfscat1*3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
            )


        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.n_flayer2, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = self.first_two_layers(x)
        x2 = torch.autograd.Variable(self.scat1(x.data), requires_grad = False)
        x2 = x2.view(x.size(0), self.nfscat1*3, self.nspace1, self.nspace1)
        x2 = torch.autograd.Variable(self.scat2(x2.data), requires_grad = False)
        x2 = x2.view(x.size(0), x2.size(1)*x2.size(2), self.nspace2, self.nspace2)
        x = torch.cat([x1,x2], 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexscat2_sep(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexScat2_Sep(**kwargs)
    return model


#apython -i run_cifar100.py -a alexscat2_sep --schedule 81 122 164 --epochs 206 --checkpoint checkpoint/alexscat2_sep_extra