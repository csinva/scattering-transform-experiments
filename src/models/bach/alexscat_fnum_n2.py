'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
from .scatwave.scattering import Scattering



__all__ = ['alexscat_fnum_n2']


class AlexScat_FNum_n2(nn.Module):

    def __init__(self, num_classes=10, n = 32, j = 2, l = 2, extra_conv = 0):
        super(AlexScat_FNum_n2, self).__init__()
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

        if self.nfscat*3 < self.n_flayer:        
            self.first_layer = nn.Sequential(
                nn.Conv2d(3, self.n_flayer - self.nfscat*3, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True)
                )
        elif self.extra_conv:
            self.first_layer = nn.Sequential(
                nn.Conv2d(3, self.extra_conv, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True)
                )
            self.n_flayer = self.nfscat * 3 + self.extra_conv
        else:
            self.n_flayer = self.nfscat*3

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
        if self.remove is not None:
            x2[:,self.remove,:,:] = 0
        if self.nfscat*3 < self.n_flayer:
            x1 = self.first_layer(x)
            x = torch.cat([x1,x2], 1)
        else:
            x = x2

        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexscat_fnum_n2(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexScat_FNum_n2(**kwargs)
    return model


#apython run_cifar100.py -a alexscat2_sep --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexscat2_sep_extra;apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l2_n2_extra;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 3 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l3_n2_extra;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 4 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l4_n2_extra;apython run_cifar100.py -a alexscat2_sep_schannel --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexscat2_sep_schannel_extra

#apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.005 --lr 0.005;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l2_n2_extra_lr_0.005 --lr 0.005;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 3 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l3_n2_extra_lr_0.005 --lr 0.005;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 4 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l4_n2_extra_lr_0.005 --lr 0.005;
#\ apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.01 --lr 0.01;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l2_n2_extra_lr_0.01 --lr 0.01;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 3 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l3_n2_extra_lr_0.01 --lr 0.01;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 4 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l4_n2_extra_lr_0.01 --lr 0.01;
#apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05 --lr 0.05;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l2_n2_extra_lr_0.05 --lr 0.05;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 3 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l3_n2_extra_lr_0.05 --lr 0.05;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 4 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l4_n2_extra_lr_0.05 --lr 0.05; apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.1 --lr 0.1;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l2_n2_extra_lr_0.1 --lr 0.1;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 3 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l3_n2_extra_lr_0.1 --lr 0.1;apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 4 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l4_n2_extra_lr_0.1 --lr 0.1;


#apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_preload_10 --lr 0.05 --copy-path checkpoint/alexnet_n2_extra_lr_0.05 --copy-num 10; apython run_cifar100.py -a alexnet_n2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_preload_all --lr 0.05 --copy-path checkpoint/alexnet_n2_extra_lr_0.05 --copy-num -1

#apython run_cifar100.py -a alexnet_n2_batchnorm --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_bnorm_preload_all --lr 0.05 --copy-path checkpoint/alexnet_n2_extra_lr_0.05 --copy-num -1

#apython run_cifar100.py -a alexnet_n2_batchnorm --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_bnorm_preload_10 --lr 0.05 --copy-path checkpoint/alexnet_n2_extra_lr_0.05 --copy-num 10

#apython run_cifar100.py -a alexnet_n2_copy2 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_copy2 --lr 0.05 --copy-path checkpoint/alexnet_n2_extra_lr_0.05 --copy-num -1



#apython run_cifar100.py -a alexscat_fnum_n2_bnorm --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l3_n2_extra_lr_0.05_bnorm_preload_all --lr 0.05 --copy-path checkpoint/j2l3_n2_extra_lr_0.05 --copy-num -1


#apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 8 --extra_conv 16 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05 --lr 0.05;

#apython run_cifar100.py -a alexnet_n2 --schedule 81 122 206 --epochs 248 --checkpoint checkpoint/alexnet_n2_extra_lr --lr 0.01
#apython run_cifar100.py --arch alexnet_n2 --checkpoint checkpoint/alexnet_n2; apython run_cifar100.py --arch alexscat_fnum_n2 --ascat_j 2 --ascat_l 2 --checkpoint checkpoint/j2l2_n2; apython run_cifar100.py --arch alexscat_fnum_n2 --ascat_j 2 --ascat_l 3 --checkpoint checkpoint/j2l3_n2; apython run_cifar100.py --arch alexscat_fnum_n2 --ascat_j 2 --ascat_l 4 --checkpoint checkpoint/j2l4_n2

#apython importance_plots_alexnet.py --checkpoint checkpoint/alexnet_n2_extra_lr_0.1
#apython importance_plots_alexnet.py --arch alexnet_n2 --checkpoint checkpoint/alexnet_n2_extra_lr_0.05
#apython importance_plots_alexnet.py --arch alexnet_n2 --checkpoint checkpoint/alexnet_n2_extra_lr_0.01;  apython importance_plots_alexnet.py --arch alexnet_n2 --checkpoint checkpoint/alexnet_n2_extra_lr_0.005;
#apython importance_plots_alexnet.py --arch alexnet_n2_batchnorm --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_bnorm_preload_all;
#apython importance_plots_alexnet.py --arch alexnet_n2_batchnorm --checkpoint checkpoint/alexnet_n2_extra_lr_0.05_bnorm_preload_10;
#apython importance_plots.py --arch alexscat_fnum_n2_bnorm --checkpoint checkpoint/j2l3_n2_extra_lr_0.05_bnorm_preload_all -l 3;

#apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 8 --extra_conv 64 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_64conv --lr 0.05; apython importance_plots.py --arch alexscat_fnum_n2 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_64conv -l 8 --extra_conv 64;




#apython run_cifar100.py -a alexscat_fnum_n2 --ascat_j 2 --ascat_l 8 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05 --lr 0.05; apython importance_plots.py --checkpoint checkpoint/j2l8_n2_extra_lr_0.05 -l 8

#apython  importance_plots.py --checkpoint checkpoint/j2l2_n2_extra_lr_0.005 -l 2;
#apython  importance_plots.py --checkpoint checkpoint/j2l2_n2_extra_lr_0.01 -l 2; apython  importance_plots.py --checkpoint checkpoint/j2l2_n2_extra_lr_0.05 -l 2; apython  importance_plots.py --checkpoint checkpoint/j2l2_n2_extra_lr_0.1 -l 2; apython  importance_plots.py --checkpoint checkpoint/j2l3_n2_extra_lr_0.005 -l 3; apython  importance_plots.py --checkpoint checkpoint/j2l3_n2_extra_lr_0.01 -l 3; apython  importance_plots.py --checkpoint checkpoint/j2l3_n2_extra_lr_0.05 -l 3; apython  importance_plots.py --checkpoint checkpoint/j2l3_n2_extra_lr_0.1 -l 3;
#apython  importance_plots.py --checkpoint checkpoint/j2l4_n2_extra_lr_0.005 -l 4; apython  importance_plots.py --checkpoint checkpoint/j2l4_n2_extra_lr_0.01 -l 4; apython  importance_plots.py --checkpoint checkpoint/j2l4_n2_extra_lr_0.05 -l 4; apython  importance_plots.py --checkpoint checkpoint/j2l4_n2_extra_lr_0.1 -l 4;






##apython run_cifar100.py -a alexscat_fnum_n2_1x1 --ascat_j 2 --ascat_l 8 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_1x1 --lr 0.05;
#apython importance_plots_1x1.py --arch alexscat_fnum_n2_1x1 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_1x1 -l 8;
#apython run_cifar100.py -a alexscat_fnum_n2_1x1 --ascat_j 2 --ascat_l 8 --extra_conv 64 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_64conv_1x1 --lr 0.05; apython importance_plots_1x1.py --arch alexscat_fnum_n2_1x1 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_64conv_1x1 -l 8 --extra_conv 64;

#apython run_cifar100.py -a alexscat_fnum_n2_res --ascat_j 2 --ascat_l 8 --schedule 81 122 164 206 --epochs 248 --checkpoint checkpoint/j2l8_n2_extra_lr_0.05_res --lr 0.05;
#apython run_cifar100.py -a alexscat_fnum_n2_res --ascat_j 2 --ascat_l 8 --schedule 50 90 --epochs 124 --checkpoint checkpoint/j2l8_lin_lr_0.05_res --lr 0.05;
#apython run_cifar100.py -a alexscat_fnum_n2_res --ascat_j 2 --ascat_l 8 --schedule 50 90 --epochs 124 --extra_conv 64 --checkpoint checkpoint/j2l8_lin_lr_0.05_res_2 --lr 0.05 --copy-path checkpoint/j2l8_lin_lr_0.05_res --copy-num -1; apython  importance_plots.py --checkpoint checkpoint/j2l8_lin_lr_0.05_res_2 -l 8 --extra_conv 64 --arch alexscat_fnum_n2_res;
