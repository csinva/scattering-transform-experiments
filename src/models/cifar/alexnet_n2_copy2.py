'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch
import torch.nn as nn


__all__ = ['alexnet_n2_copy2']


class AlexNet_n2_copy2(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_n2_copy2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192)
        )

        self.features.requires_grad = False

        self.nonfrozen = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192)
            )
        self.combined = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            #nn.Conv2d(192, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            )
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x1 = self.features(x)
        #x1 = x1.view(x.size(0), -1)
        x2 = self.nonfrozen(x)
        #x2 = x2.view(x.size(0), -1)
        x = torch.cat([x1,x2], 1)
        x = self.combined(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet_n2_copy2(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet_n2_copy2(**kwargs)
    return model
