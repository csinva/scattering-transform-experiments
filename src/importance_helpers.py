from __future__ import absolute_import

import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torchvision import models
import models.cifar as models
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torchvision.datasets as datasets
import copy
from utils import AverageMeter, accuracy


from scatwave.filters_bank import filters_bank
from scatwave.differentiable import scattering, cast
import new

def test(testloader, model, criterion, epoch, use_cuda):
    '''
    Evaluates the test accuracy of the model. This is used to evaluate the importance of a filter after removing it from the model.
    This code is taken directly from the training code that plots the test error.
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time

        # plot progress
    print("top1.avg")
    print(top1.avg)
    print("TEST OVER")
    return (losses.avg, top1.avg)


def make_image():
    '''
    Makes a noisy image and turns it into a torch array so we can later turn it into a variable and calculate gradients on it and apply them.
    '''
    images = []
    for _ in range(3):
        image = np.random.randn(11,11)/5+0.5
        image = np.pad(image, ((10, 11), (11, 10)), 'constant')
        images.append(image)
    image = np.stack(images)
    image = image.transpose(0,2,1)

    torch_image = torch.from_numpy(image).float()
    torch_image = torch_image.unsqueeze_(0)
    return torch_image


def get_alexnet_important_filts(checkpoint, num_filts=-1):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    epoch = 164


    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    best_alexnet = torch.load(checkpoint+"/model_best.pth.tar")
    model = models.__dict__["alexnet_n2"](num_classes=100)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(best_alexnet['state_dict'])
    conv1_weights = best_alexnet['state_dict']['module.features.0.weight'].cpu().numpy()
    tensor = np.swapaxes(conv1_weights,1,3)


    model2 = models.__dict__["alexnet_n2"](num_classes=100)
    model2 = torch.nn.DataParallel(model2).cuda()
    model2.load_state_dict(best_alexnet['state_dict'])
    losses, top1 = test(testloader, model2, criterion, epoch, True)
    allFilters = top1



    scores = []
    for f_num in range(64):
        best_alexnet2 = copy.deepcopy(best_alexnet['state_dict'])
        best_alexnet2['module.features.0.weight'][f_num] = 0
        model2.load_state_dict(best_alexnet2)
        losses, top1 = test(testloader, model2, criterion, epoch, True)
        scores.append(top1)

    scores = np.array(scores)
    if num_filts != -1:
        best = np.argsort(scores)[:num_filts]
        return best_alexnet['state_dict']['module.features.0.weight'][best]
    else:
        return best_alexnet['state_dict']['module.features.0.weight']
        #best = np.argsort(scores)

def get_important_filts(checkpoint, l, num_filts):

    #This code prepares the data to evaluate the test accuracy. This was taken directly from the training code.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    epoch = 164



    #This code loads the model in and extracts the conv1_weights
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"


    best_ascat = torch.load(checkpoint+"/model_best.pth.tar")
    model = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=l)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(best_ascat['state_dict'])


    if l < 4:
        conv1_weights = best_ascat['state_dict']['module.first_layer.0.weight'].cpu().numpy()
        tensor = np.swapaxes(conv1_weights,1,3)

    #This is another copy of the same model so we can alter it.
    model2 = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=l)
    model2 = torch.nn.DataParallel(model2).cuda()


    best_ascat2 = copy.deepcopy(best_ascat['state_dict'])
    model2.load_state_dict(best_ascat2)
    losses, top1 = test(testloader, model2, criterion, epoch, True)
    allFilters = top1

    #This code just gets the importance scores of each filter
    scores = []
    for f_num in range(model.module.n_flayer - model.module.nfscat*3):
	best_ascat2 = copy.deepcopy(best_ascat['state_dict'])
	best_ascat2['module.first_layer.0.weight'][f_num] = 0
	model2.load_state_dict(best_ascat2)
	losses, top1 = test(testloader, model2, criterion, epoch, True)
	scores.append(top1)


    scores = np.array(scores)
    if num_filts != -1:
        best = np.argsort(scores)[:num_filts]
        return best_ascat['state_dict']['module.first_layer.0.weight'][best]
    else:
        return best_ascat['state_dict']['module.first_layer.0.weight']
