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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Importance Plots')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to load checkpoint (default: checkpoint)')
parser.add_argument('-l', default=2, type=int)
args = parser.parse_args()


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


#im_as_var = Variable(torch_image.cuda(), requires_grad=True)

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


best_ascat = torch.load(args.checkpoint+"/model_best.pth.tar")
model = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=args.l)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(best_ascat['state_dict'])


if args.l < 4:
    conv1_weights = best_ascat['state_dict']['module.first_layer.0.weight'].cpu().numpy()
    tensor = np.swapaxes(conv1_weights,1,3)

#This is another copy of the same model so we can alter it.
model2 = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=args.l)
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

print("switching to scat filter importance")

model3 = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=args.l)
for s_num in range(model.module.nfscat*3):
    model3.remove=s_num
    model4 = torch.nn.DataParallel(model3).cuda()
    model4.load_state_dict(best_ascat['state_dict'])
    losses, top1 = test(testloader, model4, criterion, epoch, True)
    scores.append(top1)



scores = np.array(scores)
torch_image = make_image()
#scat_f = filters_bank(32,32,2,l)
psi, phi = cast(model.module.scat.Psi, model.module.scat.Phi, torch.cuda.FloatTensor)
nonscat = model.module.n_flayer - model.module.nfscat*3
num_cols = 8
num_rows = 1 + len(scores)//num_cols
#if not os.path.exists('../importance/j2l3'):
#    os.makedirs('../importance/j2l3')
#import pdb
#pdb.set_trace()
#This code goes through and plots each filter.
num_cols = 8
num_rows = 1 + len(scores)//num_cols
fig = plt.figure(figsize=(num_cols, num_rows))
for importance, f_num in enumerate(np.argsort(scores)):
    if(f_num < nonscat):
        ax1 = fig.add_subplot(num_rows, num_cols, importance + 1)
        minned = tensor[f_num] - np.min(tensor[f_num])
        ax1.imshow(minned/np.max(minned))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_title(str(allFilters - scores[f_num]))
plt.subplots_adjust(wspace=1.0, hspace=0.1)
plt.savefig(args.checkpoint+"/importance_filters.png")

plt.close()

#This code applies the maximal activation for each filter and then plots the resulting image.
fig = plt.figure(figsize=(num_cols, num_rows))
REGULARIZATION = 0.0001
for importance, f_num in enumerate(np.argsort(scores)):
    print(importance)
    im_as_var = Variable(torch_image.cuda(), requires_grad=True)
    optimizer = SGD([im_as_var], lr=12,  weight_decay=1e-4)
    for i in range(1, 501):
        optimizer.zero_grad()

        x = im_as_var

        if(f_num < nonscat):
            x = model.module.first_layer[0](x)
            #y = model.module.features[0](im_as_var)
            #x = x.view(x.size(0), model.module.nfscat*3, model.module.nspace, model.module.nspace)
            loss = -1.0* x[0, f_num, 4, 4]
        else:
            x = scattering(x, psi, phi, 2, args.l)
            x = x.view(x.size(0), model.module.nfscat*3, model.module.nspace, model.module.nspace)
            loss = -1.0* x[0, f_num - nonscat, 4, 4]
        #https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
        #reg_loss = REGULARIZATION * (
        #torch.sum(torch.abs(im_as_var[:, :, :-1] - im_as_var[ :, :, 1:])) + 
        #torch.sum(torch.abs(im_as_var[ :, :-1, :] - im_as_var[:, 1:, :]))
        #)
        reg_loss = 0


        loss = loss + reg_loss


        loss.backward()
        optimizer.step()

    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0]).transpose(2,1,0)
    recreated_im = recreated_im[11:22,11:22,:]
    minned = recreated_im - np.min(recreated_im)
    ax1 = fig.add_subplot(num_rows, num_cols, importance + 1)
    ax1.imshow(minned/np.max(minned))
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    if(f_num < nonscat):
        ax1.set_title(str(allFilters - scores[f_num]))
    else:
        ax1.set_title(str(allFilters - scores[f_num]) + " (s)")
plt.subplots_adjust(wspace=1.0, hspace=0.1)
plt.savefig(args.checkpoint+"/max_act.png")

plt.close()


	#cv2.imwrite("../importance/j2l3/"+"i"+str(importance)+"a"+str(scores[f_num])+"f"+str(f_num)+"dream.jpg", fin*255)
	#minned = tensor[f_num] - np.min(tensor[f_num])
	#fin = minned/np.max(minned)
	#cv2.imwrite("../importance/j2l3/"+"i"+str(importance)+"a"+str(scores[f_num])+"f"+str(f_num)+"filter.jpg", fin*255)






