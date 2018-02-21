import os
#import cv2
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

#image = cv2.randn(np.zeros((32,32,3)), (0), (0.5,0.5,0.5))
#image = np.zeros((32,32,3)) + 0.5


def test(testloader, model, criterion, epoch, use_cuda):

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
    images = []
    for _ in range(3):
        image = np.random.randn(11,11)/5+0.5
        image = np.pad(image, ((11, 11), (11, 11)), 'constant')
        images.append(image)
    image = np.stack(images)
    image = image.transpose(0,2,1)

    torch_image = torch.from_numpy(image).float()
    torch_image = torch_image.unsqueeze_(0)
    return torch_image



#im_as_var = Variable(torch_image.cuda(), requires_grad=True)




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
best_ascat = torch.load("checkpoint/j2l2_n2/model_best.pth.tar")
model = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=2)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(best_ascat['state_dict'])
conv1_weights = best_ascat['state_dict']['module.first_layer.0.weight'].cpu().numpy()
tensor = np.swapaxes(conv1_weights,1,3)


model2 = models.__dict__["alexscat_fnum_n2"](num_classes=100,n=32,j=2,l=2)
model2 = torch.nn.DataParallel(model2).cuda()



scores = []
for f_num in range(model.module.n_flayer - model.module.nfscat*3):
	best_ascat2 = copy.deepcopy(best_ascat['state_dict'])	
	best_ascat2['module.first_layer.0.weight'][f_num] = 0
	model2.load_state_dict(best_ascat2)
	losses, top1 = test(testloader, model2, criterion, epoch, True)
	scores.append(top1)

scores = np.array(scores)
torch_image = make_image()

#if not os.path.exists('../importance/j2l3'):
#    os.makedirs('../importance/j2l3')

num_cols = 8
num_rows = 1 + len(scores)//num_cols
fig = plt.figure(figsize=(num_cols, num_rows))
for importance, f_num in enumerate(np.argsort(scores)):
    ax1 = fig.add_subplot(num_rows, num_cols, importance + 1)
    minned = tensor[f_num] - np.min(tensor[f_num])
    ax1.imshow(minned/np.max(minned))
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(str(scores[f_num]))
plt.subplots_adjust(wspace=1.0, hspace=0.1)
plt.savefig("filters_alexscat_n2_j2l2.png")

plt.show()


fig = plt.figure(figsize=(num_cols, num_rows))
REGULARIZATION = 0.0001
for importance, f_num in enumerate(np.argsort(scores)):
    print(importance)
    im_as_var = Variable(torch_image.cuda(), requires_grad=True)
    optimizer = SGD([im_as_var], lr=12,  weight_decay=1e-4)
    for i in range(1, 501):
        optimizer.zero_grad()

        x = im_as_var
        x = model.module.first_layer[0](x)
        #x = x.view(x.size(0), model.module.nfscat*3, model.module.nspace, model.module.nspace)
        loss = x[0, f_num, 4, 4]

        #https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
        reg_loss = REGULARIZATION * (
        torch.sum(torch.abs(im_as_var[:, :, :-1] - im_as_var[ :, :, 1:])) + 
        torch.sum(torch.abs(im_as_var[ :, :-1, :] - im_as_var[:, 1:, :]))
        )


        loss = loss + reg_loss

        loss.backward()
        optimizer.step()

    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0]).transpose(1,2,0)
    recreated_im = recreated_im[11:22,11:22,:]
    minned = recreated_im - np.min(recreated_im)
    ax1 = fig.add_subplot(num_rows, num_cols, importance + 1)
    ax1.imshow(minned/np.max(minned))
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(str(scores[f_num]))
plt.subplots_adjust(wspace=1.0, hspace=0.1)
plt.savefig("deep_dream_alexscat_n2_j2l2.png")

plt.show()


	#cv2.imwrite("../importance/j2l3/"+"i"+str(importance)+"a"+str(scores[f_num])+"f"+str(f_num)+"dream.jpg", fin*255)
	#minned = tensor[f_num] - np.min(tensor[f_num])
	#fin = minned/np.max(minned)
	#cv2.imwrite("../importance/j2l3/"+"i"+str(importance)+"a"+str(scores[f_num])+"f"+str(f_num)+"filter.jpg", fin*255)






