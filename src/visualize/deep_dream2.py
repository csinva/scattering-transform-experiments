import os
import cv2

import torch
from torch.optim import SGD
from torchvision import models

import models.cifar as models

import numpy as np 
from torch.autograd import Variable

import copy

#image = cv2.randn(np.zeros((32,32,3)), (0), (0.5,0.5,0.5))
#image = np.zeros((32,32,3)) + 0.5

def make_image():    
    image = np.random.randn(32,32,3)/5+0.5
    image = image.transpose(2,0,1)

    torch_image = torch.from_numpy(image).float()
    torch_image = torch_image.unsqueeze_(0)
    return torch_image



#im_as_var = Variable(torch_image.cuda(), requires_grad=True)
torch_image = make_image()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
best_ascat = torch.load("checkpoints/cifar100/alexscat_j2l2/model_best.pth.tar")
model = models.__dict__["alexscat_fnum"](num_classes=100,n=32,j=2,l=2)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(best_ascat['state_dict'])



for f_num in range(32):
    im_as_var = Variable(torch_image.cuda(), requires_grad=True)
    optimizer = SGD([im_as_var], lr=12,  weight_decay=1e-4)
    for i in range(1, 501):
        optimizer.zero_grad()

        x = im_as_var
        x = model.module.first_layer[0](x)
        #x = model.module.scat(x.data)
        #x = torch.autograd.Variable(model.module.scat(x.data.contiguous()), requires_grad = True)
        #x = x.view(x.size(0), model.module.nfscat*3, model.module.nspace, model.module.nspace)
        loss = x[0, f_num, 4, 4]
        #print(loss)
        # if i == 1:
        #     count = 0
        #     while loss.data[0] == 0 and count < 500:
        #         torch_image = make_image()
        #         im_as_var = Variable(torch_image.cuda(), requires_grad=True)
        #         optimizer = SGD([im_as_var], lr=12,  weight_decay=1e-4)
        #         optimizer.zero_grad()
        #         x = im_as_var
        #         x = model.module.first_layer(x)
        #         loss = x[0, f_num, 4, 4]
        #         count += 1
        #     if count >= 250:
        #         print("FILTER " + str(f_num) + " COULD NOT FIND A NON ZERO VALUE")

        loss.backward()
        optimizer.step()

    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0]).transpose(1,2,0)*255
        
    cv2.imwrite("Test"+str(f_num)+".jpg", recreated_im)