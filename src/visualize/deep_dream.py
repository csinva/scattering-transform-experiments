"""
Created on Mon Nov 21 21:57:29 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2

import torch
from torch.optim import SGD
from torchvision import models

from misc_functions import recreate_image

import models.cifar as models

import numpy as np 
from torch.autograd import Variable

import copy

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im



class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, im_path = False, a = None):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.a = a
        # Generate a random image
        if im_path:
        	self.created_image = cv2.imread(im_path, 1)
        else:
        	#self.created_image = np.zeros((32,32,3))
        	self.created_image = cv2.randn(np.zeros((32,32,3)), (0), (99))
        # Hook the layers to get result of the convolution
        self.hook_layer()
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, False)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas layer layers need less
        optimizer = SGD([self.processed_image], lr=12,  weight_decay=1e-4)
        for i in range(1, 251):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            if self.a:
                #print(self.a)
                x1 = self.a.first_layer(x)
                x2 = torch.autograd.Variable(self.a.scat(x.data.contiguous()), requires_grad = True)
                x2 = x2.view(x.size(0), self.a.nfscat*3, self.a.nspace, self.a.nspace)

                x = torch.cat([x1,x2], 1)

            for index, layer in enumerate(self.model):
                # Forward
                x = layer(x)
                # Only need to forward until we the selected layer is reached
                if index == self.selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()[0]))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image every 20 iteration
            if i % 20 == 0:
                cv2.imwrite('../generated/ddream_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
                            self.created_image)


if __name__ == '__main__':
    ### THIS OPERATION IS MEMORY HUNGRY! ###
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    cnn_layer = 10
    filter_pos = 1

    #im_path = 'pytorch-cnn-visualizations/input_images/cat_dog.png'


    #im_path = 'pytorch-cnn-visualizations/input_images/dd_tree.jpg'
    #im_path = 'cifar_image.png'

    # Fully connected layer is not needed
    #pretrained_model = models.vgg19(pretrained=True).features
    im_path = None

    #ALEXNET HERE
    best_alexnet = torch.load('model_best.pth.tar')
    a = models.__dict__["alexnet"](num_classes=100)
    a = torch.nn.DataParallel(a).cuda()
    a.load_state_dict(best_alexnet['state_dict'])
    dd = DeepDream(a.module.features, cnn_layer, filter_pos, im_path)


    #ALEXSCAT HERE
    #NOTE THAT ALEXSCAT_FNUM SELF.N MUST CHANGE TO FIT IMAGE SIZE. BRING THAT UP WITH CHANDAN
    #best_ascat = torch.load("checkpoints/cifar100/alexscat_j2l2/model_best.pth.tar")
    #a = models.__dict__["alexscat_fnum"](num_classes=100,n=32)
    #a = torch.nn.DataParallel(a).cuda()
    #a.load_state_dict(best_ascat['state_dict'])
    #dd = DeepDream(a.module.features, cnn_layer, filter_pos, im_path, a.module)


    # This operation can also be done without Pytorch hooks
    # See layer visualisation for the implementation without hooks
    dd.dream()
