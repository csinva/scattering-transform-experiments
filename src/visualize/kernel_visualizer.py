#Largely taken from
#https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/8
import os

import torch
import torchvision.models as models
import numpy as np
from matplotlib import pyplot as plt

def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if tensor.shape[-1]==3:
        num_kernels = tensor.shape[0]
        num_rows = 1+ num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols,num_rows))
        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            minned = tensor[i] - np.min(tensor[i])
            ax1.imshow(minned/np.max(minned))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
    else:
        num_kernels = tensor.shape[0]
        num_rows = 1+ num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols,num_rows))
        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            summed = np.sum(tensor[i],axis=2)
            minned = summed - np.min(summed)
            ax1.imshow(minned/np.max(minned))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        print(tensor.shape[0])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#best_alexnet = torch.load('model_best.pth.tar')
#conv1_weights = best_alexnet['state_dict']['module.features.0.weight'].cpu().numpy()
#tensor = np.swapaxes(conv1_weights,1,3)

#plot_kernels(tensor)
# print("SHOWING NORAML ALEXNET")
# for i in range(tensor.shape[0]):
#     minned = tensor[i] - np.min(tensor[i])
#     norm = minned/np.max(minned)
#     plt.imshow(norm)
#     plt.show()

alexscat = torch.load('checkpoints/cifar100/alexscat_j2l3_-2layers/model_best.pth.tar')
conv1_weights = alexscat['state_dict']['module.first_layer.0.weight'].cpu().numpy()
tensor = np.swapaxes(conv1_weights,1,3)
plot_kernels(tensor)


alexscat = torch.load('checkpoints/cifar100/alexscat_j2l3_-2layers/model_best.pth.tar')
conv2_weights = alexscat['state_dict']['module.features.1.weight'].cpu().numpy()
tensor = np.swapaxes(conv2_weights,1,3)
plot_kernels(tensor, num_cols=20)


best_alexnet = torch.load('model_best.pth.tar')
conv2_weights = best_alexnet['state_dict']['module.features.3.weight'].cpu().numpy()
tensor = np.swapaxes(conv2_weights,1,3)
plot_kernels(tensor, num_cols=20)

# print("NOW SHOWING SCATTERING VERSION")

# for i in range(tensor.shape[0]):
#     minned = tensor[i] - np.min(tensor[i])
#     norm = minned/np.max(minned)
#     plt.imshow(norm)
#     plt.show()

#vis = Visdom()
#vis.image(conv1_weights[0])