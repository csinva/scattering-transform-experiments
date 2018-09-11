import sys
import scipy.io
import h5py
import numpy as np
import os, sys, time, subprocess, h5py, argparse, logging, pickle
import numpy as np
from PIL import Image
from os.path import join as oj
from scipy.ndimage import imread
from scipy.misc import imresize
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path                   
from torch.utils.data import Dataset
import pandas as pd

# dataset to access conditions / images
class Star_dset(Dataset):
    def __init__(self, star_dir='star_polymer'):
        self.conditions = pd.read_csv(oj(star_dir, 'star_polymer_conditions.csv'), delimiter=',')
        self.ims = [self.read_and_crop_tif(oj(star_dir, fname)) for fname in self.conditions['im_fname']]
        self.star_dir = star_dir

    def read_and_crop_tif(self, fname):
        im = Image.open(oj(fname + '.TIF'))
        imarray = np.array(im)[:, :, 0] # convert to grayscale
        # im_downsample = imresize(imarray, size=(imarray.shape[0]//8, imarray.shape[1]//8)) # downsample by 8
        im_downsample = imresize(imarray, size=(764, 915)) # downsample by 8
        im_cropped = im_downsample[20: 695, 102: 777]
        return im_cropped
    
    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        conditions_dict = self.conditions.loc[0].to_dict()
        im_dict = {'im': self.ims[idx]}
        return {**conditions_dict, **im_dict}

    
# dataset to access just images    
class Mixed_sam_dset(Dataset):
    def __init__(self, mixed_sam_dir='mixed_sam'):
        fnames = os.listdir(mixed_sam_dir)
        self.ims = [self.read_and_crop_tif(oj(mixed_sam_dir, fname)) for fname in fnames if '.TIF' in fname]
        self.mixed_sam_dir = mixed_sam_dir

    def read_and_crop_tif(self, fname):
        im = Image.open(oj(fname))
        imarray = np.array(im)[:, :, 0] # convert to grayscale
        im_downsample = imresize(imarray, size=(imarray.shape[0]//8, imarray.shape[1]//8)) # downsample by 8
        im_cropped = im_downsample[15: 338, 73: 396]
        return im_cropped
    
    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        return {'im': self.ims[idx]}