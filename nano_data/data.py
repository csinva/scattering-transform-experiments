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
        self.ims = [self.read_and_crop_tif(oj(star_dir, fname)) for fname in self.conditions['Filename']]
        self.star_dir = star_dir

    def read_and_crop_tif(self, fname):
        im = Image.open(oj(fname + '.TIF'))
        imarray = np.array(im)
        im_downsample = imresize(imarray, size=(6112//8, 7320//8)) # downsample by 8
        im_cropped = im_downsample[20: 695, 102: 777]
        return im_cropped
    
    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        conditions_dict = self.conditions.loc[0].to_dict()
        im_dict = {'im': self.ims[idx]}
        return {**conditions_dict, **im_dict}
    


    


