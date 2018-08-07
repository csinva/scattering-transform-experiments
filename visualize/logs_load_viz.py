import numpy as np
import os, sys, time, subprocess, h5py, argparse, logging, pickle, random
from os.path import join as oj
import pandas as pd
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from cycler import cycler
from math import ceil
cm = plt.get_cmap('BuGn')



# load in data from logs
def load_results(path_to_results):
    results = {}
    for dirname in os.listdir(path_to_results):
        if not '.png' in dirname and not 'Icon' in dirname and not '.DS_S' in dirname and not '.md' in dirname:
#             try:
            d = {}
            x = np.loadtxt(oj(path_to_results, dirname, 'log.txt'), skiprows=1)
            d['train'] = x[:, -2]
            d['val'] = x[:, -1]
            filter_name = [x for x in os.listdir(oj(path_to_results, dirname)) if 'max_act' in x]
            if len(filter_name) == 1:
                d['filters'] = imageio.imread(oj(path_to_results, dirname, filter_name[0]))
            results[dirname] = d
#             except:
#                 pass
    return results


def plot_train_val_curves(keys, results, color='normal', show=True):
    plt.figure(figsize=(8, 4), facecolor='white')
    
    # plot train
    ax = plt.subplot(121)
    if color == 'continuous':
        ax.set_prop_cycle(cycler('color', [cm(k) for k in np.linspace(0.1, .9, len(keys))]))
    plt.title('Train')
    for key in keys:
        plt.plot(results[key]['train'], label=key)


    # plot val
    ax = plt.subplot(122)
    if color == 'continuous':
        ax.set_prop_cycle(cycler('color', [cm(k) for k in np.linspace(0.1, .9, len(keys))]))
    plt.title('Test')
    for key in keys:
        plt.plot(results[key]['val'], label=key)
    plt.legend(ncol=1, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if show:
        plt.show()