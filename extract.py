import os, sys, time, subprocess, h5py, argparse, logging
import numpy as np
import tensorflow as tf
import data_utils
from os.path import join as oj


def get_args():
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--device', type=str, default='/cpu:0',
                        help='an integer for the accumulator')
    return parser.parse_args()


# params
args = get_args()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
model_name = "c3d"  # alexnet, scattering, c3d
layer = "conv2"  # chooses layer
num_frames_per_clip = 3  # for c3d only, must generate more than 1 clip!
device = args.device  # '/cpu:0', '/gpu:0'
data_dir = "/scratch/users/vision/reza/v4"
name = layer  # this can be anything

# indirect params
out_dir = oj("/scratch/users/vision/chandan/out", model_name + \
             "_" + name + "_" + time.strftime("%b%d_%H:%M:%S"))
np.random.seed(13)

# load data
NUM_IMS = 20000
chunk_size = 10
im_ranges_list = [(x, x + chunk_size) for x in np.arange(0, NUM_IMS, chunk_size)]

# choose model
ims, _ = data_utils.load_data(data_dir, im_range=im_ranges_list[0])
ims = data_utils.preprocess_data(ims=ims)
if model_name == "alexnet":  # alexnet alone
    from models.alexnet.alexnet_model import build_model

    placeholder, model = build_model(ims.shape[1:])
    model = model[layer]
elif model_name == "scattering":  # scattering alone
    from models.scattering.scattering_model import build_model

    ims = np.transpose(ims, (0, 3, 1, 2))  # convert NHWC -> NCHW
    placeholder, model = build_model(ims.shape[1:])
elif model_name == "c3d":  # c3d
    from models.c3d.c3d_model import build_model

    # convert NHWC -> N,N_frames,H,W,C
    # note that N get reduced by (num_frames_per_clip-1) - must throw these away for regression part
    ims = data_utils.sliding_window(ims,
                                    (num_frames_per_clip, ims.shape[1], ims.shape[2], ims.shape[3]),
                                    ss=(1, ims.shape[1], ims.shape[2], ims.shape[3]))
    print('ims reshaped size', ims.shape)
    placeholder, model = build_model(ims.shape)
    model = model[layer]

# extract features
for i in range(len(im_ranges_list)):
    im_range = im_ranges_list[i]
    ims, _ = data_utils.load_data(data_dir, im_range=im_range)
    ims = data_utils.preprocess_data(ims=ims)

    if model_name == "scattering":
        ims = np.transpose(ims, (0, 3, 1, 2))  # convert NHWC -> NCHW
        ims = np.zeros(ims.shape)
    elif model_name == "c3d":
        ims = data_utils.sliding_window(ims,
                                        (num_frames_per_clip, ims.shape[1], ims.shape[2], ims.shape[3]),
                                        ss=(1, ims.shape[1], ims.shape[2], ims.shape[3]))


    def extract_features(placeholder, model, ims):
        print('ims.shape', ims.shape)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        t = time.time()
        output = sess.run(model, feed_dict={placeholder: ims})
        print('features.shape', output.shape)
        return output


    with tf.device(device):
        features = extract_features(placeholder=placeholder, model=model, ims=ims)

    # save features / copy main file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    subprocess.call('cp /accounts/projects/vision/chandan/v4_natural_movies/extract.py ' +
                    out_dir, shell=True)
    with h5py.File(oj(out_dir, 'features.h5'), 'a') as f:
        f.create_dataset(str(i), data=features)

    logging.info('succesfully completed %s', im_range)

with h5py.File(oj(out_dir, 'features.h5'), 'a') as f:
    metadata = f.create_dataset("im_ranges_list", data=im_ranges_list)
    metadata.attrs['model_name'] = model_name
    metadata.attrs['layer'] = layer
    metadata.attrs['num_frames_per_clip'] = num_frames_per_clip
    metadata.attrs['chunk_size'] = chunk_size
print('completed all')
