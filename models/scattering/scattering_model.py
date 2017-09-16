import tensorflow as tf
from libs.scattering import scattering

# can only run on a gpu
# requires NCHW format (cuDNN default - tf is NHWC)
def build_model(im_shape):
    placeholder = tf.placeholder(tf.float32, (None,) + im_shape)
    # M, N: input image size
    M, N = placeholder.shape.as_list()[-2:]
    print("M", M, "N", N)
    # J: number of layers
    scat = scattering.Scattering(M=M, N=N, J=1)(placeholder)
    return placeholder, scat
