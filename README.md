# scattering transform
- *code for development of adaptive scattering transform that works with conv on small data*

# setup
- download cifar data by running download.py in data folder

# references
- *note*: actual weights have to be downloaded to be used
- uses AlexNet code from: https://github.com/guerzh/tf_weights
- uses scattering transform code from: https://github.com/tdeboissiere/DeepLearningImplementations

# usage
parameters - M, N, J, L=8 (hidden in filters_bank function)

- M, N - size of images
- J - number of scales (window size at any scale j: )
- L - number of orientations
