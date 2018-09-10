[TOC]

# people

- core
  - edouard oyallan
  - joan bruna
  - stephan mallat
- other
  - max welling

# goals

- benefits        
   - all filters are defined
   - more interpretable
   - more biophysically plausible
- scattering transform - computes a translation invariant repr. by cascading wavelet transforms and modulus pooling operators, which average the amplitude of iterated wavelet coefficients

# review-type
- mallat_16 "Understanding deep convolutional networks"
- vidal_17 "Mathematics of deep learning"
- bronstein_17 "Geometric deep learning: going beyond euclidean data"

# initial papers

- bruna_10 "classification with scattering operators"
- mallat_10 "recursive interferometric repr."
- mallat_12 "group invariant scattering"
  - introduce scat transform
- oyallan_13 "Generic deep networks with wavelet scattering"
- bruna_13 "Invariant Scattering Convolution Networks"
   - introduces the scattering transform implemented as a cnn
- anden_14 "Deep scattering spectrum"


## scat_conv
- oyallan_15 "Deep roto-translation scattering for object classification"
    - can capture rounded figures
    - can further impose robustness to rotation variability (although not full rotation invariance)
- cotter_17 "[Visualizing and improving scattering networks](https://arxiv.org/pdf/1709.01355.pdf)"
  - add deconvnet to visualize
- oyallan_18 "[Scattering Networks for Hybrid Representation Learning](https://hal.inria.fr/hal-01837587/document)"
    - using early layers scat is good enough
- oyallan_18 "i-RevNet: Deep Invertible Networks"
- oyallan_17 "[Scaling the scattering transform: Deep hybrid networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Oyallon_Scaling_the_Scattering_ICCV_2017_paper.pdf)"
    - use 1x1 convolutions to collapse accross channels
- jacobsen_17 "Hierarchical Attribute CNNs"
    - modularity
- cheng_16 "Deep Haar scattering networks"

## papers by other groups

- cohen_16 "[Group equivariant convolutional networks](http://www.jmlr.org/proceedings/papers/v48/cohenc16.pdf)"
  - introduce G-convolutions which share more wieghts than normal conv
- worrall_17 "[Interpretable transformations with encoder-decoder networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Worrall_Interpretable_Transformations_With_ICCV_2017_paper.pdf)"
  - look at interpretability
- bietti_17 "[Invariance and stability of deep convolutional representations](http://papers.nips.cc/paper/7201-invariance-and-stability-of-deep-convolutional-representations)"
  - theory paper

# nano papers

- yu_06 "A Nanoengineering Approach to Regulate the Lateral Heterogeneity of Self-Assembled Monolayers"
  - regulate heterogeneity of self-assembled monlayers
    - used nanografting + self-assembly chemistry
- bu_10 nanografting - makes more homogenous morphology
- fleming_09 "dendrimers"
  - scanning tunneling microscopy - provides highest spatial res
  - combat this for insulators
- lin_12_moire
  - prob moire effect with near-field scanning optical microscopy
- chen_12_crystallization