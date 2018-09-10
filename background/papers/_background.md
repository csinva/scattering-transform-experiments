[TOC]

#  wavelets intro

- notes based on “An intro to wavelets” - Amara 
- has frequency and duration
- convolve w/ signal and see if they match
- set of complementary wavelets - decomposes data without gaps/overlap so that decomposition process is reversible
- better than Fourier for spikes / discontinuities
- translation covariant (not invariant)

## Fourier transform

- translate function in time domain to frequency domain
- *discrete fourier transform* - estimates Fourier transform from a finite number of its sampled points
- *windowed fourier transform* - chop signal into sections and analyze each section separately
- *fast fourier transform* - factors Fourier matrix into product of few sparse matrices

## wavelet comparison to Fourier

- both linear operations
- the inverse transform matrix is transpose of the original
- both localized in frequency
- wavelets are also localized in space
- makes many functions sparse in wavelet domain
- Fourier just uses sin/cos
- wavelet has infinite set of possible basis functions

## wavelet analysis

- must adopt a wavelet prototype function $\phi(x)$, called an *analyzing wavelet*=*mother wavelet*
- orthogonal wavelet basis: $\phi_{(s,l)} (x) = 2^{-s/2} \phi (2^{-s} x-l)$
- *scaling function* $W(x) = \sum_{k=-1}^{N-2} (-1)^k c_{k+1} \phi (2x+k)$ where $\sum_{k=0,N-1} c_k=2, \: \sum_{k=0}^{N-1} c_k c_{k+2l} = 2 \delta_{l,0}$
- one pattern of coefficients is smoothing and another brings out detail = called *quadrature mirror filter pair*
- there is also a fast discrete wavelet transform (Mallat)
- wavelet packet transform - basis of *wavelet packets* = linear combinations of wavelets
- *basis of adapted waveform* - best basis function for a given signal representation
- *Marr wavelet* - developed for vision
- differential operator and capable of being tuned to act at any desired scale