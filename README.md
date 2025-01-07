# Frank-Wolfe-algorithm-for-Grid-free-reconstruction-of-Dirac-impulses

This repository contains the code of my Master thesis "A Frank-Wolfe algorithm for Grid-free reconstruction of Dirac impulses".

In this work, we propose a novel polyatomic Frank-Wolfe algorithm to
solve the BLASSO optimization problem. 
The BLASSO is the continuous (i.e. off-the-grid) counterpart to the LASSO, used for sparse spikes inverse problems. 
This approach is a variation on the classical Frank–Wolfe algorithm and improves upon the so-called Sliding Frank–Wolfe 
by incorporates polyatomic update directions, allowing for
linear combinations of multiple atoms, thereby improving exploration of the
solution space and accelerating convergence. We evaluate the effectiveness of
this method in scenarios involving random Fourier observations and deconvolution.

The 4 main files shows how to use the algorithm in different scenarios:
- `main_convolution.py` : 1D deconvolution problem
- `main_fourier.py` : 1D Fourier problem
- `main_convolution_2D.py` : 2D deconvolution problem
- `main_fourier_2D.py` : 2D Fourier problem

We implemented both our polyatomic Frank-Wolfe algorithm and the Sliding Frank-Wolfe algorithm from the paper
[The Sliding Frank-Wolfe Algorithm and its Application to Super-Resolution Microscopy](https://arxiv.org/abs/1811.06416) for comparison.