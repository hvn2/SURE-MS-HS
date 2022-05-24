import torch
import numpy as np
from utils.common_utils import *
import matplotlib.pyplot as plt
dtype = torch.cuda.FloatTensor

def get_hs_psf(N=7,n_channels=93, gaussian=True):
    '''Assume all HS bands have the same psf
    which is Gaussian with sigma=1.'''
    psf = torch.zeros(n_channels,N,N)
    if gaussian:
        for i in range(n_channels):
            psf[i,:,:]=gaussian_filter(N=N,sigma=1.)
    else:
        for i in range(n_channels):
            psf[i,:,:]=get_filter()
    return psf