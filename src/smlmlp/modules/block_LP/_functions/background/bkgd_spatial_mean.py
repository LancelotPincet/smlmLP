#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import img_gaussianfilter, kernel
import numpy as np



# %% Function
@block()
def bkgd_spatial_mean(channels, /, channel_mean_radius_pix=7., bkgds=None, noise_corrections=None, *, cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # Get channel_mean_radius_pix
    try :
        if len(channel_mean_radius_pix) != len(channels) :
            if len(channel_mean_radius_pix) == 2 :
                channel_mean_radius_pix = [channel_mean_radius_pix for _ in range(len(channels))]
            else :
                raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channel_mean_radius_pix = [(channel_mean_radius_pix, channel_mean_radius_pix) for _ in range(len(channels))]

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]
    
    # Noise corrections
    if noise_corrections is None :
        noise_corrections = [np.float32(1.) for _ in range(len(channels))]

    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        sigma = channel_mean_radius_pix[i][0] / 2, channel_mean_radius_pix[i][1] / 2
        new_bkgd = img_gaussianfilter(channel, sigma=sigma, out=bkgd, cuda=cuda, parallel=parallel, stacks=True)
        k1 = -kernel(ndims=1, sigma=sigma[0])
        k2 = -kernel(ndims=1, sigma=sigma[1])
        k1[int(len(k1)//2)] += 1.
        k2[int(len(k2)//2)] += 1.
        noise_corrections[i] *= np.sqrt(np.sum(k1**2)) * np.sqrt(np.sum(k2**2))
        new_bkgds.append(new_bkgd)

    return new_bkgds, noise_corrections