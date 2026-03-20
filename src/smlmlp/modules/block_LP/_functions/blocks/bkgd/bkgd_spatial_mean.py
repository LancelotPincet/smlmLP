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
def bkgd_spatial_mean(channels, /, mean_radius=7, bkgds=None, noise_corrections=None, *, channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # Get pixel
    pixel = Config(nfiles=len(channels), pixel=channel_pixel).pixel

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]
    
    # Noise corrections
    if noise_corrections is None :
        noise_corrections = [1. for _ in range(len(channels))]

    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        new_bkgd = img_gaussianfilter(channel, sigma=mean_radius/2, pixel=pixel[i], out=bkgd, cuda=cuda, parallel=parallel, stacks=True)
        k1 = -kernel(pixel=pixel[i][0], sigma=mean_radius/2)
        k2 = -kernel(pixel=pixel[i][1], sigma=mean_radius/2)
        k1[int(len(k1)//2)] += 1.
        k2[int(len(k2)//2)] += 1.
        noise_corrections[i] *= np.sqrt(np.sum(k1**2)) * np.sqrt(np.sum(k2**2))
        new_bkgds.append(new_bkgd)

    return new_bkgds, noise_corrections