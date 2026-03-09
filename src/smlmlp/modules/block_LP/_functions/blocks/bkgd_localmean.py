#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_gaussianfilter



# %% Function
@block()
def bkgd_localmean(channels, local_sigma=4, pixel=1., pad=0, bkgds=None, cuda=False, parallel=False) :
    '''
    This function creates the temporal median background.
    '''

    # Get temporal window in frames
    local_sigma = int(round(local_sigma / pixel))

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]) - 2*pad:
        bkgds = [bkgd[:len(channel) - pad] for channel, bkgd in zip(channels, bkgds)]
    
    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        channel = channel[pad: len(channel) - 2*pad] if bkgd is None else channel[pad: pad + len(bkgd)]
        new_bkgd = img_gaussianfilter(channel, sigma=local_sigma, out=bkgd, cuda=cuda, parallel=parallel, stacks=True)
        new_bkgds.append(new_bkgd)

    return new_bkgds