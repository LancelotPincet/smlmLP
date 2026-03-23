#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from stacklp import temporal_median
import numpy as np



# %% Function
@block()
def bkgd_temporal_median(channels, /, median_window=25, bkgds=None, noise_corrections=None, *, exposure=1., cuda=False, parallel=False) :
    '''
    This function creates the temporal local median background.
    '''

    # Get temporal window in frames
    median_window = int(round(median_window / exposure))

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
        new_bkgd = temporal_median(channel, median_window, out=bkgd, cuda=cuda, parallel=parallel)
        noise_corrections[i] *= np.float32(1.)
        new_bkgds.append(new_bkgd)

    return new_bkgds, noise_corrections