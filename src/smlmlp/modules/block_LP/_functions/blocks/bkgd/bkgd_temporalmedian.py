#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from stacklp import temporal_median



# %% Function
@block()
def bkgd_temporalmedian(channels, /, median_window=25, bkgds=None, *, exposure=1., pad=0, cuda=False, parallel=False) :
    '''
    This function creates the temporal local median background.
    '''

    # Get temporal window in frames
    median_window = int(round(median_window / exposure))

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]) - 2*pad:
        bkgds = [bkgd[:len(channel) - pad] for channel, bkgd in zip(channels, bkgds)]
    
    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        new_bkgd = temporal_median(channel, median_window, out=bkgd, pad=pad, cuda=cuda, parallel=parallel)
        new_bkgds.append(new_bkgd)

    return new_bkgds