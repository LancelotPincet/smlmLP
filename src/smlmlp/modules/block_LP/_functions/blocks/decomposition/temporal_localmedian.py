#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from stacklp import temporal_median



# %% Function
@block()
def temporal_localmedian(channels, localmedian_window=25, exposure=1., pad=0, bkgds=None, cuda=False, parallel=False) :
    '''
    This function creates the temporal median background.
    '''

    # Get temporal window in frames
    localmedian_window = int(round(localmedian_window / exposure))

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]) - 2*pad:
        bkgds = [bkgd[:len(channel) - pad] for channel, bkgd in zip(channels, bkgds)]
    
    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        new_bkgd = temporal_median(channel, localmedian_window, out=bkgd, pad=pad, cuda=cuda, parallel=parallel)
        new_bkgds.append(new_bkgd)

    return new_bkgds