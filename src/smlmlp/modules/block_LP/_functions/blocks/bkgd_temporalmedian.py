#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
import numpy as np
from stacklp import temporal_median



# %% Function
@block()
def bkgd_temporalmedian(*channels, temporal_window=25, pad=0, bkgds=None, cuda=False, parallel=False) :
    '''
    This function creates the temporal median background.
    '''

    # Correct bkgd length for end of acquisition
    if bkgds is not None :
        bkgds = [bkgd[:len(channel) - pad] for channel, bkgd in zip(channels, bkgds)]
    
    new_bkgds = []
    for i in range(len(channels)) :
        channel = channels[i]
        bkgd = None if bkgds is None else bkgds[i]
        new_bkgd = temporal_median(channel, temporal_window, out=bkgd, pad=pad, cuda=cuda, parallel=parallel)
        new_bkgds.append(new_bkgd)

    return new_bkgds