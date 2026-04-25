#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import analysis
import numpy as np
import numba as nb



# %% Function
@analysis(df_name="detections")
def lost_channels(fr, *, cuda=False, parallel=False) :
    '''
    Analysis for defining channel number with given frame coordinate.
    Assumption made is that localizations are sorted from smallest frame to biggest.
    When fr is decreasing it means we changed channel.
    '''
    return fr2ch(fr)



@nb.njit(cache=True)
def fr2ch(fr) :
    channel = np.empty_like(fr, dtype=np.uint8)
    channel[0] = 1
    for i in range(1, len(pix)) :
        if fr[i] < fr[i-1] :
            channel[i] = channel[i-1] + 1
        else :
            channel[i] = channel[i-1]
    return channel

