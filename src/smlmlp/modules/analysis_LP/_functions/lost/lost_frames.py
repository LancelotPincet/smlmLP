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
def lost_frames(pix, *, cuda=False, parallel=False) :
    '''
    Analysis for defining frame number with given pixel coordinate.
    Assumption made is that localizations are sorted from smallest pix to biggest.
    When pix is decreasing it means we changed frame.
    '''
    return pix2fr(pix)

@nb.njit(cache=True)
def pix2fr(pix) :
    frame = np.empty_like(pix, dtype=np.uint32)
    frame[0] = 1
    for i in range(1, len(pix)) :
        if pix[i] < pix[i-1] :
            frame[i] = frame[i-1] + 1
        else :
            frame[i] = frame[i-1]
    return frame

