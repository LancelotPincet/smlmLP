#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np
import numba as nb
from smlmlp import analysis



@analysis(df_name="detections")
def lost_channels(fr, *, cuda=False, parallel=False) :
    """
    Infer channel numbers from reset frame coordinates.

    Parameters
    ----------
    fr : array-like
        Frame coordinate ordered by acquisition.
    cuda, parallel : bool, optional
        Execution options accepted by all analysis functions.

    Returns
    -------
    channel : ndarray
        One-based inferred channel index.
    info : dict
        Empty diagnostics dictionary.
    """
    channel = fr2ch(fr)
    info = {}
    return channel, info



@nb.njit(cache=True)
def fr2ch(fr) :
    """Convert reset frame coordinates to channel identifiers."""
    channel = np.empty_like(fr, dtype=np.uint8)
    channel[0] = 1
    for i in range(1, len(fr)) :
        if fr[i] < fr[i-1] :
            channel[i] = channel[i-1] + 1
        else :
            channel[i] = channel[i-1]
    return channel
