#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np
import numba as nb
from smlmlp import analysis



@analysis(df_name="detections")
def lost_frames(pix, *, cuda=False, parallel=False) :
    """
    Infer frame numbers from reset pixel coordinates.

    Parameters
    ----------
    pix : array-like
        Pixel coordinate ordered by acquisition.
    cuda, parallel : bool, optional
        Execution options accepted by all analysis functions.

    Returns
    -------
    frame : ndarray
        One-based inferred frame index.
    info : dict
        Empty diagnostics dictionary.
    """
    frame = pix2fr(pix)
    info = {}
    return frame, info

@nb.njit(cache=True)
def pix2fr(pix) :
    """Convert reset pixel coordinates to frame identifiers."""
    frame = np.empty_like(pix, dtype=np.uint32)
    frame[0] = 1
    for i in range(1, len(pix)) :
        if pix[i] < pix[i-1] :
            frame[i] = frame[i-1] + 1
        else :
            frame[i] = frame[i-1]
    return frame
