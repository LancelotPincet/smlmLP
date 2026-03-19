#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp, img_correlate



# %% Function
@block()
def detec_snr(channels, bkgds, /, k_noise, *, cuda=False, parallel=False) :
    '''
    This function applyies a spatial filter to enhance signal.
    '''

    # xp
    xp = get_xp(cuda)

    # Correct signal length for end of acquisition
    if signals is not None and len(signals[0]) > len(channels[0]):
        signals = [signal[:len(channel)] for channel, signal in zip(channels, signals)]
    
    new_signals = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        signal = None if signals is None else signals[i]
        channel = channels[i]
        if bkgd is not None :
            channel = channel - bkgd
        kernel = xp.asarray(spatial_kernel[i])
        new_signal = img_correlate(channel, kernel=kernel, out=signal, cuda=cuda, parallel=parallel, stacks=True)
        new_signal /= xp.sqrt(xp.sum(kernel**2)) # correction factor for each different kernel
        new_signals.append(new_signal)

    return new_signals