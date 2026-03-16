#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import get_xp, temporal_correlate



# %% Function
@block()
def signal_temporalfilter(channels, /, temporal_kernel, signals=None, bkgds=None, *, pad=0, cuda=False, parallel=False) :
    '''
    This function applyies a temporal filter to enhance signal.
    '''

    # xp
    xp = get_xp(cuda)

    # Correct signal length for end of acquisition
    if signals is not None and len(signals[0]) > len(channels[0]) - 2*pad:
        signals = [signal[:len(channel) - pad] for channel, signal in zip(channels, signals)]
    
    new_signals = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        signal = None if signals is None else signals[i]
        channel = channels[i]
        channel = channel[pad: len(channel) - 2*pad] if signal is None else channel[pad: pad + len(signal)]
        if bkgd is not None : signal = xp.substract(channel, bkgd, out=signal)
        kernel = xp.asarray(temporal_kernel[i])
        new_signal = temporal_correlate(channel, kernel, out=signal, cuda=cuda, parallel=parallel)
        new_signals /= xp.sqrt(xp.sum(kernel**2)) # correction factor for each different kernel
        new_signals.append(new_signal)

    return new_signals