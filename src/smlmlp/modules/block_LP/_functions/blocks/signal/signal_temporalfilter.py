#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp
from stacklp import temporal_correlate



# %% Function
@block()
def signal_temporalfilter(channels, /, temporal_kernel, signals=None, bkgds=None, *, cuda=False, parallel=False) :
    '''
    This function applyies a temporal filter to enhance signal.
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
        if bkgd is not None : signal = xp.subtract(channel, bkgd, out=signal)
        kernel = xp.asarray(temporal_kernel[i])
        new_signal = temporal_correlate(signal, kernel=kernel, out=signal, cuda=cuda, parallel=parallel)
        new_signal /= xp.sqrt(xp.sum(kernel**2)) # correction factor for each different kernel
        new_signals.append(new_signal)

    return new_signals