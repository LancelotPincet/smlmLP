#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp
from stacklp import temporal_correlate
import numpy as np



# %% Function
@block()
def signal_temporal_filter(channels, /, temporal_kernel, signals=None, bkgds=None, noise_corrections=None, *, cuda=False, parallel=False) :
    '''
    This function applyies a temporal filter to enhance signal.
    '''

    # Correct signal length for end of acquisition
    if signals is not None and len(signals[0]) > len(channels[0]):
        signals = [signal[:len(channel)] for channel, signal in zip(channels, signals)]

    # Noise corrections
    if noise_corrections is None :
        noise_corrections = [np.float32(1.) for _ in range(len(channels))]

    # Kernel
    kernel = temporal_kernel
    factor = np.sqrt(np.sum(kernel**2))

    new_signals = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        signal = None if signals is None else signals[i]
        channel = channels[i]
        if bkgd is not None :
            channel = channel - bkgd
        if signal is channel : signal = None
        new_signal = temporal_correlate(channel, kernel=kernel, out=signal, cuda=cuda, parallel=parallel)
        noise_corrections[i] *= factor
        new_signals.append(new_signal)

    return new_signals, noise_corrections