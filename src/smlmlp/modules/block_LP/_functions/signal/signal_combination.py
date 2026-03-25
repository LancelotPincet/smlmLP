#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, signal_spatial_filter, signal_temporal_filter
from arrlp import gc



# %% Function
@block(timeit=False)
def signal_combination(channels, /, spatial_kernel=None, temporal_kernel=None, signals=None, bkgds=None, noise_corrections=None, *, do_spatial_filter=True, do_temporal_filter=False, cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # init
    buffers = None

    if do_spatial_filter :
        assert spatial_kernel is not None
        gc()
        signals, noise_corrections = signal_spatial_filter(channels, spatial_kernel, signals=signals, noise_corrections=noise_corrections, bkgds=bkgds, cuda=cuda, parallel=parallel)
        if buffers is None :
            buffers = [signal.copy() for signal in signals]
        else :
            for i in range(len(channels)) :
                buffers[i][:] = signals[i]
        channels = buffers
        bkgds = None
    if do_temporal_filter :
        assert temporal_kernel is not None
        gc()
        signals, noise_corrections = signal_temporal_filter(channels, temporal_kernel, signals=signals, noise_corrections=noise_corrections, bkgds=bkgds, cuda=cuda, parallel=parallel)
        if buffers is None :
            buffers = [signal.copy() for signal in signals]
        else :
            for i in range(len(channels)) :
                buffers[i][:] = signals[i]
        channels = buffers
        bkgds = None
    gc()

    return signals, noise_corrections