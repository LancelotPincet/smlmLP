#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, bkgd_spatial_opening, bkgd_temporal_median, bkgd_spatial_mean
from arrlp import gc, get_xp



# %% Function
@block(timeit=False)
def bkgd_combination(channels, /, bkgds=None, noise_corrections=None, *, do_spatial_opening=False, channels_opening_radius_pix=3., do_temporal_median=True, median_window_fr=25, do_spatial_mean=True, channel_mean_radius_pix=7., cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # init
    raws = channels
    buffers = None
    xp = get_xp(cuda)

    if do_spatial_opening :
        gc()
        bkgds, noise_corrections = bkgd_spatial_opening(channels, channels_opening_radius_pix=channels_opening_radius_pix, noise_corrections=noise_corrections, bkgds=bkgds, cuda=cuda, parallel=parallel)
        if buffers is None :
            channels = [channel - bkgd for channel, bkgd in zip(channels, bkgds)]
            buffers = channels
        else :
            for i in range(len(channels)) :
                xp.subtract(channels[i], bkgds[i], buffers[i])
    if do_temporal_median :
        gc()
        bkgds, noise_corrections = bkgd_temporal_median(channels, median_window_fr=median_window_fr, noise_corrections=noise_corrections, bkgds=bkgds, cuda=cuda, parallel=parallel)
        if buffers is None :
            channels = [channel - bkgd for channel, bkgd in zip(channels, bkgds)]
            buffers = channels
        else :
            for i in range(len(channels)) :
                xp.subtract(channels[i], bkgds[i], buffers[i])
    if do_spatial_mean :
        gc()
        bkgds, noise_corrections = bkgd_spatial_mean(channels, channel_mean_radius_pix=channel_mean_radius_pix, noise_corrections=noise_corrections, bkgds=bkgds, cuda=cuda, parallel=parallel)
        if buffers is None :
            channels = [channel - bkgd for channel, bkgd in zip(channels, bkgds)]
            buffers = channels
        else :
            for i in range(len(channels)) :
                xp.subtract(channels[i], bkgds[i], buffers[i])
    gc()
    if buffers is None :
        if bkgds is None :
            bkgds = [xp.zeros_like(channel) for channel in channels]
        else :
            for bkgd in bkgds :
                bkgd[:] = 0
    else :
        for i in range(len(channels)) :
            xp.subtract(raws[i], channels[i], bkgds[i])

    return bkgds, noise_corrections