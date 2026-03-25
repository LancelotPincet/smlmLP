#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import img_greyopening, kernel
import numpy as np



# %% Function
@block()
def bkgd_spatial_opening(channels, /, channels_opening_radius_pix=3., bkgds=None, noise_corrections=None, *, cuda=False, parallel=False) :
    '''
    This function creates the spatial opening background.
    '''

    # Get channel_mean_radius_pix
    try :
        if len(channels_opening_radius_pix) != len(channels) :
            if len(channels_opening_radius_pix) == 2 :
                channels_opening_radius_pix = [channels_opening_radius_pix for _ in range(len(channels))]
            else :
                raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_opening_radius_pix = [(channels_opening_radius_pix, channels_opening_radius_pix) for _ in range(len(channels))]

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]
    
    # Noise corrections
    if noise_corrections is None :
        noise_corrections = [np.float32(1.) for _ in range(len(channels))]

    #Footprint
    footprints = [kernel(window=(2*rad_pix[0], 2*rad_pix[1])) for rad_pix in channels_opening_radius_pix]

    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        new_bkgd = img_greyopening(channel, footprint=footprints[i], out=bkgd, cuda=cuda, parallel=parallel, stacks=True)
        noise_corrections[i] *= np.float32(1.)
        new_bkgds.append(new_bkgd)

    return new_bkgds, noise_corrections