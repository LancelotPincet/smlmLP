#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import img_greyopening, kernel



# %% Function
@block()
def bkgd_spatial_opening(channels, /, opening_radius=5, bkgds=None, noise_corrections=None, *, channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the spatial opening background.
    '''

    # Get local sigma in pixel
    pixel = Config(nfiles=len(channels), pixel=channel_pixel).pixel

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]
    
    # Noise corrections
    if noise_corrections is None :
        noise_corrections = [1. for _ in range(len(channels))]

    #Footprint
    footprints = [kernel(window=2*opening_radius, pixel=pix) for pix in pixel]

    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        new_bkgd = img_greyopening(channel, footprint=footprints[i], out=bkgd, cuda=cuda, parallel=parallel, stacks=True)
        noise_corrections[i] *= 1.
        new_bkgds.append(new_bkgd)

    return new_bkgds, noise_corrections