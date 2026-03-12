#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import img_greyopening, kernel



# %% Function
@block()
def spatial_localmini(channels, localmini_radius=5, channel_pixel=1., pad=0, bkgds=None, cuda=False, parallel=False) :
    '''
    This function creates the top hat background.
    '''

    # Get local sigma in pixel
    pixel = Config(nfiles=len(channels), pixel=channel_pixel).pixel

    # Correct bkgd length for end of acquisition
    if bkgds is not None and len(bkgds[0]) > len(channels[0]) - 2*pad:
        bkgds = [bkgd[:len(channel) - pad] for channel, bkgd in zip(channels, bkgds)]
    
    #Footprint
    footprints = [kernel(window=2*localtophat_radius, pixel=pix) for pix in pixel]

    new_bkgds = []
    for i in range(len(channels)) :
        bkgd = None if bkgds is None else bkgds[i]
        channel = channels[i]
        channel = channel[pad: len(channel) - 2*pad] if bkgd is None else channel[pad: pad + len(bkgd)]
        new_bkgd = img_greyopening(channel, footprint=footprints[i], out=bkgd, cuda=cuda, parallel=parallel, stacks=True)
        new_bkgds.append(new_bkgd)

    return new_bkgds