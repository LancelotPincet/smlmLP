#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_transform, get_xp, transform_matrix, compress



# %% Function
@block()
def registrate_optimize_images(channels, /, *, channels_pixels_nm=1., cuda=False, parallel=False) :
    '''
    This function optimizes the raw images to maximize registration efficiency.
    '''

    xp = get_xp(cuda)

    # Calculates channels_pixels_nm
    try :
        if len(channels_pixels_nm) != len(channels) :
            if len(channels_pixels_nm) == 2 :
                channels_pixels_nm = [channels_pixels_nm for _ in range(len(channels))]
            else :
                raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_pixels_nm = [(channels_pixels_nm, channels_pixels_nm) for _ in range(len(channels))]

    # Reference pixel
    ref_pix = min([pix[0] for pix in channels_pixels_nm]), min([pix[1] for pix in channels_pixels_nm])
    scales_x = [ref_pix[1]/pix[1] for pix in channels_pixels_nm]
    scales_y = [ref_pix[0]/pix[0] for pix in channels_pixels_nm]

    # Optimize
    optimized = []
    for i in range(len(channels)) :
        channel = xp.asarray(channels[i])
        matrix = transform_matrix(channel, scalex=scales_x[i], scaley=scales_y[i])
        optimize = img_transform(channel, matrix=matrix, out=optimize, cuda=cuda, parallel=parallel, stacks=True)
        optimize = compress(optimize, out=optimize, white_percent=1, black_percent=1, saturate=True, stack=True)
        optimized.append(optimize)
    return optimized, ref_pix