#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_transform, kernel, get_xp, transform_matrix, compress
import numpy as np



# %% Function
@block()
def registrate_find_matrix(optimized, /, matrix=None, *, channels_pixels_nm=1., cuda=False, parallel=False) :
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
    scales_x = [ref/pix[1] for ref, pix in zip(ref_pix[1], channels_pixels_nm)]
    scales_y = [ref/pix[0] for ref, pix in zip(ref_pix[0], channels_pixels_nm)]

    new_optimized = []
    for i in range(len(channels)) :
        optimize = None if optimized is None else xp.asarray(optimized[i])
        channel = xp.asarray(channels[i])
        matrix = transform_matrix(channel, scalex=scales_x[i], scaley=scales_y[i])
        optimize = img_transform(channel, matrix=matrix, out=optimize, cuda=cuda, parallel=parallel, stacks=True)
        optimize = compress(optimize, out=optimize, stacks=True, white_percent=1, black_percent=1, saturate=True)
        new_optimized.append(optimize)

    return new_optimized