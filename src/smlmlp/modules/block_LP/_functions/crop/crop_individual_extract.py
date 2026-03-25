#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import sortloop, get_xp, nb_threads
import numpy as np
import numba as nb
from numba import cuda as nb_cuda



# %% Function
@block()
def crop_individual_extract(channels, /, fr, x, y, ch=None, *, channels_crops_pix=11, cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''
    assert fr.min() >= 1, "Frame column starts at 1 by convention, please add 1 to your column frame before inserting it in this function"

    # Correct channels
    if ch is None :
        if len(channels) > 1 :
            raise SyntaxError("Cannot apply crop extracting on several channels without defining channel vector")
        ch = np.zeros_like(fr, dtype=np.uint8)

    # Get pixel
    channels_pixels_nm = Config(ncameras=len(channels), cameras_pixels_nm=channels_pixels_nm).cameras_pixels_nm

    # Get crop_pix
    try :
        if len(channels_crops_pix) != len(channels) :
            raise ValueError('crop_pix does not have the same length as channels')
    except TypeError :
        channels_crops_pix = [channels_crops_pix for _ in range(len(channels))]

    # Sort
    xp = get_xp(cuda)
    argsort = xp.lexsort((x, y, fr, ch))
    fr = xp.asarray(fr[argsort]-1, dtype=xp.uint32)
    y = xp.asarray(y[argsort], dtype=xp.float32)
    x = xp.asarray(x[argsort], dtype=xp.float32)
    ch = xp.asarray(ch[argsort], dtype=xp.uint8)

    # Looping on channels
    crops, X0, Y0 = [], [], []
    for ch_ch, ch_fr, ch_y, ch_x in sortloop(ch, fr, y, x) :
        channel = channels[ch_ch[0]]
        pixel = channels_pixels_nm[ch_ch[0]]
        width, height = channels_crops_pix[ch_ch[0]]
        n = len(ch_ch)
        crop = xp.empty(shape=(n, height, width), dtype=np.float32)
        x0_pix = xp.empty(shape=n, dtype=np.uint16)
        y0_pix = xp.empty(shape=n, dtype=np.uint16)

        if cuda :
            threads_per_block = (8, 8, 8)
            blocks_per_grid = (
                (n + threads_per_block[0] - 1) // threads_per_block[0],
                (height + threads_per_block[1] - 1) // threads_per_block[1],
                (width + threads_per_block[2] - 1) // threads_per_block[2],
            )
            crop_gpu[blocks_per_grid, threads_per_block](channel, crop, ch_fr, ch_x, ch_y, pixel[1], pixel[0], width, height, x0_pix, y0_pix)
        else :
            with nb_threads(parallel) :
                crop_cpu(channel, crop, ch_fr, ch_x, ch_y, pixel[1], pixel[0], width, height, x0_pix, y0_pix)
        crops.append(crop)
        X0.append(x0_pix)
        Y0.append(y0_pix)

    return crops, X0, Y0



@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
def crop_cpu(channel, crop, F, X, Y, xpixel, ypixel, w, h, x0_pix, y0_pix) :
    YY, XX = channel[0].shape
    for i in nb.prange(len(crop)) :

        # bbox
        fr, x, y = F[i], X[i], Y[i]
        if w % 2 : #if odd
            xpix = int(x / xpixel + 0.5) # rounding
            x0 = xpix - w//2
        else : #if even
            xpix = int(x / xpixel) + 0.5 # rounding
            x0 = xpix - w//2 + 0.5
        if h % 2 : #if odd
            ypix = int(y / ypixel + 0.5) # rounding
            y0 = ypix - h//2
        else : #if even
            ypix = int(y / ypixel) + 0.5 # rounding
            y0 = ypix - h//2 + 0.5
        x0, y0 = int(x0), int(y0)

        # fill
        x0_pix[i] = x0
        y0_pix[i] = y0
        for yy in range(y0, y0 + h):
            for xx in range(x0, x0 + w):
                if 0 <= yy < YY and 0 <= xx < XX:
                    crop[i, yy-y0, xx-x0] = float(channel[fr, yy, xx])
                else :
                    crop[i, yy-y0, xx-x0] = 0.0



@nb_cuda.jit(fastmath=True)
def crop_gpu(channel, crop, F, X, Y, xpixel, ypixel, w, h, x0_pix, y0_pix) :
    i, dy, dx = nb_cuda.grid(3)
    n = crop.shape[0]
    if i < n and dy < h and dx < w:
        YY, XX = channel.shape[1], channel.shape[2]

        # bbox
        fr, x, y = F[i], X[i], Y[i]
        if w % 2 : #if odd
            xpix = int(x / xpixel + 0.5) # rounding
            x0 = xpix - w//2
        else : #if even
            xpix = int(x / xpixel) + 0.5 # rounding
            x0 = xpix - w//2 + 0.5
        if h % 2 : #if odd
            ypix = int(y / ypixel + 0.5) # rounding
            y0 = ypix - h//2
        else : #if even
            ypix = int(y / ypixel) + 0.5 # rounding
            y0 = ypix - h//2 + 0.5
        x0, y0 = int(x0), int(y0)

        # store origin (only once per crop)
        if dy == 0 and dx == 0:
            x0_pix[i] = x0
            y0_pix[i] = y0

        # global coordinates
        xx = x0 + dx
        yy = y0 + dy

        # bounds check
        if 0 <= yy < YY and 0 <= xx < XX:
            crop[i, dy, dx] = float(channel[fr, yy, xx])
        else:
            crop[i, dy, dx] = 0.0

