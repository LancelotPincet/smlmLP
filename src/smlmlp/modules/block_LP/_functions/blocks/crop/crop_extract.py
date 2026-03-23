#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import sortloop, get_xp
import numpy as np



# %% Function
@block()
def crop_individual_extract(channels, /, fr, x, y, ch=None, *, crop_size=11, channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # Correct channels
    if ch is None :
        if len(channels) > 1 :
            raise SyntaxError("Cannot apply crop extracting on several channels without defining channel vector")
        ch = np.zeros_likle(fr, dtype=np.uint8)

    # Get pixel
    fake_config = Config(nfiles=len(channels), pixel=channel_pixel, crop_size=crop_size)
    channel_pixel = fake_config.pixel
    crop_size = fake_config.crop_size

    # Sort
    xp = get_xp(cuda)
    argsort = np.lexsort((x, y, fr, ch))
    fr = xp.asarray(fr[argsort])
    y = xp.asarray(y[argsort])
    x = xp.asarray(x[argsort])
    ch = xp.asarray(ch[argsort])

    # Looping on channels
    crops, X, Y = [], [], []
    for ch_ch, ch_fr, ch_y, ch_x in sortloop(ch, fr, y, x) :
        channel = channels[ch_ch[0]]
        pixel = channel_pixel[ch_ch[0]]
        pixel = channel_pixel[ch_ch[0]]
        size = crop_size[ch_ch[0]]

    return 


@njit()
def bbox_cpu(bbox, X, Y, xpixel, ypixel, w, h) :
    for i in range(len(bbox)) :
        x, y = X[i], Y[i]
        if w % 2 : #if odd
            xpix = int(round(x/xpixel))
            x0, x1 = xpix - w//2, xpix + w//2
        else : #if even
            xpix = int(round(x/xpixel - 0.5)) +0.5
            x0, x1 = xpix - w//2 + 0.5, xpix + w//2 - 0.5
        if h % 2 : #if odd
            ypix = int(round(y/ypixel))
            y0, y1 = ypix - h//2, ypix + h//2
        else : #if even
            ypix = int(round(y/ypixel - 0.5)) +0.5
            y0, y1 = ypix - h//2 + 0.5, ypix + h//2 - 0.5
        bbox[i, 0] = int(y0)
        bbox[i, 1] = int(x0)
        bbox[i, 2] = int(y1) + 1
        bbox[i, 3] = int(x1) + 1

@cuda.jit()
def bbox_gpu(bbox, X, Y, xpixel, ypixel, w, h) :
    i = cuda.grid(1)
    if i < len(X) :
        x, y = X[i], Y[i]
        if w % 2 : #if odd
            xpix = int(round(x/xpixel))
            x0, x1 = xpix - w//2, xpix + w//2
        else : #if even
            xpix = int(round(x/xpixel - 0.5)) +0.5
            x0, x1 = xpix - w//2 + 0.5, xpix + w//2 - 0.5
        if h % 2 : #if odd
            ypix = int(round(y/ypixel))
            y0, y1 = ypix - h//2, ypix + h//2
        else : #if even
            ypix = int(round(y/ypixel - 0.5)) +0.5
            y0, y1 = ypix - h//2 + 0.5, ypix + h//2 - 0.5
        bbox[i, 0] = int(y0)
        bbox[i, 1] = int(x0)
        bbox[i, 2] = int(y1) + 1
        bbox[i, 3] = int(x1) + 1

