#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp, nb_threads
import numpy as np
import numba as nb
from numba import cuda as nb_cuda



# %% Function
@block()
def locs_individual_barycenter(crops, X0, Y0, /, *, channels_pixels_nm=1., cuda=False, parallel=False) :
    '''
    Calculates individual barycenter of crops.
    '''

    # Calculate 
    try :
        if len(channels_pixels_nm) != len(crops) :
            if len(channels_pixels_nm) == 2 :
                channels_pixels_nm = [channels_pixels_nm for _ in range(len(crops))]
            else :
                raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_pixels_nm = [(channels_pixels_nm, channels_pixels_nm) for _ in range(len(crops))]

    # Looping on crops
    xp = get_xp(cuda)
    Mux, Muy = [], []
    for crop, x0, y0, pix in zip(crops, X0, Y0, channels_pixels_nm) :
        crop = xp.asarray(crop)
        x0 = xp.asarray(x0)
        y0 = xp.asarray(y0)
        mux = xp.empty_like(x0, dtype=xp.float32)
        muy = xp.empty_like(y0, dtype=xp.float32)
        
        # Calculate displacement
        if cuda :
            threads_per_block = 128
            blocks_per_grid = len(crop)
            barycenter_gpu[blocks_per_grid, threads_per_block](crop, mux, muy)
        else :
            with nb_threads(parallel) :
                barycenter_cpu(crop, mux, muy)
        
        # Calculate localization
        mux += x0
        mux *= pix[1]
        muy += y0
        muy *= pix[0]
        if cuda :
            mux = xp.asnumpy(mux)
            muy = xp.asnumpy(muy)
        
        Mux.append(mux)
        Muy.append(muy)

    return np.hstack(Mux), np.hstack(Muy)



@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
def barycenter_cpu(crop, mux, muy):
    N, Y, X = crop.shape
    for i in nb.prange(N):
        cr = crop[i]
        ynum, xnum = 0., 0.
        denom = 0.
        for y in range(Y) :
            for x in range(X) :
                val = cr[y, x]
                ynum += y * val
                xnum += x * val
                denom += val
        if denom > 0 :
            mux[i] = xnum / denom
            muy[i] = ynum / denom
        else :
            mux[i] = (X-1) / 2
            muy[i] = (Y-1) / 2



@nb_cuda.jit(fastmath=True, cache=True)
def barycenter_gpu(crop, mux, muy):
    i = nb_cuda.blockIdx.x          # tile index
    t = nb_cuda.threadIdx.x         # thread index within tile
    bdim = nb_cuda.blockDim.x
