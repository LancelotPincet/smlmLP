#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp, nb_threads
import bottleneck as bn
import numba as nb
from numba import cuda as nb_cuda



# %% Function
@block()
def crop_remove_bkgd(crops, /, *, cuda=False, parallel=False) :
    '''
    This function removes the crops border median as background.
    '''

    # Looping on crops
    xp = get_xp(cuda)
    new_crops = []
    for crop in crops :
        crop = xp.asarray(crop)
        N, Y, X = crop.shape
        borders = xp.empty_like(crop, shape=(N, (Y-1 + X-1)*2)) # Remove 4 for corners
        
        if cuda :
            threads_per_block = 128
            blocks_per_grid = N
            borders_gpu[blocks_per_grid, threads_per_block](crop, borders)
            med = xp.median(borders, axis=1)
            crop = crop - med[:, None, None]
        else :
            with nb_threads(parallel) :
                borders_cpu(crop, borders)
            med = bn.median(borders, axis=1)
            crop = crop - med[:, None, None]
        new_crops.append(crop)

    return new_crops



@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
def borders_cpu(crop, borders):
    N, Y, X = crop.shape
    for i in nb.prange(N):
        cr = crop[i]
        i1 = 0
        i0, i1 = i1, i1 + Y - 1
        borders[i, i0:i1] = cr[:-1, 0]
        i0, i1 = i1, i1 + X - 1
        borders[i, i0:i1] = cr[-1, :-1]
        i0, i1 = i1, i1 + Y - 1
        borders[i, i0:i1] = cr[1:, -1]
        i0, i1 = i1, i1 + X - 1
        borders[i, i0:i1] = cr[0, 1:]



@nb_cuda.jit(fastmath=True, cache=True)
def borders_gpu(crop, borders):
    i = nb_cuda.blockIdx.x          # tile index
    t = nb_cuda.threadIdx.x         # thread index within tile
    bdim = nb_cuda.blockDim.x

    N = crop.shape[0]
    Y = crop.shape[1]
    X = crop.shape[2]

    if i >= N:
        return

    n_left   = Y - 1
    n_bottom = X - 1
    n_right  = Y - 1
    n_top    = X - 1
    total    = n_left + n_bottom + n_right + n_top

    for k in range(t, total, bdim):
        if k < n_left:
            borders[i, k] = crop[i, k, 0]

        elif k < n_left + n_bottom:
            kk = k - n_left
            borders[i, k] = crop[i, Y - 1, kk]

        elif k < n_left + n_bottom + n_right:
            kk = k - (n_left + n_bottom)
            borders[i, k] = crop[i, kk + 1, X - 1]

        else:
            kk = k - (n_left + n_bottom + n_right)
            borders[i, k] = crop[i, 0, kk + 1]