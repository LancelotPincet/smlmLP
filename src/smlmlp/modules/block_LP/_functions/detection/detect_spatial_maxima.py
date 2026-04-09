#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import get_xp, nb_threads
import numba as nb
from numba import cuda as nb_cuda 
import numpy as np
import math



# %% Function
@block()
def detect_spatial_maxima(snrs, /, snr_thresh, channels_spatial_kernels, *, f0=0, channels_pixels_nm=1., cuda=False, parallel=False) :
    '''
    This function finds local maxima in thresholded areas.
    '''

    # xp
    xp = get_xp(cuda)

    # footprint
    k_fp = math.exp(-(0.47/0.21)**2 / 2) # Sparrow limit: 0.47 wl / NA, Gaussian 0.21 wl / NA
    footprints = [kernel > kernel.max() * k_fp for kernel in channels_spatial_kernels]
    
    # Get pixel
    channels_pixels_nm = Config(ncameras=len(snrs), channels_cameras_nm=channels_pixels_nm).channels_cameras_nm

    F, X, Y, C = [], [], [], []
    for pos, (snr, footprint, pixel) in enumerate(zip(snrs, footprints, channels_pixels_nm)) :
        snr = xp.asarray(snr)
        footprint = xp.asarray(footprint)

        # GPU detection
        if cuda :
            max_points = int(snr.size / xp.sum(footprint)) # upper bound
            fr_out = xp.empty(max_points, dtype=xp.int32)
            y_out = xp.empty(max_points, dtype=xp.float32)
            x_out = xp.empty(max_points, dtype=xp.float32)
            counter = xp.zeros(1, dtype=xp.int32)
            shape = snr.shape
            threads_per_block = (2, 16, 16)
            blocks_per_grid = (
                (shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                (shape[1]  + threads_per_block[1] - 1) // threads_per_block[1],
                (shape[2]  + threads_per_block[2] - 1) // threads_per_block[2],
            )
            det_gpu[blocks_per_grid, threads_per_block](snr, snr_thresh, footprint, fr_out, y_out, x_out, counter)
            n = xp.asnumpy(counter)[0]
            fr = xp.asnumpy(fr_out[:n])
            y = xp.asnumpy(y_out[:n])
            x = xp.asnumpy(x_out[:n])

        # CPU detection
        else :
            mask = xp.zeros_like(snr, dtype=xp.bool_)
            with nb_threads(parallel) :
                maxi_cpu(mask, snr, snr_thresh, footprint)
            fr, y_int, x_int = xp.nonzero(mask)
            y, x = xp.empty_like(y_int, dtype=xp.float32), xp.empty_like(x_int, dtype=xp.float32)

            if len(f) :
                with nb_threads(parallel) :
                    com_cpu(snr, fr, y_int, x_int, y, x)

        # Append
        fr += f0 + 1
        y *= pixel[0]
        x *= pixel[1]
        c = np.full_like(fr, fill_value=pos, dtype=np.uint8)
        argsort = np.lexsort((x, y, fr))
        F.append(fr[argsort])
        Y.append(y[argsort])
        X.append(x[argsort])
        C.append(c)

    return np.hstack(F), np.hstack(X), np.hstack(Y), np.hstack(C)



@nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
def maxi_cpu(mask, snr, snr_thresh, footprint):
    F, Y, X = snr.shape
    YY, XX = footprint.shape
    cy, cx = YY // 2, XX // 2

    for fr in nb.prange(F):
        frame = snr[fr]
        m = mask[fr]

        for y in range(Y):
            for x in range(X):

                val = frame[y, x]
                if val < snr_thresh:
                    continue

                is_max = True

                for dy in range(YY):
                    for dx in range(XX):

                        if not footprint[dy, dx]:
                            continue

                        ny = y + dy - cy
                        nx = x + dx - cx

                        if ny < 0 or ny >= Y or nx < 0 or nx >= X:
                            continue

                        if frame[ny, nx] > val:
                            is_max = False
                            break

                    if not is_max:
                        break

                if is_max:
                    m[y, x] = True



@nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
def com_cpu(snr, fr_idx, y_idx, x_idx, y_out, x_out):
    n = len(fr_idx)

    F, Y, X = snr.shape

    for i in nb.prange(n):
        fr = fr_idx[i]
        y = y_idx[i]
        x = x_idx[i]

        xnum = 0.0
        ynum = 0.0
        denom = 0.0

        for dy in range(-1, 2):
            for dx in range(-1, 2):

                ny = y + dy
                nx = x + dx

                if ny < 0 or ny >= Y or nx < 0 or nx >= X:
                    continue

                val = snr[fr, ny, nx]

                xnum += nx * val
                ynum += ny * val
                denom += val

        if denom > 0:
            x_out[i] = xnum / denom
            y_out[i] = ynum / denom
        else :
            x_out[i] = x
            y_out[i] = y



@nb_cuda.jit(fastmath=True, cache=True)
def det_gpu(snr, snr_thresh, footprint, fr_out, y_out, x_out, counter):
    fr, y, x = nb_cuda.grid(3)

    F, Y, X = snr.shape
    YY, XX = footprint.shape

    if fr >= F or y >= Y or x >= X:
        return

    val = snr[fr, y, x]
    if val < snr_thresh:
        return

    cy = YY // 2
    cx = XX // 2

    # ---- local max ----
    is_max = True

    for dy in range(YY):
        for dx in range(XX):

            if not footprint[dy, dx]:
                continue

            ny = y + dy - cy
            nx = x + dx - cx

            if ny < 0 or ny >= Y or nx < 0 or nx >= X:
                continue

            if snr[fr, ny, nx] > val:
                is_max = False
                break

        if not is_max:
            break

    if not is_max:
        return

    # ---- center of mass ----
    xnum = 0.0
    ynum = 0.0
    denom = 0.0

    for dy in range(-1, 2):
        for dx in range(-1, 2):

            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= Y or nx < 0 or nx >= X:
                continue

            v = snr[fr, ny, nx]
            xnum += nx * v
            ynum += ny * v
            denom += v

    if denom > 0:
        xf = xnum / denom
        yf = ynum / denom
    else:
        xf = x
        yf = y

    # ---- atomic append ----
    idx = nb_cuda.atomic.add(counter, 0, 1)

    fr_out[idx] = fr
    y_out[idx] = yf
    x_out[idx] = xf