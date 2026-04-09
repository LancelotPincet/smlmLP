#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import get_xp, nb_threads
import numba as nb
from numba import cuda as nb_cuda 
import math



# %% Function
@block()
def detect_snr(signals, bkgds, noise_corrections=None, channels_gains=0.25, *, cuda=False, parallel=False) :
    '''
    This function normalizes the signals into SNRs.
    '''

    # xp
    xp = get_xp(cuda)

    # lists
    channels_gains = Config(ncameras=len(signals), gain_experimental=channels_gains).channels_gains
    if noise_corrections is None :
        noise_corrections = [xp.float32(1.) for _ in range(len(signals))]

    snrs = []
    for i in range(len(signals)) :
        signal = xp.asarray(signals[i], dtype=xp.float32)
        bkgd = xp.asarray(bkgds[i], dtype=xp.float32)
        noise_correction = xp.float32(noise_corrections[i])
        gain = xp.float32(channels_gains[i])

        # Calculating SNR
        if cuda :
            threads_per_block = 256
            blocks_per_grid = (signal.size + threads_per_block - 1) // threads_per_block
            snr_gpu[blocks_per_grid, threads_per_block](signal.ravel(), bkgd.ravel(), noise_correction, gain)
        else :
            with nb_threads(parallel) :
                snr_cpu(signal.ravel(), bkgd.ravel(), noise_correction, gain)



@nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
def snr_cpu(signal, bkgd, noise_correction, gain) :
    for i in nb.prange(len(signal)) :
        signal[i] = nb.float32(signal[i] / noise_correction / math.sqrt(bkgd[i] / gain))



@nb_cuda.jit(fastmath=True, cache=True)
def snr_gpu(signal, bkgd, noise_correction, gain) :
    i = nb_cuda.grid(1)
    if i < len(signal) :
        signal[i] = nb.float32(signal[i] / noise_correction / math.sqrt(bkgd[i] / gain))