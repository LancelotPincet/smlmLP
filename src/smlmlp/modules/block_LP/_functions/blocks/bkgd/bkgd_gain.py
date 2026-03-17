#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import get_xp
import numba as nb
from numba import cuda
import math



# %% Function
@block()
def bkgd_gain(channels, /, n_iter=5, k_sigma=5., *, cuda=False, parallel=False) :
    '''
    This function calculates the gain maps.
    '''

    xp = get_xp(cuda)

    G, Mu, Sigma, N = [], [], [], []
    for channel in channels :
        channel = xp.asarray(channel)
        
        # GPU
        if cuda :

            # Allocate
            T, Y, X = channel.shape
            g = xp.empty(shape=(Y, X), dtype=np.float32)
            b = xp.empty(shape=(Y, X), dtype=np.float32)
            sig = xp.empty(shape=(Y, X), dtype=np.float32)
            n = xp.empty(shape=(Y, X), dtype=np.float32)

            # Call kernel
            tx, ty = threads_per_block
            bx = math.ceil(X / tx)
            by = math.ceil(Y / ty)
            blocks_per_grid = (bx, by)
            gain_gpu[blocks_per_grid, threads_per_block](channel, g, b, sig, n, n_iter, k_sigma)
            cuda.synchronize()

            # Bring back
            g = xp.asnumpy(g)
            b = xp.asnumpy(b)
            sig = xp.asnumpy(sig)
            n = xp.asnumpy(n)
            
        # CPU
        else :
            with nb_threads(parallel) :
                g, b, sig, n = gain_gpu(channel, n_iter, k_sigma)
            
        # Append
        G.append(g)
        B.append(b)
        Sigma.append(sig)
        N.append(n)
        
    return G, dict(background=B, noise=Sigma, npixels=N)



@nb.njit(cache=True, fastmath=True)
def _gain_cpu(trace: np.ndarray,
                n_iter: int,
                k_sigma: float) -> tuple[float, float, float, int]:
    """
    Estimate gain, background, and noise at a single pixel
    from its temporal trace.

    Parameters
    ----------
    trace   : 1D array of length T, raw ADU values
    n_iter  : number of outlier rejection iterations
    k_sigma : sigma threshold for flagging activation frames

    Returns
    -------
    G      : gain in e-/ADU
    B      : background mean in ADU
    sigma  : noise std in ADU = sqrt(B/G)
    n_clean: number of clean frames used
    """
    T = len(trace)

    # Initialize mask — all frames included
    # Use float as proxy for bool (numba limitation with bool arrays)
    mask = np.ones(T, dtype=nb.float32)

    mu = 0.0
    var = 0.0

    for iteration in range(n_iter):

        # Compute mean over masked frames
        total_weight = 0.0
        mu = 0.0
        for t in range(T):
            if mask[t] > 0.0:
                mu += trace[t]
                total_weight += 1.0

        if total_weight < 2.0:
            return 0.0, 0.0, 0.0, 0

        mu /= total_weight

        # Compute variance over masked frames
        var = 0.0
        for t in range(T):
            if mask[t] > 0.0:
                diff = trace[t] - mu
                var += diff * diff

        var /= (total_weight - 1.0)
        sigma_t = var ** 0.5

        if sigma_t < 1e-10:
            break

        # Update mask — flag frames deviating > k_sigma
        for t in range(T):
            diff = trace[t] - mu
            if diff < 0.0:
                diff = -diff
            if diff > k_sigma * sigma_t:
                mask[t] = 0.0

    # Final clean frame count
    n_clean = 0
    for t in range(T):
        if mask[t] > 0.0:
            n_clean += 1

    # Guard against degenerate case
    if var < 1e-10 or mu < 1e-10:
        return 0.0, mu, 0.0, n_clean

    G = mu / var
    sigma_out = (mu / G) ** 0.5 if G > 0.0 else 0.0

    return G, mu, sigma_out, n_clean



@nb.njit(parallel=True, cache=True, fastmath=True)
def gain_cpu(stack: np.ndarray,
                          n_iter: int = 5,
                          k_sigma: float = 5.0
                          ) -> tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray]:
    """
    Estimate per-pixel gain map from full SMLM stack.
    CPU parallel version using Numba prange.

    Parameters
    ----------
    stack   : (T, Y, X) float32 array, raw ADU values
    n_iter  : outlier rejection iterations
    k_sigma : activation flagging threshold in sigma units

    Returns
    -------
    G_map      : (Y, X) gain in e-/ADU
    B_map      : (Y, X) background mean in ADU
    sigma_map  : (Y, X) noise std in ADU
    n_clean_map: (Y, X) number of clean frames per pixel
    """
    T, Y, X = stack.shape

    G_map       = np.zeros((Y, X), dtype=nb.float32)
    B_map       = np.zeros((Y, X), dtype=nb.float32)
    sigma_map   = np.zeros((Y, X), dtype=nb.float32)
    n_clean_map = np.zeros((Y, X), dtype=nb.int32)

    for y in nb.prange(Y):
        for x in range(X):
            # Extract temporal trace for this pixel
            trace = stack[:, y, x].astype(nb.float64)

            G, B, sigma, n_clean = _gain_cpu(trace, n_iter, k_sigma)

            G_map[y, x]       = G
            B_map[y, x]       = B
            sigma_map[y, x]   = sigma
            n_clean_map[y, x] = n_clean

    return G_map, B_map, sigma_map, n_clean_map



@cuda.jit(device=True, fastmath=True)
def _gain_gpu(trace, T, n_iter, k_sigma, result):
    """
    Device function — runs on a single CUDA thread for one pixel.
    result: float32 array of length 4 → [G, B, sigma, n_clean]
    """
    # First pass: compute mean over all frames
    mu = 0.0
    count = 0.0
    for t in range(T):
        mu += trace[t]
        count += 1.0
    mu /= count

    # Iterative outlier rejection
    for iteration in range(n_iter):

        # Compute variance on current clean set
        var = 0.0
        n = 0.0
        for t in range(T):
            diff = trace[t] - mu
            var += diff * diff
            n += 1.0
        var /= (n - 1.0)
        sigma_t = math.sqrt(var) if var > 0.0 else 0.0

        if sigma_t < 1e-10:
            break

        # Recompute mean excluding outliers
        mu_new = 0.0
        n_clean = 0.0
        for t in range(T):
            diff = trace[t] - mu
            if diff < 0.0:
                diff = -diff
            if diff <= k_sigma * sigma_t:
                mu_new += trace[t]
                n_clean += 1.0

        if n_clean < 2.0:
            break

        mu = mu_new / n_clean

    # Final variance on clean frames
    var_final = 0.0
    n_final = 0.0
    for t in range(T):
        diff = trace[t] - mu
        if diff < 0.0:
            diff = -diff
        # Use last sigma_t as acceptance criterion
        if diff <= k_sigma * sigma_t:
            var_final += (trace[t] - mu) ** 2
            n_final += 1.0

    if n_final > 1.0:
        var_final /= (n_final - 1.0)

    # Compute outputs
    if var_final > 1e-10 and mu > 1e-10:
        G = mu / var_final
        sigma_out = math.sqrt(mu / G) if G > 0.0 else 0.0
    else:
        G = 0.0
        sigma_out = 0.0

    result[0] = G
    result[1] = mu
    result[2] = sigma_out
    result[3] = n_final



@cuda.jit(fastmath=True)
def gain_gpu(stack, G_map, B_map, sigma_map,
                     n_clean_map, n_iter, k_sigma):
    """
    CUDA kernel — one thread per pixel.

    stack      : (T, Y, X) float32
    G_map      : (Y, X) float32
    B_map      : (Y, X) float32
    sigma_map  : (Y, X) float32
    n_clean_map: (Y, X) float32
    """
    x, y = cuda.grid(2)
    T, Y, X = stack.shape

    if x >= X or y >= Y:
        return

    # Copy pixel trace into local array
    # Maximum trace length — adjust if needed
    MAX_T = 100000
    trace = cuda.local.array(MAX_T, dtype=nb.float32)

    for t in range(T):
        trace[t] = stack[t, y, x]

    # Temporary result array
    result = cuda.local.array(4, dtype=nb.float32)

    _gain_pixel_cuda(trace, T, n_iter, k_sigma, result)

    G_map[y, x]       = result[0]
    B_map[y, x]       = result[1]
    sigma_map[y, x]   = result[2]
    n_clean_map[y, x] = result[3]