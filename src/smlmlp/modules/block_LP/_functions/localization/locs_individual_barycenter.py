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
def locs_individual_barycenter(
    crops,
    X0,
    Y0,
    /,
    *,
    channels_pixels_nm=1.0,
    cuda=False,
    parallel=False,
):
    """
    Compute individual barycenter localizations from image crops.

    For each crop stack, the function computes the intensity barycenter per
    event, adds the crop origin offsets, and converts coordinates to
    nanometers using the provided pixel sizes.

    Parameters
    ----------
    crops : sequence of array-like
        Sequence of crop stacks, one per channel, shaped ``(N, Y, X)``.
    X0 : sequence of array-like
        Sequence of x-origin pixel indices for each crop.
    Y0 : sequence of array-like
        Sequence of y-origin pixel indices for each crop.
    channels_pixels_nm : float or sequence, optional
        Pixel size specification. It can be a scalar, a ``(py, px)`` tuple,
        or a per-channel sequence.
    cuda : bool, optional
        Whether to execute the computation on GPU.
    parallel : bool, optional
        Whether to enable CPU parallelization.

    Returns
    -------
    tuple of ndarray
        A tuple ``(mux, muy)`` with all localized x and y coordinates in
        nanometers, concatenated across channels.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(3, 5, 5).astype(np.float32)]
    >>> x0 = [np.array([10, 20, 30], dtype=np.float32)]
    >>> y0 = [np.array([5, 15, 25], dtype=np.float32)]
    >>> mux, muy = locs_individual_barycenter(crops, x0, y0)
    >>> mux.shape == muy.shape
    True

    >>> pix = [(100.0, 120.0)]
    >>> mux, muy = locs_individual_barycenter(crops, x0, y0, channels_pixels_nm=pix)
    >>> mux.ndim
    1
    """
    n_channels = len(crops)
    channels_pixels_nm = _normalize_channels_pixels_nm(
        channels_pixels_nm,
        n_channels,
    )

    xp = get_xp(cuda)
    mux_all = []
    muy_all = []

    for crop, x0, y0, pixel in zip(crops, X0, Y0, channels_pixels_nm):
        crop = xp.asarray(crop)
        x0 = xp.asarray(x0)
        y0 = xp.asarray(y0)

        mux = xp.empty_like(x0, dtype=xp.float32)
        muy = xp.empty_like(y0, dtype=xp.float32)

        if cuda:
            threads_per_block = 128
            blocks_per_grid = len(crop)
            barycenter_gpu[blocks_per_grid, threads_per_block](crop, mux, muy)
        else:
            with nb_threads(parallel):
                barycenter_cpu(crop, mux, muy)

        mux += x0
        mux *= pixel[1]
        muy += y0
        muy *= pixel[0]

        if cuda:
            mux = xp.asnumpy(mux)
            muy = xp.asnumpy(muy)

        mux_all.append(mux)
        muy_all.append(muy)

    return np.hstack(mux_all), np.hstack(muy_all)



def _normalize_channels_pixels_nm(channels_pixels_nm, n_channels):
    """Normalize pixel sizes to one ``(py, px)`` tuple per channel."""
    try:
        if len(channels_pixels_nm) != n_channels:
            if len(channels_pixels_nm) == 2:
                channels_pixels_nm = [channels_pixels_nm for _ in range(n_channels)]
            else:
                raise ValueError(
                    "channel_mean_radius_pix does not have the same length as channels"
                )
    except TypeError:
        channels_pixels_nm = [
            (channels_pixels_nm, channels_pixels_nm)
            for _ in range(n_channels)
        ]

    return channels_pixels_nm



@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
def barycenter_cpu(crop, mux, muy):
    """Compute intensity barycenters on CPU for one crop stack."""
    n_events, height, width = crop.shape
    for i in nb.prange(n_events):
        event_crop = crop[i]
        ynum = 0.0
        xnum = 0.0
        denom = 0.0

        for y in range(height):
            for x in range(width):
                value = event_crop[y, x]
                ynum += y * value
                xnum += x * value
                denom += value

        if denom > 0:
            mux[i] = xnum / denom
            muy[i] = ynum / denom
        else:
            mux[i] = (width - 1) / 2
            muy[i] = (height - 1) / 2



@nb_cuda.jit(fastmath=True, cache=True)
def barycenter_gpu(crop, mux, muy):
    """GPU kernel placeholder matching the CPU barycenter interface."""
    i = nb_cuda.blockIdx.x
    t = nb_cuda.threadIdx.x
    bdim = nb_cuda.blockDim.x
