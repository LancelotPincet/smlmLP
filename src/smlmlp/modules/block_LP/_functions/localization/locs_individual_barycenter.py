#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block, crop_remove_bkgd
from arrlp import get_xp, nb_threads
import numpy as np
import numba as nb
from numba import cuda as nb_cuda
from ._channel_values import split_channel_origins, stack_channel_values

BARYCENTER_GPU_MAX_THREADS = 256



@block()
def locs_individual_barycenter(
    crops,
    X0,
    Y0,
    /,
    ch=None,
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
    X0 : array-like
        Detection-aligned 1D vector of x-origin pixel indices.
    Y0 : array-like
        Detection-aligned 1D vector of y-origin pixel indices.
    ch : array-like or None, optional
        One-based channel index for each detection. Required when ``crops`` has
        several channels.
    channels_pixels_nm : float or sequence, optional
        Pixel size specification. It can be a scalar, a ``(py, px)`` tuple,
        or a per-channel sequence.
    cuda : bool, optional
        Whether to execute the computation on GPU.
    parallel : bool, optional
        Whether to enable CPU parallelization.

    Returns
    -------
    tuple
        A tuple ``(mux, muy, info)`` where:

        - ``mux`` is the detection-aligned x localization array in nanometers,
        - ``muy`` is the detection-aligned y localization array in nanometers,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'channels_pixels_nm'``
            Normalized per-channel pixel sizes used for coordinate conversion.

    Notes
    -----
    1. ``ch`` is converted into per-channel positions and used to split ``X0``
       and ``Y0`` so each origin vector matches the corresponding crop stack.
    2. Each crop barycenter is computed in local pixel coordinates; zero-sum
       crops fall back to the crop center.
    3. Local barycenters are shifted by the crop origins, converted to
       nanometers with the channel pixel size, and remapped to detection order.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(3, 5, 5).astype(np.float32)]
    >>> x0 = np.array([10, 20, 30], dtype=np.float32)
    >>> y0 = np.array([5, 15, 25], dtype=np.float32)
    >>> mux, muy, info = locs_individual_barycenter(crops, x0, y0)
    >>> mux.shape == muy.shape
    True

    >>> pix = [(100.0, 120.0)]
    >>> mux, muy, info = locs_individual_barycenter(crops, x0, y0, channels_pixels_nm=pix)
    >>> mux.ndim
    1
    """
    n_channels = len(crops)
    X0, Y0, positions = split_channel_origins(crops, X0, Y0, ch, cuda=cuda)
    channels_pixels_nm = _normalize_channels_pixels_nm(channels_pixels_nm, n_channels)

    cuda = bool(cuda)
    xp = get_xp(cuda)
    mux_all = []
    muy_all = []

    new_crops, _bkgd_info = crop_remove_bkgd(crops, cuda=cuda, parallel=parallel)

    for crop, x0, y0, pixel in zip(new_crops, X0, Y0, channels_pixels_nm):
        crop = xp.asarray(crop)
        x0 = xp.asarray(x0)
        y0 = xp.asarray(y0)

        mux = xp.empty_like(x0, dtype=xp.float32)
        muy = xp.empty_like(y0, dtype=xp.float32)

        if len(crop) == 0:
            if cuda:
                mux = xp.asnumpy(mux)
                muy = xp.asnumpy(muy)
            mux_all.append(mux)
            muy_all.append(muy)
            continue

        if cuda:
            threads_per_block = _barycenter_gpu_threads(crop.shape[1], crop.shape[2])
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

    info = {
        "channels_pixels_nm": channels_pixels_nm,
    }

    return stack_channel_values(mux_all, positions), stack_channel_values(muy_all, positions), info



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



def _barycenter_gpu_threads(height, width):
    """Return a power-of-two CUDA block size for one crop reduction."""
    n_pixels = max(1, min(int(height) * int(width), BARYCENTER_GPU_MAX_THREADS))
    threads = 32
    while threads < n_pixels:
        threads *= 2
    return threads



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
    """Compute intensity barycenters on GPU for one crop stack."""
    i = nb_cuda.blockIdx.x
    t = nb_cuda.threadIdx.x
    bdim = nb_cuda.blockDim.x
    height = crop.shape[1]
    width = crop.shape[2]
    npixels = height * width

    xnum_cache = nb_cuda.shared.array(BARYCENTER_GPU_MAX_THREADS, dtype=nb.float32)
    ynum_cache = nb_cuda.shared.array(BARYCENTER_GPU_MAX_THREADS, dtype=nb.float32)
    denom_cache = nb_cuda.shared.array(BARYCENTER_GPU_MAX_THREADS, dtype=nb.float32)

    xnum = 0.0
    ynum = 0.0
    denom = 0.0
    for pixel_index in range(t, npixels, bdim):
        y = pixel_index // width
        x = pixel_index - y * width
        value = crop[i, y, x]
        xnum += x * value
        ynum += y * value
        denom += value

    xnum_cache[t] = xnum
    ynum_cache[t] = ynum
    denom_cache[t] = denom
    nb_cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if t < stride:
            xnum_cache[t] += xnum_cache[t + stride]
            ynum_cache[t] += ynum_cache[t + stride]
            denom_cache[t] += denom_cache[t + stride]
        nb_cuda.syncthreads()
        stride //= 2

    if t == 0:
        denom = denom_cache[0]
        if denom > 0.0:
            mux[i] = xnum_cache[0] / denom
            muy[i] = ynum_cache[0] / denom
        else:
            mux[i] = (width - 1) / 2.0
            muy[i] = (height - 1) / 2.0
