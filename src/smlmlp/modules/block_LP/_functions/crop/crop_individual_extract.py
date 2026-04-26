#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import sortloop, get_xp, nb_threads
import numpy as np
import numba as nb
from numba import cuda as nb_cuda



@block()
def crop_individual_extract(
    channels,
    /,
    fr,
    x_det,
    y_det,
    ch=None,
    *,
    channels_crops_pix=11,
    channels_pixels_nm=100.0,
    cuda=False,
    parallel=False,
):
    """
    Extract individual image crops centered on given coordinates.

    This function extracts per-event crops from multi-frame image channels
    using provided frame indices and spatial coordinates. Crops are grouped
    per channel and computed either on CPU or GPU.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel.
    fr : array-like
        Frame indices (starting at 1).
    x_det, y_det : array-like
        Channel-local detection coordinates in nanometers.
    ch : array-like or None, optional
        One-based channel indices for each event. If None, assumes one channel.
    channels_crops_pix : int or sequence, optional
        Crop size in pixels. Can be scalar, (h, w), or per-channel.
    channels_pixels_nm : float or sequence, optional
        Pixel size in nanometers. Can be scalar, (y, x), or per-channel values.
    cuda : bool, optional
        Whether to use GPU acceleration.
    parallel : bool, optional
        Whether to enable CPU parallelization.

    Returns
    -------
    tuple
        A tuple ``(crops, X0, Y0, info)`` where:

        - ``crops`` is a list of arrays containing extracted crops per channel,
        - ``X0`` is a list of x-origin pixel coordinates,
        - ``Y0`` is a list of y-origin pixel coordinates,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'channels_crops_pix'``
            Normalized per-channel crop sizes.
        ``'channels_pixels_nm'``
            Normalized per-channel pixel sizes.
        ``'argsort'``
            Sorting indices applied to inputs before processing.

    Examples
    --------
    >>> import numpy as np
    >>> channel = np.random.rand(10, 32, 32).astype(np.float32)
    >>> fr = np.array([1, 2, 3])
    >>> x = np.array([100., 150., 200.])
    >>> y = np.array([120., 180., 220.])
    >>> crops, X0, Y0, info = crop_individual_extract([channel], fr, x, y)
    >>> len(crops)
    1
    >>> len(crops[0])
    3
    """
    assert fr.min() >= 1, (
        "Frame column starts at 1 by convention, please add 1 to your column frame before inserting it in this function"
    )

    # Handle channel indices
    if ch is None:
        if len(channels) > 1:
            raise SyntaxError(
                "Cannot apply crop extracting on several channels without defining channel vector"
            )
        ch = np.ones_like(fr, dtype=np.uint8)

    # Normalize crop sizes per channel
    try:
        if len(channels_crops_pix) != len(channels):
            if len(channels_crops_pix) == 2:
                channels_crops_pix = [
                    channels_crops_pix for _ in range(len(channels))
                ]
            else:
                raise ValueError(
                    "channels_crops_pix does not have the same length as channels"
                )
    except TypeError:
        channels_crops_pix = [
            (channels_crops_pix, channels_crops_pix)
            for _ in range(len(channels))
        ]

    # Normalize pixel sizes per channel
    try:
        if len(channels_pixels_nm) != len(channels):
            if len(channels_pixels_nm) == 2:
                channels_pixels_nm = [
                    channels_pixels_nm for _ in range(len(channels))
                ]
            else:
                raise ValueError(
                    "channels_pixels_nm does not have the same length as channels"
                )
    except TypeError:
        channels_pixels_nm = [
            (channels_pixels_nm, channels_pixels_nm)
            for _ in range(len(channels))
        ]

    # Sorting inputs for efficient grouped processing
    xp = get_xp(cuda)
    fr = xp.asarray(fr, dtype=xp.uint32)
    y = xp.asarray(y_det, dtype=xp.float32)
    x = xp.asarray(x_det, dtype=xp.float32)
    ch = xp.asarray(ch, dtype=xp.uint8)

    if len(ch):
        _ch_min = int(ch.min().get() if hasattr(ch.min(), "get") else ch.min())
        _ch_max = int(ch.max().get() if hasattr(ch.max(), "get") else ch.max())
        if _ch_min < 1 or _ch_max > len(channels):
            raise ValueError("Channel indices must be one-based and within channels.")
        ch = ch - 1

    keys = xp.stack((x, y, fr, ch))
    argsort = xp.lexsort(keys)

    fr = fr[argsort] - 1  # convert to 0-based indexing
    y = y[argsort]
    x = x[argsort]
    ch = ch[argsort]

    crops, X0, Y0 = [], [], []

    # Loop over grouped channels
    for _, ch_ch, ch_fr, ch_y, ch_x in sortloop(ch, fr, y, x, cuda=cuda):
        channel = channels[ch_ch]
        pixel = channels_pixels_nm[ch_ch]
        width, height = channels_crops_pix[ch_ch]

        n = len(ch_fr)

        crop = xp.empty((n, height, width), dtype=np.float32)
        x0_pix = xp.empty(n, dtype=np.uint16)
        y0_pix = xp.empty(n, dtype=np.uint16)

        if cuda:
            threads_per_block = (8, 8, 8)
            blocks_per_grid = (
                (n + threads_per_block[0] - 1) // threads_per_block[0],
                (height + threads_per_block[1] - 1) // threads_per_block[1],
                (width + threads_per_block[2] - 1) // threads_per_block[2],
            )

            crop_gpu[blocks_per_grid, threads_per_block](
                channel,
                crop,
                ch_fr,
                ch_x,
                ch_y,
                pixel[1],
                pixel[0],
                width,
                height,
                x0_pix,
                y0_pix,
            )
        else:
            with nb_threads(parallel):
                crop_cpu(
                    channel,
                    crop,
                    ch_fr,
                    ch_x,
                    ch_y,
                    pixel[1],
                    pixel[0],
                    width,
                    height,
                    x0_pix,
                    y0_pix,
                )

        crops.append(crop)
        X0.append(x0_pix)
        Y0.append(y0_pix)

    info = {
        "channels_crops_pix": channels_crops_pix,
        "channels_pixels_nm": channels_pixels_nm,
        "argsort": argsort,
    }

    return crops, X0, Y0, info



@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
def crop_cpu(channel, crop, F, X, Y, xpixel, ypixel, w, h, x0_pix, y0_pix):
    """CPU implementation of crop extraction."""
    YY, XX = channel[0].shape

    for i in nb.prange(len(crop)):

        fr, x, y = F[i], X[i], Y[i]

        # Compute bounding box (handling even/odd sizes)
        if w % 2:
            xpix = int(x / xpixel + 0.5)
            x0 = xpix - w // 2
        else:
            xpix = int(x / xpixel) + 0.5
            x0 = xpix - w // 2 + 0.5

        if h % 2:
            ypix = int(y / ypixel + 0.5)
            y0 = ypix - h // 2
        else:
            ypix = int(y / ypixel) + 0.5
            y0 = ypix - h // 2 + 0.5

        x0, y0 = int(x0), int(y0)

        x0_pix[i] = x0
        y0_pix[i] = y0

        # Fill crop
        for yy in range(y0, y0 + h):
            for xx in range(x0, x0 + w):
                if 0 <= yy < YY and 0 <= xx < XX:
                    crop[i, yy - y0, xx - x0] = float(channel[fr, yy, xx])
                else:
                    crop[i, yy - y0, xx - x0] = 0.0



@nb_cuda.jit(fastmath=True)
def crop_gpu(channel, crop, F, X, Y, xpixel, ypixel, w, h, x0_pix, y0_pix):
    """GPU implementation of crop extraction."""
    i, dy, dx = nb_cuda.grid(3)
    n = crop.shape[0]

    if i < n and dy < h and dx < w:
        YY, XX = channel.shape[1], channel.shape[2]

        fr, x, y = F[i], X[i], Y[i]

        if w % 2:
            xpix = int(x / xpixel + 0.5)
            x0 = xpix - w // 2
        else:
            xpix = int(x / xpixel) + 0.5
            x0 = xpix - w // 2 + 0.5

        if h % 2:
            ypix = int(y / ypixel + 0.5)
            y0 = ypix - h // 2
        else:
            ypix = int(y / ypixel) + 0.5
            y0 = ypix - h // 2 + 0.5

        x0, y0 = int(x0), int(y0)

        if dy == 0 and dx == 0:
            x0_pix[i] = x0
            y0_pix[i] = y0

        xx = x0 + dx
        yy = y0 + dy

        if 0 <= yy < YY and 0 <= xx < XX:
            crop[i, dy, dx] = float(channel[fr, yy, xx])
        else:
            crop[i, dy, dx] = 0.0
