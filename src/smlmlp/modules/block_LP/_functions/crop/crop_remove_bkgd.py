#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import get_xp, nb_threads
import warnings
import bottleneck as bn
import numba as nb
from numba import cuda as nb_cuda



@block(timeit=False)
def crop_remove_bkgd(crops, /, *, cuda=False, parallel=False):
    """
    Remove a per-crop background estimated from the border median.

    This function estimates the background of each crop from the median value
    of its border pixels, excluding duplicated corner contributions. The
    estimated border median is then subtracted from the full crop.

    Parameters
    ----------
    crops : sequence of ndarray
        Sequence of crop stacks. Each element is expected to have shape
        ``(n_crops, height, width)``.
    cuda : bool, optional
        Whether to use GPU acceleration.
    parallel : bool, optional
        Whether to enable CPU parallelization.

    Returns
    -------
    tuple
        A tuple ``(new_crops, info)`` where:

        - ``new_crops`` is the list of background-corrected crop stacks,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'border_medians'``
            List of 1D arrays containing the border median value used for each
            crop in each input stack.

    Notes
    -----
    For each crop, the border values are collected in the following order:

    1. left border excluding the last corner,
    2. bottom border excluding the last corner,
    3. right border excluding the first corner,
    4. top border excluding the first corner.

    This preserves the original logic while avoiding duplicate corner entries.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(4, 7, 7).astype(np.float32)]
    >>> new_crops, info = crop_remove_bkgd(crops)
    >>> len(new_crops)
    1
    >>> new_crops[0].shape
    (4, 7, 7)
    >>> len(info["border_medians"])
    1

    >>> crops = [
    ...     np.random.rand(3, 9, 9).astype(np.float32),
    ...     np.random.rand(2, 11, 11).astype(np.float32),
    ... ]
    >>> new_crops, info = crop_remove_bkgd(crops, cuda=False, parallel=True)
    >>> len(new_crops)
    2
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    new_crops = []
    border_medians = []

    for crop in crops:
        crop = xp.asarray(crop)
        n_crops, height, width = crop.shape
        pad_mask = edge_connected_zero_mask(crop == 0, xp)

        # Store the crop border values without duplicating the four corners.
        borders = xp.empty_like(
            crop,
            shape=(n_crops, (height - 1 + width - 1) * 2),
            dtype=xp.float32,
        )

        if cuda:
            threads_per_block = 128
            blocks_per_grid = n_crops

            borders_gpu[blocks_per_grid, threads_per_block](crop, pad_mask, borders)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                med = xp.nanmedian(borders, axis=1)
        else:
            with nb_threads(parallel):
                borders_cpu(crop, pad_mask, borders)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                med = bn.nanmedian(borders, axis=1)

        med = xp.where(xp.isnan(med), xp.float32(0.0), med)
        crop = crop - med[:, None, None]
        crop = xp.where(pad_mask, xp.float32(0.0), crop)

        new_crops.append(crop)
        border_medians.append(med)

    info = {
        "border_medians": border_medians,
    }

    return new_crops, info



def edge_connected_zero_mask(zero_mask, xp):
    """Return the mask of zero pixels connected to crop borders."""
    _, height, width = zero_mask.shape

    pad_mask = xp.zeros_like(zero_mask, dtype=bool)
    pad_mask[:, 0, :] = zero_mask[:, 0, :]
    pad_mask[:, -1, :] = zero_mask[:, -1, :]
    pad_mask[:, :, 0] = zero_mask[:, :, 0]
    pad_mask[:, :, -1] = zero_mask[:, :, -1]

    for _ in range(height + width):
        neighbors = xp.zeros_like(pad_mask, dtype=bool)
        neighbors[:, 1:, :] |= pad_mask[:, :-1, :]
        neighbors[:, :-1, :] |= pad_mask[:, 1:, :]
        neighbors[:, :, 1:] |= pad_mask[:, :, :-1]
        neighbors[:, :, :-1] |= pad_mask[:, :, 1:]
        pad_mask = pad_mask | (zero_mask & neighbors)

    return pad_mask



@nb.njit(cache=True, nogil=True, parallel=True)
def borders_cpu(crop, pad_mask, borders):
    """Fill the border buffer for each crop on CPU, masking padded border pixels."""
    n_crops, height, width = crop.shape

    for i in nb.prange(n_crops):
        i1 = 0

        for y in range(height - 1):
            v = crop[i, y, 0]
            borders[i, i1] = float("nan") if pad_mask[i, y, 0] else v
            i1 += 1

        for x in range(width - 1):
            v = crop[i, height - 1, x]
            borders[i, i1] = float("nan") if pad_mask[i, height - 1, x] else v
            i1 += 1

        for y in range(1, height):
            v = crop[i, y, width - 1]
            borders[i, i1] = float("nan") if pad_mask[i, y, width - 1] else v
            i1 += 1

        for x in range(1, width):
            v = crop[i, 0, x]
            borders[i, i1] = float("nan") if pad_mask[i, 0, x] else v
            i1 += 1



@nb_cuda.jit(cache=True)
def borders_gpu(crop, pad_mask, borders):
    """Fill the border buffer for each crop on GPU, masking padded border pixels."""
    i = nb_cuda.blockIdx.x
    t = nb_cuda.threadIdx.x
    bdim = nb_cuda.blockDim.x

    n_crops = crop.shape[0]
    height = crop.shape[1]
    width = crop.shape[2]

    if i >= n_crops:
        return

    n_left = height - 1
    n_bottom = width - 1
    n_right = height - 1
    n_top = width - 1
    total = n_left + n_bottom + n_right + n_top

    for k in range(t, total, bdim):
        if k < n_left:
            v = crop[i, k, 0]
            yb = k
            xb = 0

        elif k < n_left + n_bottom:
            kk = k - n_left
            v = crop[i, height - 1, kk]
            yb = height - 1
            xb = kk

        elif k < n_left + n_bottom + n_right:
            kk = k - (n_left + n_bottom)
            v = crop[i, kk + 1, width - 1]
            yb = kk + 1
            xb = width - 1

        else:
            kk = k - (n_left + n_bottom + n_right)
            v = crop[i, 0, kk + 1]
            yb = 0
            xb = kk + 1

        borders[i, k] = float("nan") if pad_mask[i, yb, xb] else v
