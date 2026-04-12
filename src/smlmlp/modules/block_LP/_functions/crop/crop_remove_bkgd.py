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

        # Store the crop border values without duplicating the four corners.
        borders = xp.empty_like(
            crop,
            shape=(n_crops, (height - 1 + width - 1) * 2),
        )

        if cuda:
            threads_per_block = 128
            blocks_per_grid = n_crops

            borders_gpu[blocks_per_grid, threads_per_block](crop, borders)
            med = xp.median(borders, axis=1)
            crop = crop - med[:, None, None]
        else:
            with nb_threads(parallel):
                borders_cpu(crop, borders)
            med = bn.median(borders, axis=1)
            crop = crop - med[:, None, None]

        new_crops.append(crop)
        border_medians.append(med)

    info = {
        "border_medians": border_medians,
    }

    return new_crops, info



@nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
def borders_cpu(crop, borders):
    """Fill the border buffer for each crop on CPU."""
    n_crops, height, width = crop.shape

    for i in nb.prange(n_crops):
        cr = crop[i]
        i1 = 0

        i0, i1 = i1, i1 + height - 1
        borders[i, i0:i1] = cr[:-1, 0]

        i0, i1 = i1, i1 + width - 1
        borders[i, i0:i1] = cr[-1, :-1]

        i0, i1 = i1, i1 + height - 1
        borders[i, i0:i1] = cr[1:, -1]

        i0, i1 = i1, i1 + width - 1
        borders[i, i0:i1] = cr[0, 1:]



@nb_cuda.jit(fastmath=True, cache=True)
def borders_gpu(crop, borders):
    """Fill the border buffer for each crop on GPU."""
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
            borders[i, k] = crop[i, k, 0]

        elif k < n_left + n_bottom:
            kk = k - n_left
            borders[i, k] = crop[i, height - 1, kk]

        elif k < n_left + n_bottom + n_right:
            kk = k - (n_left + n_bottom)
            borders[i, k] = crop[i, kk + 1, width - 1]

        else:
            kk = k - (n_left + n_bottom + n_right)
            borders[i, k] = crop[i, 0, kk + 1]