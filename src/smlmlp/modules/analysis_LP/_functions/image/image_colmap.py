#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


import numba as nb
import numpy as np

from smlmlp import analysis
from smlmlp.modules.analysis_LP._functions.image.image_smlm import _resolve_shape


@analysis(df_name="pixels")
def image_colmap(
    col,
    /,
    pix,
    x_shape=None,
    y_shape=None,
    shape=None,
    *,
    reducer="mean",
    fill=np.nan,
    cuda=False,
    parallel=False,
):
    """
    Create a 2D image map by aggregating a column over flattened pixel ids.

    Parameters
    ----------
    col : array-like
        Values to map.
    pix : array-like
        Flattened pixel identifiers, encoded as ``y * x_shape + x``.
    x_shape, y_shape, shape : int or tuple, optional
        Output dimensions. If omitted, a one-row image large enough to contain
        the maximum pixel id is returned because a flat id alone cannot recover
        the original width.
    reducer : {'mean', 'sum'}, optional
        Aggregation method for duplicate pixels.
    fill : float, optional
        Value assigned to pixels without data. Default is NaN.
    cuda, parallel : bool, optional
        Accepted for analysis API consistency.

    Returns
    -------
    image : ndarray
        Aggregated ``float32`` map.
    info : dict
        Aggregation metadata.
    """

    del cuda, parallel

    values = np.ascontiguousarray(np.asarray(col, dtype=np.float32).ravel())
    pixels = _prepare_pix(pix, len(values))
    y_size, x_size = _resolve_colmap_shape(pixels, shape, x_shape, y_shape)

    reducer = reducer.lower()
    if reducer not in {"mean", "sum"}:
        raise ValueError("reducer must be 'mean' or 'sum'.")

    sums, counts = _colmap_sum_count(values, pixels, y_size * x_size)
    image = sums.reshape((y_size, x_size)).astype(np.float32)
    count_image = counts.reshape((y_size, x_size))

    mask = count_image > 0
    if reducer == "mean":
        image[mask] = image[mask] / count_image[mask]
    image[~mask] = fill

    info = {
        "reducer": reducer,
        "n_used": int(np.sum(counts)),
        "x_shape": int(x_size),
        "y_shape": int(y_size),
    }
    return image, info


def _prepare_pix(pix, n):
    """Return integer pixel ids, using -1 for invalid entries."""
    raw = np.asarray(pix).ravel()
    if len(raw) != n:
        raise ValueError(f"pix must have length {n}.")

    raw_float = raw.astype(np.float64, copy=False)
    pixels = np.full(n, -1, dtype=np.int64)
    finite = np.isfinite(raw_float)
    pixels[finite] = np.rint(raw_float[finite]).astype(np.int64)
    return np.ascontiguousarray(pixels)


def _resolve_colmap_shape(pixels, shape, x_shape, y_shape):
    """Resolve output shape for flat pixel ids."""
    y_size, x_size = _resolve_shape(shape, x_shape, y_shape)
    if y_size is not None:
        return y_size, x_size

    valid = pixels[pixels >= 0]
    x_size = int(np.max(valid)) + 1 if len(valid) else 1
    return 1, max(x_size, 1)


@nb.njit(cache=True)
def _colmap_sum_count(values, pixels, size):
    """Accumulate sums and counts for flattened pixel ids."""
    sums = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.int64)

    for i in range(len(values)):
        value = values[i]
        pix = pixels[i]
        if pix < 0 or pix >= size:
            continue
        if not np.isfinite(value):
            continue
        sums[pix] += value
        counts[pix] += 1

    return sums, counts
