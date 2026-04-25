#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


# %% Libraries
from smlmlp import analysis
import numpy as np
import numba as nb
from arrlp import nb_threads


# %% Functions
@analysis(df_name="detections")
def inv_transform_locs(
    x,
    y,
    ch,
    channels_locs_transform_matrices,
    *,
    cuda=False,
    parallel=False,
):
    """
    Transform localizations from aligned coordinates back to non-aligned coordinates.

    matrices are still defined in the direction:
        non-aligned -> aligned

    This function applies the inverse matrix.

    Matrix convention:
        [y_old, x_old, 1] = inv(matrix) @ [y_aligned, x_aligned, 1]

    Coordinates and matrix shifts are both in nm.
    """

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ch = np.asarray(ch, dtype=np.uint8)
    channels_locs_transform_matrices = np.asarray(channels_locs_transform_matrices, dtype=np.float32)

    if channels_locs_transform_matrices.ndim == 2:
        channels_locs_transform_matrices = channels_locs_transform_matrices[None, :, :]

    inv_matrices = np.empty_like(channels_locs_transform_matrices)

    for i in range(len(matrices)):
        inv_matrices[i] = np.linalg.inv(channels_locs_transform_matrices[i]).astype(np.float32)

    with nb_threads(parallel):
        x_t, y_t = _transform_locs(
            x,
            y,
            ch,
            inv_matrices,
        )

    return x_t, y_t, {}

@nb.njit(cache=True, parallel=True)
def _transform_locs(
    x,
    y,
    ch,
    matrices,
):
    n = len(x)

    x_t = np.empty(n, dtype=np.float32)
    y_t = np.empty(n, dtype=np.float32)

    n_matrices = len(matrices)

    for i in nb.prange(n):

        matrix_index = ch[i] - 1

        if matrix_index < 0 or matrix_index >= n_matrices:
            x_t[i] = np.nan
            y_t[i] = np.nan
            continue

        m = matrices[matrix_index]

        # Matrix convention is image-like:
        # input vector = [y, x, 1]
        y_new = m[0, 0] * y[i] + m[0, 1] * x[i] + m[0, 2]
        x_new = m[1, 0] * y[i] + m[1, 1] * x[i] + m[1, 2]

        x_t[i] = x_new
        y_t[i] = y_new

    return x_t, y_t