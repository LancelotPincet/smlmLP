#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


# %% Libraries
from smlmlp import analysis
import numpy as np
import numba as nb
from arrlp import nb_threads


# %% Function
@analysis(df_name="points")
def aggregate_ratio(
    col,
    /,
    pnt,
    ch,
    *,
    x_channels=None,
    y_channels=None,
    cuda=False,
    parallel=False,
):
    """
    Calculate ratio_x and ratio_y for each point.

    Rules
    -----
    1. For each point, ratio_x = sum col over x_channels
    2. For each point, ratio_y = sum col over y_channels
    3. If one requested channel is missing for a point -> NaN
    4. If one requested channel appears several times for a point -> error
    """

    col = np.asarray(col, dtype=np.float32)
    pnt = np.asarray(pnt, dtype=np.uint64)
    ch = np.asarray(ch, dtype=np.uint32)

    if x_channels is None or y_channels is None :
        x_channels = np.arange(1, ch.max()-1, dtype=np.uint32)
        y_channels = [ch.max()]
    x_channels = np.asarray(x_channels, dtype=np.uint32)
    y_channels = np.asarray(y_channels, dtype=np.uint32)

    n = len(pnt)

    # Sort by point first, then channel
    order = np.lexsort((ch, pnt))

    pnt_s = pnt[order]
    ch_s = ch[order]
    col_s = col[order]

    unique_point, start, counts = np.unique(
        pnt_s,
        return_index=True,
        return_counts=True,
    )

    with nb_threads(parallel):
        ratio_x, ratio_y = _aggregate_ratio(
            ch_s,
            col_s,
            start.astype(np.int64),
            counts.astype(np.int64),
            x_channels,
            y_channels,
        )

    return ratio_x, ratio_y, {}


@nb.njit(cache=True, parallel=True)
def _aggregate_ratio(
    ch,
    col,
    start,
    counts,
    x_channels,
    y_channels,
):
    n_points = len(start)

    ratio_x = np.empty(n_points, dtype=np.float32)
    ratio_y = np.empty(n_points, dtype=np.float32)

    ratio_x[:] = np.nan
    ratio_y[:] = np.nan

    for i in nb.prange(n_points):
        s = start[i]
        c = counts[i]
        e = s + c

        x_sum = 0.0
        y_sum = 0.0

        x_found = np.zeros(len(x_channels), dtype=np.uint8)
        y_found = np.zeros(len(y_channels), dtype=np.uint8)

        for j in range(s, e):
            channel = ch[j]
            value = col[j]

            for k in range(len(x_channels)):
                if channel == x_channels[k]:
                    x_found[k] += 1
                    x_sum += value

                    if x_found[k] > 1:
                        raise ValueError(
                            "A requested ratio_x channel appears several times for one point."
                        )

            for k in range(len(y_channels)):
                if channel == y_channels[k]:
                    y_found[k] += 1
                    y_sum += value

                    if y_found[k] > 1:
                        raise ValueError(
                            "A requested ratio_y channel appears several times for one point."
                        )

        x_ok = True
        for k in range(len(x_channels)):
            if x_found[k] != 1:
                x_ok = False

        y_ok = True
        for k in range(len(y_channels)):
            if y_found[k] != 1:
                y_ok = False

        if x_ok:
            ratio_x[i] = x_sum

        if y_ok:
            ratio_y[i] = y_sum

    return ratio_x, ratio_y