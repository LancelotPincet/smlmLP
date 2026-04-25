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
def aggregate_flux(
    intensity_eff,
    blk,
    fr,
    *,
    cuda=False,
    parallel=False,
):
    """
    Calculate blink flux from localization intensity_eff.

    Rules
    -----
    1. If blink has less than 3 frames/localizations -> NaN
    2. If one blink contains several localizations in same frame -> error
    3. Else flux = mean intensity_eff excluding first and last frame

    Also returns a boolean array "switching":
        True  -> localization is first or last in blink (excluded)
        False -> localization used in flux
    """

    blk = np.asarray(blk, dtype=np.uint64)
    fr = np.asarray(fr, dtype=np.uint32)
    intensity_eff = np.asarray(intensity_eff, dtype=np.float32)

    n = len(blk)

    # Sort by blink first, then frame
    order = np.lexsort((fr, blk))

    blink_s = blk[order]
    fr_s = fr[order]
    intensity_eff_s = intensity_eff[order]

    unique_blink, start, counts = np.unique(
        blink_s,
        return_index=True,
        return_counts=True,
    )

    with nb_threads(parallel) :
        flux, switching_s = _aggregate_flux(
            blink_s,
            fr_s,
            intensity_eff_s,
            start.astype(np.int64),
            counts.astype(np.int64),
            n,
        )

    # Reorder switching to original order
    switching = np.empty(n, dtype=np.bool_)
    switching[order] = switching_s

    return flux, switching, {}



@nb.njit(cache=True, parallel=True)
def _aggregate_flux(blink, fr, intensity_eff, start, counts, n):
    n_blinks = len(start)

    flux = np.empty(n_blinks, dtype=np.float32)
    flux[:] = np.nan

    switching = np.zeros(n, dtype=np.bool_)

    for i in nb.prange(n_blinks):
        s = start[i]
        c = counts[i]
        e = s + c

        # Mark switching frames (first and last)
        if c >= 1:
            switching[s] = True
        if c >= 2:
            switching[e - 1] = True

        # Less than 3 → no middle points
        if c < 3:
            continue

        # Check duplicated frames inside blink
        for j in range(s + 1, e):
            if fr[j] == fr[j - 1]:
                raise ValueError(
                    "A blink contains several localizations in the same frame."
                )

        # Compute mean excluding first and last
        total = 0.0
        n_mid = 0

        for j in range(s + 1, e - 1):
            total += intensity_eff[j]
            n_mid += 1

        flux[i] = total / n_mid

    return flux, switching