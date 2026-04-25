#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from smlmlp import analysis



@analysis(df_name="points")
def associate_consecutive_frames_radius(
    xx,
    yy,
    fr,
    *,
    bins=1000,
    r_min=1.0,
    r_max=None,
    smooth=3,
    cuda=False,
    parallel=False,
):
    """
    Estimate an xy association radius from consecutive frames.

    Parameters
    ----------
    xx, yy : array-like
        Localization coordinates.
    fr : array-like
        Frame index.
    bins : int, optional
        Number of logarithmic histogram bins.
    r_min : float, optional
        Minimum distance considered.
    r_max : float or None, optional
        Maximum distance considered. If None, uses a high percentile.
    smooth : int, optional
        Moving-average smoothing window on the histogram.
    cuda, parallel : bool, optional
        Execution options accepted by all analysis functions.

    Returns
    -------
    radius : float
        Estimated association radius.

    info : dict
        Diagnostic information.
    """

    x = np.asarray(xx, dtype=np.float32)
    y = np.asarray(yy, dtype=np.float32)
    fr = np.asarray(fr, dtype=np.uint32)

    order = np.argsort(fr, kind="stable")
    xs = x[order]
    ys = y[order]
    fs = fr[order]

    unique, start, counts = np.unique(
        fs,
        return_index=True,
        return_counts=True,
    )

    distances = []

    for k in range(len(unique) - 1):

        if unique[k + 1] != unique[k] + 1:
            continue

        a0 = start[k]
        a1 = a0 + counts[k]

        b0 = start[k + 1]
        b1 = b0 + counts[k + 1]

        pts_a = np.column_stack((xs[a0:a1], ys[a0:a1]))
        pts_b = np.column_stack((xs[b0:b1], ys[b0:b1]))

        if len(pts_a) == 0 or len(pts_b) == 0:
            continue

        tree_b = cKDTree(pts_b)
        d_ab, _ = tree_b.query(pts_a, k=1)
        distances.append(d_ab)

        tree_a = cKDTree(pts_a)
        d_ba, _ = tree_a.query(pts_b, k=1)
        distances.append(d_ba)

    if len(distances) == 0:
        raise ValueError("No consecutive frames with localizations found.")

    distances = np.concatenate(distances)
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    if len(distances) == 0:
        raise ValueError("No valid nearest-neighbor distances found.")

    if r_max is None:
        r_max = np.percentile(distances, 99.5)

    distances = distances[(distances >= r_min) & (distances <= r_max)]

    if len(distances) == 0:
        raise ValueError("No distances remain after r_min/r_max filtering.")

    edges = np.logspace(np.log10(r_min), np.log10(r_max), bins + 1)
    hist, edges = np.histogram(distances, bins=edges)

    centers = np.sqrt(edges[:-1] * edges[1:])

    hist_smooth = hist.astype(np.float32)

    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        hist_smooth = np.convolve(hist_smooth, kernel, mode="same")

    peaks, _ = find_peaks(hist_smooth)

    if len(peaks) >= 2:
        # Take the first two peaks in radius order
        p1 = peaks[0]
        p2 = peaks[1]

        # Valley between first and second peak
        valley_region = hist_smooth[p1:p2 + 1]
        valley_local = np.argmin(valley_region)
        valley = p1 + valley_local

        radius = centers[valley]

    elif len(peaks) == 1:
        # Fallback: find first minimum after the first peak
        p1 = peaks[0]

        if p1 < len(hist_smooth) - 2:
            after = hist_smooth[p1 + 1:]
            valley = p1 + 1 + np.argmin(after)
            radius = centers[valley]
        else:
            radius = centers[p1]

    else:
        # Last fallback: use a robust low percentile
        radius = np.percentile(distances, 25)

    info = {
        "distances": distances,
        "hist": hist,
        "hist_smooth": hist_smooth,
        "bin_edges": edges,
        "bin_centers": centers,
        "peaks": peaks,
        "n_distances": len(distances),
        "r_min": r_min,
        "r_max": r_max,
    }

    return float(radius), info
