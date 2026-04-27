#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import os

from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from smlmlp import analysis



@analysis(df_name="points")
def metric_nena(
    col,
    xx,
    yy,
    fr,
    zz=None,
    *,
    association_radius_nm=30.0,
    z_association_radius_nm=100.0,
    bins=100,
    cuda=False,
    parallel=False,
):
    """
    Estimate a one-dimensional NeNA precision from consecutive-frame pairs.

    Parameters
    ----------
    col : array-like
        Values used for the one-dimensional NeNA metric. For each accepted pair,
        the fitted sample is ``col_t - col_t_plus_1``. This can be ``xx``, ``yy``,
        or any other scalar localization column.
    xx, yy : array-like
        Localization coordinates used to find nearest neighbors in xy.
    fr : array-like
        Frame identifiers. Only pairs between frames ``t`` and ``t + 1`` are used.
    zz : array-like or None, optional
        Optional z coordinates used as an independent z-distance filter.
    association_radius_nm : float or array-like, optional
        Maximum xy nearest-neighbor radius. Array inputs are interpreted per
        localization, and a pair is accepted within the minimum of the two radii.
    z_association_radius_nm : float or array-like, optional
        Maximum z-distance when ``zz`` is provided. Array inputs are interpreted
        per localization, and a pair is accepted within the minimum of the two
        z radii.
    bins : int or array-like, optional
        Histogram bin specification passed to :func:`numpy.histogram`.
    cuda, parallel : bool or int, optional
        Execution options accepted by all analysis functions. ``cuda`` is accepted
        for API consistency; the current implementation runs on CPU.

    Returns
    -------
    precision : float
        One-dimensional NeNA precision, computed as ``sigma / sqrt(2)`` from the
        fitted Gaussian width.
    info : dict
        Diagnostic information containing the paired differences, histogram,
        Gaussian fit parameters, selected indices, and association counts.

    Raises
    ------
    ValueError
        If input lengths are inconsistent, no valid pairs are found, or the
        Gaussian fit cannot be initialized from a non-degenerate distribution.

    Notes
    -----
    For each consecutive frame pair, localizations in frame ``t`` are linked to
    their nearest localization in frame ``t + 1`` among candidates passing the xy
    radius and optional z-radius filters. The links are directional and are not
    forced to be one-to-one: several localizations in frame ``t`` may select the
    same neighbor in frame ``t + 1``.

    This differs from classical radial NeNA: the fitted distribution is the
    signed one-dimensional difference of an arbitrary column, not a radial
    distance fitted with a Rayleigh model.

    Examples
    --------
    >>> import numpy as np
    >>> from smlmlp import metric_nena
    >>> rng = np.random.default_rng(0)
    >>> n = 50
    >>> x0 = np.arange(n, dtype=np.float32) * 50.0
    >>> xx = np.concatenate((x0, x0))
    >>> yy = np.zeros(2 * n, dtype=np.float32)
    >>> fr = np.concatenate(
    ...     (np.ones(n, dtype=np.uint32), np.full(n, 2, dtype=np.uint32))
    ... )
    >>> delta = rng.normal(0.0, 2.0, size=n).astype(np.float32)
    >>> col = np.concatenate((np.zeros(n, dtype=np.float32), -delta))
    >>> precision, info = metric_nena(col, xx, yy, fr, bins=12)
    >>> info["n_pairs"]
    50
    """

    # Configure execution
    if parallel is False or (
        isinstance(parallel, int) and not isinstance(parallel, bool) and parallel == 1
    ):
        n_jobs = 1
    elif parallel is True or parallel == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    elif isinstance(parallel, int) and parallel > 1:
        n_jobs = parallel
    else:
        raise ValueError("Invalid value for 'parallel'")

    values = np.asarray(col, dtype=np.float32)
    x = np.asarray(xx, dtype=np.float32)
    y = np.asarray(yy, dtype=np.float32)
    frames = np.asarray(fr, dtype=np.int64)

    n_input = len(frames)
    if not (len(values) == len(x) == len(y) == n_input):
        raise ValueError("col, xx, yy, and fr must have the same length.")

    if np.ndim(association_radius_nm) == 0:
        radius = np.full(n_input, association_radius_nm, dtype=np.float32)
    else:
        radius = np.asarray(association_radius_nm, dtype=np.float32)
        if len(radius) != n_input:
            raise ValueError(
                "association_radius_nm must be scalar or match input length."
            )

    use_z = zz is not None

    if use_z:
        z = np.asarray(zz, dtype=np.float32)
        if len(z) != n_input:
            raise ValueError("zz must have the same length as fr.")

        if np.ndim(z_association_radius_nm) == 0:
            z_radius = np.full(n_input, z_association_radius_nm, dtype=np.float32)
        else:
            z_radius = np.asarray(z_association_radius_nm, dtype=np.float32)
            if len(z_radius) != n_input:
                raise ValueError(
                    "z_association_radius_nm must be scalar or match input length."
                )
    else:
        z = None
        z_radius = None

    # Remove values that cannot contribute to nearest-neighbor matching or fitting
    valid = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    valid &= np.isfinite(radius)
    valid &= radius > 0

    if use_z:
        valid &= np.isfinite(z) & np.isfinite(z_radius)
        valid &= z_radius > 0

        z = z[valid]
        z_radius = z_radius[valid]

    values = values[valid]
    x = x[valid]
    y = y[valid]
    frames = frames[valid]
    radius = radius[valid]
    original_indices = np.flatnonzero(valid).astype(np.uint64)

    n = len(frames)
    if n == 0:
        raise ValueError("No valid localizations found.")

    # Sort frames once and reuse slices for every consecutive-frame pair
    order = np.argsort(frames, kind="stable")
    values_s = values[order]
    xs = x[order]
    ys = y[order]
    fs = frames[order]
    rs = radius[order]
    original_s = original_indices[order]

    if use_z:
        zs = z[order]
        zrs = z_radius[order]
    else:
        zs = None
        zrs = None

    unique, start, counts = np.unique(
        fs,
        return_index=True,
        return_counts=True,
    )

    def _nena_one(k):
        """Compute one-directional NeNA differences for one frame pair."""

        if unique[k + 1] != unique[k] + 1:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.uint64)
            return empty_f, empty_f, empty_i, empty_i

        a0 = start[k]
        a1 = a0 + counts[k]
        b0 = start[k + 1]
        b1 = b0 + counts[k + 1]

        pts_a = np.column_stack((xs[a0:a1], ys[a0:a1]))
        pts_b = np.column_stack((xs[b0:b1], ys[b0:b1]))

        if len(pts_a) == 0 or len(pts_b) == 0:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.uint64)
            return empty_f, empty_f, empty_i, empty_i

        r_a = rs[a0:a1]
        r_b = rs[b0:b1]
        max_r = float(max(r_a.max(), r_b.max()))

        tree_a = cKDTree(pts_a)
        tree_b = cKDTree(pts_b)
        sparse = tree_a.sparse_distance_matrix(
            tree_b,
            max_distance=max_r,
            output_type="coo_matrix",
        )

        if sparse.nnz == 0:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.uint64)
            return empty_f, empty_f, empty_i, empty_i

        row = sparse.row.astype(np.int64)
        col_nn = sparse.col.astype(np.int64)
        dist = sparse.data.astype(np.float64)

        keep = dist <= np.minimum(r_a[row], r_b[col_nn])

        if use_z:
            z_a = zs[a0:a1]
            z_b = zs[b0:b1]
            zr_a = zrs[a0:a1]
            zr_b = zrs[b0:b1]

            dz = np.abs(z_a[row] - z_b[col_nn])
            keep &= dz <= np.minimum(zr_a[row], zr_b[col_nn])

        row = row[keep]
        col_nn = col_nn[keep]
        dist = dist[keep]

        if len(dist) == 0:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.uint64)
            return empty_f, empty_f, empty_i, empty_i

        # Keep the nearest accepted neighbor for each localization in frame t
        local_order = np.lexsort((col_nn, dist, row))
        row = row[local_order]
        col_nn = col_nn[local_order]
        dist = dist[local_order]

        first = np.concatenate(([True], row[1:] != row[:-1]))
        row = row[first]
        col_nn = col_nn[first]
        dist = dist[first]

        src = original_s[a0 + row]
        dst = original_s[b0 + col_nn]
        differences = values_s[a0 + row] - values_s[b0 + col_nn]

        return differences.astype(np.float64), dist, src, dst

    if len(unique) < 2:
        links = []
    elif n_jobs == 1:
        links = [_nena_one(k) for k in range(len(unique) - 1)]
    else:
        links = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_nena_one)(k) for k in range(len(unique) - 1)
        )

    differences = []
    distances = []
    src_indices = []
    dst_indices = []

    for diff, dist, src, dst in links:
        if len(diff) == 0: continue
        differences.append(diff)
        distances.append(dist)
        src_indices.append(src)
        dst_indices.append(dst)

    if len(differences) == 0:
        raise ValueError("No consecutive-frame nearest-neighbor pairs found.")

    differences = np.concatenate(differences)
    distances = np.concatenate(distances)
    src_indices = np.concatenate(src_indices)
    dst_indices = np.concatenate(dst_indices)

    if len(differences) < 3:
        raise ValueError("At least three non-degenerate differences are required.")

    sigma0 = float(np.std(differences, ddof=1))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("At least three non-degenerate differences are required.")

    # Fit a Gaussian to the one-dimensional difference histogram
    hist, edges = np.histogram(differences, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if len(centers) < 3:
        raise ValueError("At least three histogram bins are required.")

    amplitude0 = float(hist.max())
    mean0 = float(np.mean(differences))
    p0 = (amplitude0, mean0, sigma0)
    lower = (0.0, -np.inf, np.finfo(float).eps)
    upper = (np.inf, np.inf, np.inf)

    try:
        popt, pcov = curve_fit(
            _gaussian_1d,
            centers,
            hist.astype(np.float64),
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Gaussian fit failed.") from exc

    fit_amplitude = float(popt[0])
    fit_mean = float(popt[1])
    fit_sigma = float(popt[2])
    precision = fit_sigma / np.sqrt(2.0)

    info = {
        "differences": differences,
        "distances": distances,
        "hist": hist,
        "bin_edges": edges,
        "bin_centers": centers,
        "fit_params": popt,
        "fit_covariance": pcov,
        "fit_amplitude": fit_amplitude,
        "fit_mean": fit_mean,
        "fit_sigma": fit_sigma,
        "precision": float(precision),
        "n_input_localizations": n_input,
        "n_localizations": n,
        "n_pairs": len(differences),
        "n_frames": len(unique),
        "links_per_frame": np.array([len(diff) for diff, _, _, _ in links]),
        "locs_per_frame": counts,
        "source_indices": src_indices,
        "neighbor_indices": dst_indices,
        "association_radius_nm": association_radius_nm,
        "z_association_radius_nm": z_association_radius_nm if use_z else None,
    }

    return float(precision), info


def _gaussian_1d(x, amplitude, mean, sigma):
    """Evaluate a one-dimensional Gaussian profile."""

    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

