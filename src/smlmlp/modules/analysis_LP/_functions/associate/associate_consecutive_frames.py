#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import analysis
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
import os



# %% Function
@analysis(df_name="points")
def associate_consecutive_frames(
    xx,
    yy,
    fr,
    zz=None,
    *,
    association_radius_nm=30.0,
    z_association_radius_nm=100.0,
    cuda=False,
    parallel=False,
):
    """
    Associate localizations in consecutive frames using:
    - 2D cKDTree candidate search on x/y
    - optional independent z validation
    - connected components
    - Hungarian assignment only when needed
    
    A link is valid if:
        xy_distance <= min(association_radius_nm_i, association_radius_nm_j)

    and, if z is provided:
        abs(z_i - z_j) <= min(z_association_radius_nm_i, z_association_radius_nm_j)
    """

    # --------------------------------------------------------------
    # Normalize parallel argument
    # --------------------------------------------------------------
    if parallel is False or parallel == 1:
        n_jobs = 1
    elif parallel is True or parallel == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    elif isinstance(parallel, int) and parallel > 1:
        n_jobs = parallel
    else:
        raise ValueError("Invalid value for 'parallel'")

    # --------------------------------------------------------------
    # Input conversion
    # --------------------------------------------------------------
    x = np.asarray(xx, dtype=np.float32)
    y = np.asarray(yy, dtype=np.float32)
    fr = np.asarray(fr, dtype=np.uint32)

    n = len(fr)

    if n == 0:
        return np.empty(0, dtype=np.uint32)

    if np.ndim(association_radius_nm) == 0:
        association_radius_nm = np.full(n, association_radius_nm, dtype=np.float32)
    else:
        association_radius_nm = np.asarray(association_radius_nm, dtype=np.float32)

    use_z = zz is not None

    if use_z:
        z = np.asarray(zz, dtype=np.float32)

        if np.ndim(z_association_radius_nm) == 0:
            z_association_radius_nm = np.full(n, z_association_radius_nm, dtype=np.float32)
        else:
            z_association_radius_nm = np.asarray(z_association_radius_nm, dtype=np.float32)
    else:
        z = None
        z_association_radius_nm = None

    # --------------------------------------------------------------
    # Sort by frame
    # --------------------------------------------------------------
    order = np.argsort(fr, kind="stable")

    xs = x[order]
    ys = y[order]
    fs = fr[order]
    rs = association_radius_nm[order]

    if use_z:
        zs = z[order]
        zrs = z_association_radius_nm[order]
    else:
        zs = None
        zrs = None

    unique, start, counts = np.unique(
        fs,
        return_index=True,
        return_counts=True,
    )

    # --------------------------------------------------------------
    # Worker function
    # --------------------------------------------------------------
    def _associate_one(k):

        if unique[k + 1] != unique[k] + 1:
            return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

        a0 = start[k]
        a1 = a0 + counts[k]

        b0 = start[k + 1]
        b1 = b0 + counts[k + 1]

        pts_a = np.column_stack((xs[a0:a1], ys[a0:a1]))
        pts_b = np.column_stack((xs[b0:b1], ys[b0:b1]))

        r_a = rs[a0:a1]
        r_b = rs[b0:b1]

        n_a = len(pts_a)
        n_b = len(pts_b)

        if n_a == 0 or n_b == 0:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)

        max_r = float(max(r_a.max(), r_b.max()))

        tree_a = cKDTree(pts_a)
        tree_b = cKDTree(pts_b)

        sparse = tree_a.sparse_distance_matrix(
            tree_b,
            max_distance=max_r,
            output_type="coo_matrix",
        )

        if sparse.nnz == 0:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)

        row = sparse.row.astype(np.uint64)
        col = sparse.col.astype(np.uint64)
        dist = sparse.data.astype(np.float64)

        keep = dist <= np.minimum(r_a[row], r_b[col])

        if use_z:
            z_a = zs[a0:a1]
            z_b = zs[b0:b1]
            zr_a = zrs[a0:a1]
            zr_b = zrs[b0:b1]

            dz = np.abs(z_a[row] - z_b[col])
            keep &= dz <= np.minimum(zr_a[row], zr_b[col])

        row = row[keep]
        col = col[keep]
        dist = dist[keep]

        if len(dist) == 0:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)

        graph_row = np.concatenate((row, n_a + col))
        graph_col = np.concatenate((n_a + col, row))

        graph = coo_matrix(
            (np.ones(len(graph_row)), (graph_row, graph_col)),
            shape=(n_a + n_b, n_a + n_b),
        )

        _, labels = connected_components(graph, directed=False)

        used_nodes = np.unique(graph_row)
        component_ids = np.unique(labels[used_nodes])

        accepted_a = []
        accepted_b = []

        for cid in component_ids:

            mask = labels[row] == cid

            comp_row = row[mask]
            comp_col = col[mask]
            comp_dist = dist[mask]

            if len(comp_dist) == 1:
                accepted_a.append(comp_row[0])
                accepted_b.append(comp_col[0])
                continue

            comp_a = np.unique(comp_row)
            comp_b = np.unique(comp_col)

            if len(comp_a) == 1 or len(comp_b) == 1:
                best = np.argmin(comp_dist)
                accepted_a.append(comp_row[best])
                accepted_b.append(comp_col[best])
                continue

            deg_a = np.bincount(
                np.searchsorted(comp_a, comp_row),
                minlength=len(comp_a),
            )

            deg_b = np.bincount(
                np.searchsorted(comp_b, comp_col),
                minlength=len(comp_b),
            )

            if deg_a.max() == 1 and deg_b.max() == 1:
                accepted_a.extend(comp_row.tolist())
                accepted_b.extend(comp_col.tolist())
                continue

            local_row = np.searchsorted(comp_a, comp_row)
            local_col = np.searchsorted(comp_b, comp_col)

            cost = np.full((len(comp_a), len(comp_b)), 1e20)
            cost[local_row, local_col] = comp_dist

            ia, ib = linear_sum_assignment(cost)

            valid = cost[ia, ib] < 1e20

            for i, j in zip(ia[valid], ib[valid]):
                accepted_a.append(comp_a[i])
                accepted_b.append(comp_b[j])

        return a0 + np.array(accepted_a), b0 + np.array(accepted_b)

    # --------------------------------------------------------------
    # Run (parallel or not)
    # --------------------------------------------------------------
    if n_jobs == 1:
        links = [_associate_one(k) for k in range(len(unique) - 1)]
    else:
        links = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_associate_one)(k) for k in range(len(unique) - 1)
        )

    # --------------------------------------------------------------
    # Merge links into graph
    # --------------------------------------------------------------
    src = []
    dst = []

    for s, d in links:
        if len(s) > 0:
            src.append(s)
            dst.append(d)

    if len(src) == 0:
        out = np.arange(n)
        out_original = np.empty_like(out)
        out_original[order] = out
        return out_original

    src = np.concatenate(src)
    dst = np.concatenate(dst)

    graph = coo_matrix(
        (np.ones(len(src) * 2),
         (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(n, n),
    )

    _, labels = connected_components(graph, directed=False)

    out = np.empty_like(labels)
    out[order] = labels

    _, out = np.unique(out, return_inverse=True)

    info = {
        "n_localizations": n,
        "n_links": len(src),
        "n_tracks": np.max(out) + 1,
        "mean_track_length": n / (np.max(out) + 1),
        "link_ratio": len(src) / n,
        "links_per_frame": np.array([len(s) for s, _ in links]),
        "locs_per_frame": counts,
    }
    return out.astype(np.uint64) + 1, info