#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


import os

from joblib import Parallel, delayed
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from smlmlp import analysis


@analysis(df_name="detections")
def associate_different_channels(
    x_globfit,
    y_globfit,
    fr,
    ch,
    association_radius_nm=30.0,
    *,
    cuda=False,
    parallel=False,
):
    """
    Associate same-frame localizations from different channels.

    Parameters
    ----------
    x_globfit, y_globfit : array-like
        Localization coordinates.
    fr : array-like
        Frame identifiers.
    ch : array-like
        Channel identifiers.
    association_radius_nm : float, optional
        Maximum spatial association radius.
    cuda, parallel : bool or int, optional
        Execution options accepted by all analysis functions.

    Returns
    -------
    point : ndarray uint64
        Point index per localization.
        0 means unassigned/problematic.

    info : dict
        Diagnostics.
    """

    if parallel is False or parallel == 0 or parallel == 1:
        n_jobs = 1
    elif parallel is True or parallel == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    elif isinstance(parallel, int) and parallel > 1:
        n_jobs = parallel
    else:
        raise ValueError("Invalid value for 'parallel'")

    x = np.asarray(x_globfit, dtype=np.float32)
    y = np.asarray(y_globfit, dtype=np.float32)
    fr = np.asarray(fr, dtype=np.uint32)
    ch = np.asarray(ch, dtype=np.uint8)

    required_channels = np.unique(ch)
    if required_channels[0] == 0 : required_channels = required_channels[1:]

    n = len(fr)

    if n == 0:
        info = {
            "n_ambiguous_components": 0,
            "n_groups": 0,
            "max_component_size": 0,
        }
        return np.empty(0, dtype=np.uint64), info

    radius = float(association_radius_nm)
    radius2 = radius * radius

    order = np.argsort(fr, kind="stable")

    xs = x[order]
    ys = y[order]
    fs = fr[order]
    cs = ch[order]

    unique, start, counts = np.unique(
        fs,
        return_index=True,
        return_counts=True,
    )

    def _associate_one_frame(k):
        """Associate localizations within one frame."""

        a0 = start[k]
        a1 = a0 + counts[k]

        xf = xs[a0:a1]
        yf = ys[a0:a1]
        cf = cs[a0:a1]

        n_frame = len(xf)

        point_local = np.zeros(n_frame, dtype=np.int64)

        stats = {
            "n_candidate_edges": 0,
            "n_components": 0,
            "n_missing_components": 0,
            "n_ambiguous_components": 0,
            "n_valid_components": 0,
            "n_groups": 0,
            "max_component_size": 0,
        }

        if n_frame == 0:
            return point_local, stats

        pts = np.column_stack((xf, yf))

        tree = cKDTree(pts)
        pairs = tree.query_pairs(radius, output_type="ndarray")

        if len(pairs) == 0:
            return point_local, stats

        keep = cf[pairs[:, 0]] != cf[pairs[:, 1]]
        pairs = pairs[keep]

        if len(pairs) == 0:
            return point_local, stats

        stats["n_candidate_edges"] = len(pairs)

        graph = coo_matrix(
            (
                np.ones(len(pairs) * 2, dtype=np.uint8),
                (
                    np.concatenate((pairs[:, 0], pairs[:, 1])),
                    np.concatenate((pairs[:, 1], pairs[:, 0])),
                ),
            ),
            shape=(n_frame, n_frame),
        )

        _, labels = connected_components(graph, directed=False)

        used_nodes = np.unique(pairs.ravel())
        component_ids = np.unique(labels[used_nodes])

        stats["n_components"] = len(component_ids)

        current_point = 1

        for cid in component_ids:

            idx = np.where(labels == cid)[0]

            stats["max_component_size"] = max(
                stats["max_component_size"],
                len(idx),
            )

            if len(idx) < len(required_channels):
                stats["n_missing_components"] += 1
                continue

            comp_x = xf[idx]
            comp_y = yf[idx]
            comp_ch = cf[idx]

            groups, component_status = _solve_multichannel_component(
                comp_x,
                comp_y,
                comp_ch,
                required_channels,
                radius2,
            )

            if component_status == 0:
                stats["n_missing_components"] += 1
                continue

            if component_status == 2:
                stats["n_ambiguous_components"] += 1

            if len(groups) == 0:
                continue

            stats["n_valid_components"] += 1
            stats["n_groups"] += len(groups)

            for g in range(len(groups)):
                for j in range(groups.shape[1]):
                    point_local[idx[groups[g, j]]] = current_point
                current_point += 1

        return point_local, stats

    if n_jobs == 1:
        per_frame = [_associate_one_frame(k) for k in range(len(unique))]
    else:
        per_frame = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_associate_one_frame)(k)
            for k in range(len(unique))
        )

    point_sorted = np.zeros(n, dtype=np.int64)

    offset = 0
    stats_all = []

    for k, (point_local, stats) in enumerate(per_frame):

        stats_all.append(stats)

        s = start[k]
        e = s + counts[k]

        mask = point_local > 0

        point_local_global = point_local.copy()
        point_local_global[mask] += offset

        point_sorted[s:e] = point_local_global

        if point_local.max() > 0:
            offset += point_local.max()

    point = np.zeros_like(point_sorted)
    point[order] = point_sorted

    info = {
        "n_ambiguous_components": int(sum(s["n_ambiguous_components"] for s in stats_all)),
        "n_groups": int(sum(s["n_groups"] for s in stats_all)),
        "max_component_size": int(max(s["max_component_size"] for s in stats_all)),
    }

    return point.astype(np.uint64), info


def _solve_multichannel_component(
    x,
    y,
    ch,
    required_channels,
    radius2,
):
    """Solve one multichannel connected component."""

    n_required = len(required_channels)

    channel_indices = []

    ambiguous = False

    for channel in required_channels:
        idx = np.where(ch == channel)[0]

        if len(idx) == 0:
            return np.zeros((0, n_required), dtype=np.int64), 0

        if len(idx) > 1:
            ambiguous = True

        channel_indices.append(idx)

    groups = []
    costs = []

    current = np.empty(n_required, dtype=np.int64)

    def build(level):
        """Build valid groups recursively."""

        if level == n_required:

            cx = 0.0
            cy = 0.0

            for i in range(n_required):
                cx += x[current[i]]
                cy += y[current[i]]

            cx /= n_required
            cy /= n_required

            cost = 0.0

            for i in range(n_required):
                dx = x[current[i]] - cx
                dy = y[current[i]] - cy
                cost += dx * dx + dy * dy

            groups.append(current.copy())
            costs.append(cost)
            return

        for candidate in channel_indices[level]:

            valid = True

            for prev_level in range(level):
                previous = current[prev_level]

                dx = x[candidate] - x[previous]
                dy = y[candidate] - y[previous]

                if dx * dx + dy * dy > radius2:
                    valid = False
                    break

            if not valid:
                continue

            current[level] = candidate
            build(level + 1)

    build(0)

    if len(groups) == 0:
        return np.zeros((0, n_required), dtype=np.int64), 0

    groups = np.asarray(groups, dtype=np.int64)
    costs = np.asarray(costs, dtype=np.float64)

    if not ambiguous:
        return groups[:1], 1

    order = np.argsort(costs)

    used = np.zeros(len(x), dtype=bool)
    selected = []

    for gidx in order:

        group = groups[gidx]

        if np.any(used[group]):
            continue

        selected.append(group.copy())
        used[group] = True

    if len(selected) == 0:
        return np.zeros((0, n_required), dtype=np.int64), 2

    return np.asarray(selected, dtype=np.int64), 2
