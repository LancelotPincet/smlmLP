#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import transform_matrix, transform_parameters
import numpy as np
import bottleneck as bn



@block()
def registrate_solve_redundant_affine(
    shiftx,
    shifty,
    angle,
    shearx,
    sheary,
    scalex,
    scaley,
    pair_info=None,
    /,
    sigma_thresh=3.0,
    max_outliers=None,
    shape=None,
    ref_pix=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Solve absolute channel affine transforms from redundant pairwise transforms.

    This function reconstructs one affine transform per channel from redundant
    pairwise affine measurements. It follows the same two-pass residual-based
    outlier rejection used for redundant shifts, but solves the affine matrices
    directly. The solved matrices are then recentered by the inverse of their
    mean matrix so the global movement is distributed across all channels.

    Parameters
    ----------
    shiftx, shifty : array-like
        Pairwise shifts. Values are interpreted in the units defined by
        ``ref_pix``.
    angle : array-like
        Pairwise rotations in degrees.
    shearx, sheary : array-like
        Pairwise shear parameters.
    scalex, scaley : array-like
        Pairwise scale parameters.
    pair_info : dict or None, optional
        Optional info dictionary returned by ``registrate_ecc_affine``. When
        provided, ``shape`` and ``ref_pix`` are read from it unless explicitly
        supplied.
    sigma_thresh : float, optional
        Threshold multiplier applied to the robust residual dispersion for
        outlier rejection in the second pass.
    max_outliers : int or None, optional
        Maximum number of outlier pairs to reject. If ``None``, all detected
        outliers are removed.
    shape : tuple or None, optional
        Image shape used to rebuild pairwise affine matrices.
    ref_pix : float or tuple of float or None, optional
        Reference pixel size used to convert shifts back to pixels. If ``None``,
        it is read from ``pair_info`` or defaults to ``1.0``.
    cuda : bool, optional
        Unused in this function. It is kept for API consistency.
    parallel : bool, optional
        Unused in this function. It is kept for API consistency.

    Returns
    -------
    tuple
        A tuple ``(abs_shiftx, abs_shifty, abs_angle, abs_shearx, abs_sheary,
        abs_scalex, abs_scaley, info)`` where the first seven arrays contain one
        solved transform parameter per channel.
    """
    # Convert the input pairwise parameters to NumPy arrays.
    shiftx = np.asarray(shiftx, dtype=float)
    shifty = np.asarray(shifty, dtype=float)
    angle = np.asarray(angle, dtype=float)
    shearx = np.asarray(shearx, dtype=float)
    sheary = np.asarray(sheary, dtype=float)
    scalex = np.asarray(scalex, dtype=float)
    scaley = np.asarray(scaley, dtype=float)

    assert len(shiftx) == len(shifty)
    assert len(shiftx) == len(angle)
    assert len(shiftx) == len(shearx)
    assert len(shiftx) == len(sheary)
    assert len(shiftx) == len(scalex)
    assert len(shiftx) == len(scaley)

    if pair_info is not None:
        shape = pair_info.get("shape", shape) if shape is None else shape
        ref_pix = pair_info.get("ref_pix", ref_pix) if ref_pix is None else ref_pix

    ref_pix = 1.0 if ref_pix is None else ref_pix
    ref_pix = _normalize_ref_pix(ref_pix)

    npairs = len(shiftx)
    nchannels = int((1 + np.sqrt(1 + 8 * npairs)) / 2)

    # Build the canonical list of channel pairs corresponding to a fully
    # redundant upper-triangular pair ordering.
    pairs = [(i, j) for i in range(nchannels) for j in range(i + 1, nchannels)]

    # Rebuild pairwise matrices from the pairwise parameters.
    matrices = np.asarray(
        [
            transform_matrix(
                shape,
                shiftx=shiftx[k] / ref_pix[1],
                shifty=shifty[k] / ref_pix[0],
                angle=angle[k],
                shearx=shearx[k],
                sheary=sheary[k],
                scalex=scalex[k],
                scaley=scaley[k],
            )
            for k in range(npairs)
        ],
        dtype=float,
    )

    def _unknown_index(channel, row, col):
        """Return the least-squares index for an affine matrix entry."""
        return (channel - 1) * 6 + row * 3 + col

    def _build_system(pairs_kept, matrices_kept):
        """Build the linear system M_j = W_ij @ M_i for kept pairs."""
        A = np.zeros((len(pairs_kept) * 6, (nchannels - 1) * 6), dtype=float)
        b = np.zeros(len(pairs_kept) * 6, dtype=float)
        identity = np.eye(3, dtype=float)
        eq = 0

        for (i, j), matrix in zip(pairs_kept, matrices_kept):
            for row in range(2):
                for col in range(3):
                    known = 0.0

                    if j == 0:
                        known += identity[row, col]
                    else:
                        A[eq, _unknown_index(j, row, col)] += 1.0

                    for inner in range(2):
                        coeff = -matrix[row, inner]
                        if i == 0:
                            known += coeff * identity[inner, col]
                        else:
                            A[eq, _unknown_index(i, inner, col)] += coeff

                    known += -matrix[row, 2] * (1.0 if col == 2 else 0.0)
                    b[eq] = -known
                    eq += 1

        return A, b

    def _solve(mask_keep):
        """Solve absolute affine matrices for kept pairwise measurements."""
        if nchannels == 1:
            return np.eye(3, dtype=float)[None, :, :]

        pairs_kept = [p for p, keep in zip(pairs, mask_keep) if keep]
        matrices_kept = matrices[mask_keep]

        A, b = _build_system(pairs_kept, matrices_kept)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)

        abs_matrices = np.tile(np.eye(3, dtype=float), (nchannels, 1, 1))

        for channel in range(1, nchannels):
            start = (channel - 1) * 6
            abs_matrices[channel, :2, :] = sol[start:start + 6].reshape(2, 3)

        return abs_matrices

    def _predict_pairwise(abs_matrices):
        """Predict pairwise matrices from solved absolute matrices."""
        return np.asarray(
            [
                abs_matrices[j] @ np.linalg.inv(abs_matrices[i])
                for i, j in pairs
            ],
            dtype=float,
        )

    def _residuals(abs_matrices):
        """Compute residual matrices and one scalar residual per pair."""
        pred = _predict_pairwise(abs_matrices)
        resid_matrices = matrices - pred
        resid = np.linalg.norm(resid_matrices[:, :2, :], axis=(1, 2))
        return pred, resid_matrices, resid

    def _robust_sigma(x):
        """Estimate a robust dispersion from the median absolute deviation."""
        med = bn.median(x)
        mad = bn.median(np.abs(x - med))
        return 1.4826 * mad if mad > 0 else 0.0

    # First pass: solve using all pairwise measurements.
    mask_keep_1 = np.ones(npairs, dtype=bool)
    abs_matrices_1 = _solve(mask_keep_1)

    pred_1, resid_matrices_1, resid_1 = _residuals(abs_matrices_1)

    # Estimate a robust outlier threshold from the first-pass residuals.
    sigma_r = _robust_sigma(resid_1)
    med_r = bn.median(resid_1)

    if sigma_r == 0:
        mask_keep_2 = mask_keep_1.copy()
        outlier_idx = np.array([], dtype=int)
    else:
        threshold = med_r + sigma_thresh * sigma_r
        outlier_idx = np.where(resid_1 > threshold)[0]

        if max_outliers is not None and len(outlier_idx) > max_outliers:
            worst = np.argsort(resid_1[outlier_idx])[::-1][:max_outliers]
            outlier_idx = outlier_idx[worst]

        mask_keep_2 = np.ones(npairs, dtype=bool)
        mask_keep_2[outlier_idx] = False

        # Keep all pairs if too many were rejected and the system would become
        # underdetermined in principle.
        if mask_keep_2.sum() < (nchannels - 1):
            mask_keep_2[:] = True
            outlier_idx = np.array([], dtype=int)

    # Second pass: solve again after excluding the detected outliers.
    abs_matrices_2 = _solve(mask_keep_2)

    pred_2, resid_matrices_2, resid_2 = _residuals(abs_matrices_2)

    # Recenter origin by right-composing every absolute transform with the
    # inverse average transform. This preserves all pairwise transforms.
    origin_matrix = np.mean(abs_matrices_2, axis=0)
    center_matrix = np.linalg.inv(origin_matrix)
    abs_matrices_2 = np.asarray(
        [matrix @ center_matrix for matrix in abs_matrices_2],
        dtype=float,
    )

    (
        abs_shiftx,
        abs_shifty,
        abs_angle,
        abs_shearx,
        abs_sheary,
        abs_scalex,
        abs_scaley,
    ) = _parameters_from_matrices(abs_matrices_2, shape, ref_pix)

    info = {
        "ref_pix": ref_pix,
        "shape": shape,
        "pairs": pairs,
        "pair_matrices": matrices,
        "mask_keep_first_pass": mask_keep_1,
        "mask_keep_second_pass": mask_keep_2,
        "outlier_idx": outlier_idx,
        "outlier_pairs": [pairs[k] for k in outlier_idx],
        "pred_matrices_first_pass": pred_1,
        "resid_matrices_first_pass": resid_matrices_1,
        "resid_first_pass": resid_1,
        "pred_matrices_second_pass": pred_2,
        "resid_matrices_second_pass": resid_matrices_2,
        "resid_second_pass": resid_2,
        "median_resid_first_pass": med_r,
        "robust_sigma_first_pass": sigma_r,
        "abs_matrices_first_pass": abs_matrices_1,
        "abs_matrices_second_pass": abs_matrices_2,
        "origin_matrix": origin_matrix,
        "center_matrix": center_matrix,
    }

    return (
        abs_shiftx,
        abs_shifty,
        abs_angle,
        abs_shearx,
        abs_sheary,
        abs_scalex,
        abs_scaley,
        info,
    )



def _normalize_ref_pix(ref_pix):
    """Normalize a scalar or pair reference pixel size to ``(y, x)``."""
    try:
        if len(ref_pix) != 2:
            raise ValueError("ref_pix does not have 2 values (y, x)")
    except TypeError:
        ref_pix = (ref_pix, ref_pix)

    return ref_pix



def _parameters_from_matrices(matrices, shape, ref_pix):
    """Recover parameter arrays from solved affine matrices."""
    shiftx = []
    shifty = []
    angle = []
    shearx = []
    sheary = []
    scalex = []
    scaley = []

    for matrix in matrices:
        dx, dy, da, dsx, dsy, scx, scy = _matrix_to_parameters(
            matrix,
            shape,
            ref_pix,
        )
        shiftx.append(dx)
        shifty.append(dy)
        angle.append(da)
        shearx.append(dsx)
        sheary.append(dsy)
        scalex.append(scx)
        scaley.append(scy)

    return (
        np.asarray(shiftx, dtype=float),
        np.asarray(shifty, dtype=float),
        np.asarray(angle, dtype=float),
        np.asarray(shearx, dtype=float),
        np.asarray(sheary, dtype=float),
        np.asarray(scalex, dtype=float),
        np.asarray(scaley, dtype=float),
    )



def _matrix_to_parameters(matrix, shape, ref_pix):
    """Recover transform parameters from an affine matrix."""
    rotation = _polar_rotation_angle(matrix[:2, :2])
    dx, dy, dsx, dsy, da, scx, scy = transform_parameters(
        matrix,
        shape,
        angle=rotation,
    )

    return dx * ref_pix[1], dy * ref_pix[0], da, dsx, dsy, scx, scy



def _polar_rotation_angle(linear):
    """Return the closest proper-rotation angle for a 2x2 affine matrix."""
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    theta = np.arctan2(rotation[1, 0], rotation[0, 0])
    return -np.degrees(theta)
