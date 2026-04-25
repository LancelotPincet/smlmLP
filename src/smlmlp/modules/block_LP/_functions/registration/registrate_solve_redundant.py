#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
import numpy as np
import bottleneck as bn



@block()
def registrate_solve_redundant(
    shiftx,
    shifty,
    /,
    sigma_thresh=3.0,
    max_outliers=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Solve absolute channel shifts from redundant pairwise measurements.

    This function reconstructs absolute x and y channel shifts from redundant
    pairwise shift measurements using a two-pass least-squares procedure. A
    first fit is used to estimate residuals and detect outliers, and a second
    fit is then computed after removing the detected outlier pairs.

    Parameters
    ----------
    shiftx : array-like
        Pairwise shifts along x.
    shifty : array-like
        Pairwise shifts along y. Must have the same length as ``shiftx``.
    sigma_thresh : float, optional
        Threshold multiplier applied to the robust residual dispersion for
        outlier rejection in the second pass.
    max_outliers : int or None, optional
        Maximum number of outlier pairs to reject. If ``None``, all detected
        outliers are removed.
    cuda : bool, optional
        Unused in this function. It is kept for API consistency.
    parallel : bool, optional
        Unused in this function. It is kept for API consistency.

    Returns
    -------
    tuple
        A tuple ``(abs_shiftx, abs_shifty, info)`` where:

        - ``abs_shiftx`` is the solved absolute shift along x for each channel,
        - ``abs_shifty`` is the solved absolute shift along y for each channel,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'pairs'``
            List of channel pairs corresponding to the input pairwise shifts.
        ``'mask_keep_first_pass'``
            Boolean mask of pairs kept in the first pass.
        ``'mask_keep_second_pass'``
            Boolean mask of pairs kept in the second pass.
        ``'outlier_idx'``
            Indices of rejected outlier pairs.
        ``'outlier_pairs'``
            Channel pairs corresponding to the rejected outliers.
        ``'residx_first_pass'``
            Residuals along x after the first pass.
        ``'residy_first_pass'``
            Residuals along y after the first pass.
        ``'resid_first_pass'``
            Euclidean residual norm after the first pass.
        ``'residx_second_pass'``
            Residuals along x after the second pass.
        ``'residy_second_pass'``
            Residuals along y after the second pass.
        ``'resid_second_pass'``
            Euclidean residual norm after the second pass.
        ``'median_resid_first_pass'``
            Median residual norm after the first pass.
        ``'robust_sigma_first_pass'``
            Robust residual dispersion estimate after the first pass.

    Notes
    -----
    Channel 0 is used as the reference channel, so its absolute shift is fixed
    to zero. The remaining channel shifts are solved by least squares.

    The robust residual dispersion is estimated from the median absolute
    deviation:

    .. math::

        \\sigma_\\mathrm{robust} = 1.4826 \\times \\mathrm{MAD}

    Examples
    --------
    Solve a simple 3-channel system:

    >>> shiftx = [1.0, 2.0, 1.0]
    >>> shifty = [0.0, 1.0, 1.0]
    >>> abs_shiftx, abs_shifty, info = registrate_solve_redundant(shiftx, shifty)
    >>> abs_shiftx.shape[0]
    3
    >>> abs_shifty.shape[0]
    3
    >>> len(info["pairs"])
    3

    Limit the number of rejected outliers:

    >>> shiftx = [1.0, 2.0, 10.0]
    >>> shifty = [0.0, 1.0, 8.0]
    >>> abs_shiftx, abs_shifty, info = registrate_solve_redundant(
    ...     shiftx,
    ...     shifty,
    ...     sigma_thresh=2.0,
    ...     max_outliers=1,
    ... )
    >>> info["outlier_idx"].ndim
    1
    """
    # Convert the input pairwise shifts to NumPy arrays.
    shiftx = np.asarray(shiftx, dtype=float)
    shifty = np.asarray(shifty, dtype=float)

    assert len(shiftx) == len(shifty)

    npairs = len(shiftx)
    nchannels = int((1 + np.sqrt(1 + 8 * npairs)) / 2)

    # Build the canonical list of channel pairs corresponding to a fully
    # redundant upper-triangular pair ordering.
    pairs = [(i, j) for i in range(nchannels) for j in range(i + 1, nchannels)]

    def _build_A(pairs_kept):
        """Build the least-squares design matrix for kept channel pairs."""
        A = np.zeros((len(pairs_kept), nchannels - 1), dtype=float)

        for k, (i, j) in enumerate(pairs_kept):
            if i != 0:
                A[k, i - 1] -= 1.0
            if j != 0:
                A[k, j - 1] += 1.0

        return A

    def _solve(mask_keep):
        """Solve absolute shifts for the subset of kept pairwise measurements."""
        pairs_kept = [p for p, keep in zip(pairs, mask_keep) if keep]
        bx = shiftx[mask_keep]
        by = shifty[mask_keep]

        A = _build_A(pairs_kept)

        solx, *_ = np.linalg.lstsq(A, bx, rcond=None)
        soly, *_ = np.linalg.lstsq(A, by, rcond=None)

        abs_shiftx = np.zeros(nchannels, dtype=float)
        abs_shifty = np.zeros(nchannels, dtype=float)
        abs_shiftx[1:] = solx
        abs_shifty[1:] = soly

        return abs_shiftx, abs_shifty

    def _predict_pairwise(abs_shiftx, abs_shifty):
        """Predict pairwise shifts from solved absolute channel shifts."""
        predx = np.array(
            [abs_shiftx[j] - abs_shiftx[i] for i, j in pairs],
            dtype=float,
        )
        predy = np.array(
            [abs_shifty[j] - abs_shifty[i] for i, j in pairs],
            dtype=float,
        )
        return predx, predy

    def _robust_sigma(x):
        """Estimate a robust dispersion from the median absolute deviation."""
        med = bn.median(x)
        mad = bn.median(np.abs(x - med))
        return 1.4826 * mad if mad > 0 else 0.0

    # First pass: solve using all pairwise measurements.
    mask_keep_1 = np.ones(npairs, dtype=bool)
    abs_shiftx_1, abs_shifty_1 = _solve(mask_keep_1)

    predx_1, predy_1 = _predict_pairwise(abs_shiftx_1, abs_shifty_1)
    residx_1 = shiftx - predx_1
    residy_1 = shifty - predy_1
    resid_1 = np.sqrt(residx_1 ** 2 + residy_1 ** 2)

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
    abs_shiftx_2, abs_shifty_2 = _solve(mask_keep_2)

    predx_2, predy_2 = _predict_pairwise(abs_shiftx_2, abs_shifty_2)
    residx_2 = shiftx - predx_2
    residy_2 = shifty - predy_2
    resid_2 = np.sqrt(residx_2 ** 2 + residy_2 ** 2)

    # Recenter origin

    ox = np.mean(abs_shiftx_2)
    oy = np.mean(abs_shifty_2)

    abs_shiftx_2 = abs_shiftx_2 - ox
    abs_shifty_2 = abs_shifty_2 - oy

    info = {
        "pairs": pairs,
        "mask_keep_first_pass": mask_keep_1,
        "mask_keep_second_pass": mask_keep_2,
        "outlier_idx": outlier_idx,
        "outlier_pairs": [pairs[k] for k in outlier_idx],
        "residx_first_pass": residx_1,
        "residy_first_pass": residy_1,
        "resid_first_pass": resid_1,
        "residx_second_pass": residx_2,
        "residy_second_pass": residy_2,
        "resid_second_pass": resid_2,
        "median_resid_first_pass": med_r,
        "robust_sigma_first_pass": sigma_r,
        "origin_shift": (oy, ox),
    }

    return abs_shiftx_2, abs_shifty_2, info
