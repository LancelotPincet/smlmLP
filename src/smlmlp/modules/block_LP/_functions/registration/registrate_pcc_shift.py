#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_crosscorr, get_xp
import numpy as np



# %% Function
@block()
def registrate_pcc_shift(
    optimized,
    /,
    ref_pix=1.0,
    *,
    cuda=False,
    parallel=False,
):
    """
    Estimate redundant pairwise shifts from phase cross-correlation images.

    This function computes the phase cross-correlation between every pair of
    optimized channels and estimates a subpixel shift for each frame from the
    cross-correlation peak.

    Parameters
    ----------
    optimized : sequence of ndarray
        Sequence of optimized image stacks, one per channel.
    ref_pix : float or tuple of float, optional
        Reference pixel size used to convert shifts to physical units. If a
        scalar is provided, it is applied to both y and x as ``(ref_pix,
        ref_pix)``.
    cuda : bool, optional
        Whether to enable CUDA processing.
    parallel : bool, optional
        Whether to enable parallel processing.

    Returns
    -------
    tuple
        A tuple ``(CC, shiftx, shifty, info)`` where:

        - ``CC`` is the list of phase cross-correlation stacks for all channel
          pairs,
        - ``shiftx`` is the list of per-frame x shifts for all channel pairs,
        - ``shifty`` is the list of per-frame y shifts for all channel pairs,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'ref_pix'``
            Reference pixel size used for shift conversion.
        ``'pairs'``
            List of channel index pairs ``(i, j)`` corresponding to the
            cross-correlation and shift outputs.

    Notes
    -----
    For each channel pair, the peak of the phase cross-correlation is detected
    on each frame. When the peak is not on the image border, a quadratic fit
    on the local 3x3 neighborhood is used to estimate a subpixel offset.

    Examples
    --------
    >>> import numpy as np
    >>> ch1 = np.random.rand(4, 16, 16).astype(np.float32)
    >>> ch2 = np.random.rand(4, 16, 16).astype(np.float32)
    >>> CC, shiftx, shifty, info = registrate_pcc_shift([ch1, ch2])
    >>> len(CC)
    1
    >>> info["pairs"]
    [(0, 1)]

    >>> ch3 = np.random.rand(4, 16, 16).astype(np.float32)
    >>> CC, shiftx, shifty, info = registrate_pcc_shift(
    ...     [ch1, ch2, ch3],
    ...     ref_pix=(100.0, 120.0),
    ... )
    >>> len(info["pairs"])
    3
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Normalize the reference pixel size to a (y, x) pair.
    try:
        if len(ref_pix) != 2:
            raise ValueError("ref_pix does not have 2 values (y, x)")
    except TypeError:
        ref_pix = (ref_pix, ref_pix)

    # Compute the phase cross-correlation for each channel pair and estimate
    # the corresponding per-frame shifts.
    CC = []
    shiftx = []
    shifty = []
    pairs = []

    for i in range(len(optimized)):
        for j in range(i + 1, len(optimized)):
            cc = img_crosscorr(
                optimized[i],
                optimized[j],
                phase=True,
                cuda=cuda,
                parallel=parallel,
                stack=True,
            )
            if cuda:
                cc = xp.asnumpy(cc)

            dx, dy = subpixel_peak_stack(cc, ref_pix=ref_pix)

            CC.append(cc)
            shiftx.append(dx)
            shifty.append(dy)
            pairs.append((i, j))

    info = {
        "ref_pix": ref_pix,
        "pairs": pairs,
    }

    return CC, shiftx, shifty, info



def subpixel_peak_stack(cc, ref_pix=(1.0, 1.0)):
    """Estimate one subpixel peak position per frame from a CC stack."""
    nframes = cc.shape[0]
    shiftx = np.empty(nframes, dtype=float)
    shifty = np.empty(nframes, dtype=float)

    ny, nx = cc.shape[-2:]

    for k in range(nframes):
        c = cc[k]

        iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)

        # Skip subpixel refinement when the peak touches the border.
        if not (0 < iy < ny - 1 and 0 < ix < nx - 1):
            dy_sub, dx_sub = 0.0, 0.0
        else:
            win = c[iy - 1:iy + 2, ix - 1:ix + 2].astype(float)
            dy_sub, dx_sub = subpixel_peak_2d(win)

        dy = ((iy - ny // 2) + dy_sub) * ref_pix[0]
        dx = ((ix - nx // 2) + dx_sub) * ref_pix[1]

        shiftx[k] = dx
        shifty[k] = dy

    return shiftx, shifty



def subpixel_peak_2d(win):
    """Estimate the subpixel peak offset from a 3x3 quadratic fit."""
    # Coordinates relative to the window center.
    y, x = np.mgrid[-1:2, -1:2]

    # Build the quadratic design matrix and flatten the input window.
    X = np.column_stack(
        [
            x.ravel() ** 2,
            y.ravel() ** 2,
            x.ravel() * y.ravel(),
            x.ravel(),
            y.ravel(),
            np.ones(9),
        ]
    )
    Z = win.ravel()

    # Solve the least-squares problem for the quadratic coefficients.
    coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    a, b, c, d, e, f = coeffs

    # Solve for the stationary point of the fitted quadratic surface.
    A = np.array([[2 * a, c], [c, 2 * b]])
    bvec = -np.array([d, e])

    try:
        offset = np.linalg.solve(A, bvec)
    except np.linalg.LinAlgError:
        offset = np.array([0.0, 0.0])

    dx_sub, dy_sub = offset
    return dy_sub, dx_sub