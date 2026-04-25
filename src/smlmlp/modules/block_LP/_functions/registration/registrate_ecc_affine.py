#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from smlmlp import block



@block()
def registrate_ecc_affine(
    optimized,
    /,
    ref_pix=1.0,
    *,
    cuda=False,
    parallel=False,
):
    """Estimate affine registration from enhanced correlation coefficient images.

    Parameters
    ----------
    optimized : sequence of ndarray
        Sequence of optimized registration images.
    ref_pix : float or tuple of float, optional
        Reference pixel size used to convert fitted parameters.
    cuda : bool, optional
        Whether to use CUDA execution.
    parallel : bool, optional
        Whether to use parallel execution.

    Returns
    -------
    tuple
        A tuple whose last item is an ``info`` dictionary.

    Raises
    ------
    SyntaxError
        Always raised because this block is not implemented.
    """
    return shiftx, shifty, angle, shearx, sheary, info
