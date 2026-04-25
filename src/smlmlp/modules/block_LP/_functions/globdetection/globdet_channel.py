#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from smlmlp import block



@block()
def globdet_channels(
    channels,
    *,
    x_shift_nm=0.,
    y_shift_nm=0.,
    rotation_deg=0.,
    x_shear=0.,
    y_shear=0.,
    cuda=False,
    parallel=False,
):
    """Create a global channel for detection.

    Parameters
    ----------
    channels : sequence of ndarray
        Input channel images.
    x_shift_nm, y_shift_nm : float, optional
        Global translation parameters in nanometers.
    rotation_deg : float, optional
        Global rotation angle in degrees.
    x_shear, y_shear : float, optional
        Global shear parameters.
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
    raise SyntaxError("Not implemented yet.")
