#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from smlmlp import block



@block()
def globloc_fit(
    crops,
    X0,
    Y0,
    /,
    *,
    x_shift_nm=0.,
    y_shift_nm=0.,
    rotation_deg=0.,
    x_shear=0.,
    y_shear=0.,
    optimizer="lm",
    estimator="mle",
    distribution="poisson",
    channels_pixels_nm=1.0,
    channels_gains=1.0,
    channels_QE=1.0,
    cuda=False,
    parallel=False,
):
    """Fit global localizations from channel crops.

    Parameters
    ----------
    crops : sequence of ndarray
        Crop stacks to fit.
    X0 : sequence of ndarray
        Crop x origins.
    Y0 : sequence of ndarray
        Crop y origins.
    x_shift_nm, y_shift_nm : float, optional
        Global translation parameters in nanometers.
    rotation_deg : float, optional
        Global rotation angle in degrees.
    x_shear, y_shear : float, optional
        Global shear parameters.
    optimizer : str, optional
        Optimizer key.
    estimator : str, optional
        Estimator key.
    distribution : str, optional
        Distribution key used by the estimator.
    channels_pixels_nm : float or sequence, optional
        Pixel size specification per channel.
    channels_gains : float or sequence, optional
        Gain value(s) used for fitted amplitudes.
    channels_QE : float or sequence, optional
        Quantum efficiency value(s) used for fitted amplitudes.
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
