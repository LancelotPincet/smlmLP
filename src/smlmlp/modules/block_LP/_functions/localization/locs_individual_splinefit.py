#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from funclp import LM, MLE, LSE, Poisson, Normal, Spline3D
from arrlp import get_xp, nb_threads, coordinates
import numpy as np

SIGMA = 0.21 * 670 / 1.5



# %% Function
@block()
def locs_individual_splinefit(
    crops,
    X0,
    Y0,
    /,
    *,
    optimizer="lm",
    estimator="mle",
    distribution="poisson",
    channels_pixels_nm=1.0,
    channels_gains=1.0,
    channels_QE=1.0,
    cuda=False,
    parallel=False,
    channels_psf_xtangents=None,
    channels_psf_ytangents=None,
    channels_psf_ztangents=None,
    channels_psf_coeffs=None,
):
    """
    Fit each crop independently with a 3D spline PSF model.

    The function loops through channels, initializes a :class:`funclp.Spline3D`
    model per event, runs the selected optimizer/estimator combination, and
    returns localized coordinates with fitted photometric values.

    Parameters
    ----------
    crops : sequence of array-like
        Sequence of crop stacks, one per channel, shaped ``(N, Y, X)``.
    X0 : sequence of array-like
        Sequence of x-origin pixel indices for each crop.
    Y0 : sequence of array-like
        Sequence of y-origin pixel indices for each crop.
    optimizer : str, optional
        Optimizer key.
    estimator : str, optional
        Estimator key.
    distribution : str, optional
        Distribution key used by the estimator.
    channels_pixels_nm : float or sequence, optional
        Pixel size specification per channel.
    channels_gains : float or sequence, optional
        Gain value(s) used to convert fitted amplitudes.
    channels_QE : float or sequence, optional
        Quantum efficiency value(s) used to convert fitted amplitudes.
    cuda : bool, optional
        Whether to run the fit on GPU.
    parallel : bool, optional
        Whether to enable CPU parallelization.
    channels_psf_xtangents : sequence
        Spline x tangents, one set per channel.
    channels_psf_ytangents : sequence
        Spline y tangents, one set per channel.
    channels_psf_ztangents : sequence
        Spline z tangents, one set per channel.
    channels_psf_coeffs : sequence
        Spline coefficients, one set per channel.

    Returns
    -------
    tuple
        A tuple ``(mux, muy, muz, info)`` where:

        - ``mux`` is the concatenated x localization array in nanometers,
        - ``muy`` is the concatenated y localization array in nanometers,
        - ``muz`` is the concatenated z localization array,
        - ``info`` is a dictionary with fitted parameter arrays.

        ``info`` contains:

        ``'amp'``
            Concatenated converted amplitudes.
        ``'offset'``
            Concatenated converted offsets.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = [np.array([10, 20], dtype=np.float32)]
    >>> y0 = [np.array([30, 40], dtype=np.float32)]
    >>> tx = [np.linspace(-1.0, 1.0, 5, dtype=np.float32)]
    >>> ty = [np.linspace(-1.0, 1.0, 5, dtype=np.float32)]
    >>> tz = [np.linspace(-0.5, 0.5, 5, dtype=np.float32)]
    >>> coeffs = [np.ones((4, 4, 4), dtype=np.float32)]
    >>> mux, muy, muz, info = locs_individual_splinefit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ...     channels_psf_xtangents=tx,
    ...     channels_psf_ytangents=ty,
    ...     channels_psf_ztangents=tz,
    ...     channels_psf_coeffs=coeffs,
    ... )
    >>> mux.shape == muy.shape == muz.shape
    True
    >>> sorted(info)
    ['amp', 'offset']

    >>> mux, muy, muz, info = locs_individual_splinefit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 120.0)],
    ...     channels_psf_xtangents=tx,
    ...     channels_psf_ytangents=ty,
    ...     channels_psf_ztangents=tz,
    ...     channels_psf_coeffs=coeffs,
    ... )
    >>> info['amp'].ndim
    1
    """
    n_channels = len(crops)

    channels_pixels_nm = _normalize_channels_pixels_nm(
        channels_pixels_nm,
        n_channels,
    )
    channels_gains = _normalize_channels_parameter(channels_gains, n_channels)
    channels_QE = _normalize_channels_parameter(channels_QE, n_channels)

    optimizer = _resolve_optimizer(optimizer)
    distribution = _resolve_distribution(distribution)
    estimator = _resolve_estimator(estimator, distribution)

    if channels_psf_xtangents is None:
        raise SyntaxError("channels_psf_xtangents must be specified as a kwarg")
    if len(channels_psf_xtangents) != n_channels:
        raise ValueError("channels_psf_xtangents does not have the same length as crops")

    if channels_psf_ytangents is None:
        raise SyntaxError("channels_psf_ytangents must be specified as a kwarg")
    if len(channels_psf_ytangents) != n_channels:
        raise ValueError("channels_psf_ytangents does not have the same length as crops")

    if channels_psf_ztangents is None:
        raise SyntaxError("channels_psf_ztangents must be specified as a kwarg")
    if len(channels_psf_ztangents) != n_channels:
        raise ValueError("channels_psf_ztangents does not have the same length as crops")

    if channels_psf_coeffs is None:
        raise SyntaxError("channels_psf_coeffs must be specified as a kwarg")
    if len(channels_psf_coeffs) != n_channels:
        raise ValueError("channels_psf_coeffs does not have the same length as crops")

    fit_kwargs = [
        dict(
            tx=tx,
            ty=ty,
            tz=tz,
            coeffs=coeffs,
        )
        for tx, ty, tz, coeffs in zip(
            channels_psf_coeffs,
            channels_psf_ytangents,
            channels_psf_ztangents,
            channels_psf_coeffs,
        )
    ]

    xp = get_xp(cuda)
    mux_all = []
    muy_all = []
    muz_all = []
    amp_all = []
    offset_all = []

    for crop, x0, y0, pixel, gain, qe, function_kw in zip(
        crops,
        X0,
        Y0,
        channels_pixels_nm,
        channels_gains,
        channels_QE,
        fit_kwargs,
    ):
        crop = xp.asarray(crop)
        _, height, width = crop.shape

        yy, xx = coordinates(shape=(height, width), pixel=pixel, cuda=cuda)
        zz = xp.zeros_like(xx)

        x0 = xp.asarray(x0) * pixel[1]
        y0 = xp.asarray(y0) * pixel[0]

        mux = xp.full_like(x0, fill_value=(width - 1) / 2 * pixel[1])
        muy = xp.full_like(y0, fill_value=(height - 1) / 2 * pixel[0])
        muz = xp.zeros_like(x0)
        amp = xp.max(crop, axis=(1, 2))
        offset = xp.min(crop, axis=(1, 2))

        function = Spline3D(
            mux=mux,
            muy=muy,
            muz=muz,
            amp=amp,
            offset=offset,
            cuda=cuda,
            **function_kw,
        )

        fit = optimizer(function, estimator)
        if cuda:
            fit(crop, xx, yy, zz)
        else:
            with nb_threads(parallel):
                fit(crop, xx, yy, zz)

        mux, muy, muz = function.mux, function.muy, function.muz
        mux += x0
        muy += y0
        amp = function.amp / qe * gain
        offset = function.offset / qe * gain

        if cuda:
            mux = xp.asnumpy(mux)
            muy = xp.asnumpy(muy)
            muz = xp.asnumpy(muz)
            amp = xp.asnumpy(amp)
            offset = xp.asnumpy(offset)

        mux_all.append(mux)
        muy_all.append(muy)
        muz_all.append(muz)
        amp_all.append(amp)
        offset_all.append(offset)

    info = {
        "amp": np.hstack(amp_all),
        "offset": np.hstack(offset_all),
    }

    return np.hstack(mux_all), np.hstack(muy_all), np.hstack(muz_all), info



def _normalize_channels_pixels_nm(channels_pixels_nm, n_channels):
    """Normalize pixel sizes to one ``(py, px)`` tuple per channel."""
    try:
        if len(channels_pixels_nm) != n_channels:
            if len(channels_pixels_nm) == 2:
                channels_pixels_nm = [channels_pixels_nm for _ in range(n_channels)]
            else:
                raise ValueError(
                    "channel_mean_radius_pix does not have the same length as channels"
                )
    except TypeError:
        channels_pixels_nm = [
            (channels_pixels_nm, channels_pixels_nm)
            for _ in range(n_channels)
        ]

    return channels_pixels_nm



def _normalize_channels_parameter(values, n_channels):
    """Normalize scalar/per-channel values to a per-channel sequence."""
    try:
        if len(values) != n_channels:
            raise ValueError(
                "channel_mean_radius_pix does not have the same length as channels"
            )
    except TypeError:
        values = [values for _ in range(n_channels)]

    return values



def _resolve_optimizer(optimizer):
    """Resolve optimizer key to optimizer class."""
    match optimizer.lower():
        case "lm":
            return LM
        case _:
            raise SyntaxError(f"Optimizer {optimizer} is not recognized")



def _resolve_distribution(distribution):
    """Resolve distribution key to instantiated distribution."""
    match distribution.lower():
        case "normal":
            return Normal()
        case "poisson":
            return Poisson()
        case _:
            raise SyntaxError(f"Distribution {distribution} is not recognized")



def _resolve_estimator(estimator, distribution):
    """Resolve estimator key to instantiated estimator."""
    match estimator.lower():
        case "mle":
            return MLE(distribution)
        case "lse":
            return LSE()
        case _:
            raise SyntaxError(f"Estimator {estimator} is not recognized")
