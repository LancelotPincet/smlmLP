#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np
from arrlp import coordinates, get_xp, nb_threads
from funclp import Gaussian2D, IsoGaussian, LM, LSE, MLE, Normal, Poisson, Spline3D

from smlmlp import block
from ._channel_values import split_channel_origins, stack_channel_values

SIGMA = 0.21 * 670 / 1.5
FIT_MODELS = ("isogauss", "gauss", "spline")



@block()
def locs_individual_fit(
    crops,
    X0,
    Y0,
    /,
    ch=None,
    *,
    channels_fit_models="isogauss",
    optimizer="lm",
    estimator="mle",
    distribution="poisson",
    channels_pixels_nm=1.0,
    channels_gains=1.0,
    channels_QE=1.0,
    cuda=False,
    parallel=False,
    channels_psf_sigmas_nm=SIGMA,
    channels_psf_xsigmas_nm=SIGMA,
    channels_psf_ysigmas_nm=SIGMA,
    channels_psf_thetas_deg=0.0,
    channels_fit_thetas=False,
    channels_psf_3d_xtangents=None,
    channels_psf_3d_ytangents=None,
    channels_psf_3d_ztangents=None,
    channels_psf_3d_spline_coeffs=None,
):
    """
    Fit each crop stack with a channel-specific localization model.

    Parameters
    ----------
    crops : sequence of array-like
        Crop stacks to fit, one stack per channel. Each stack must be shaped
        ``(N, Y, X)`` where ``N`` is the number of events in that channel.
    X0 : array-like
        Detection-aligned 1D vector of crop x-origin pixel indices.
    Y0 : array-like
        Detection-aligned 1D vector of crop y-origin pixel indices.
    ch : array-like or None, optional
        One-based channel index for each detection. Required when ``crops`` has
        several channels.
    channels_fit_models : str or sequence of str, default="isogauss"
        Model used for each channel crop stack. Accepted values are
        ``"isogauss"``, ``"gauss"``, and ``"spline"``.
    optimizer : {"lm"}, default="lm"
        Optimizer used to fit each model.
    estimator : {"mle", "lse"}, default="mle"
        Estimator used by the optimizer.
    distribution : {"poisson", "normal"}, default="poisson"
        Noise distribution used by maximum-likelihood estimators.
    channels_pixels_nm : float, tuple, or sequence, default=1.0
        Pixel size specification. A scalar is used for both axes and all
        channels, a ``(py, px)`` tuple is broadcast to all channels, and a
        sequence provides one ``(py, px)`` pair per channel.
    channels_gains : float or sequence, default=1.0
        Gain values used to convert fitted amplitudes and offsets.
    channels_QE : float or sequence, default=1.0
        Quantum efficiencies used to convert fitted amplitudes and offsets.
    cuda : bool, default=False
        Whether to run fits on CUDA when supported.
    parallel : bool, default=False
        Whether to use threaded CPU execution.
    channels_psf_sigmas_nm : float or sequence, default=SIGMA
        Initial isotropic sigma values for ``"isogauss"`` channels.
    channels_psf_xsigmas_nm : float or sequence, default=SIGMA
        Initial x sigma values for ``"gauss"`` channels.
    channels_psf_ysigmas_nm : float or sequence, default=SIGMA
        Initial y sigma values for ``"gauss"`` channels.
    channels_psf_thetas_deg : float or sequence, default=0.0
        Initial theta values, in degrees, for ``"gauss"`` channels.
    channels_fit_thetas : bool or sequence, default=False
        Whether to fit theta for ``"gauss"`` channels.
    channels_psf_3d_xtangents : sequence, optional
        Spline x tangents for ``"spline"`` channels.
    channels_psf_3d_ytangents : sequence, optional
        Spline y tangents for ``"spline"`` channels.
    channels_psf_3d_ztangents : sequence, optional
        Spline z tangents for ``"spline"`` channels.
    channels_psf_3d_spline_coeffs : sequence, optional
        Spline coefficients for ``"spline"`` channels.

    Returns
    -------
    tuple
        A tuple ``(mux, muy, muz, info)`` where ``mux`` and ``muy`` are
        detection-aligned fitted coordinates in nanometers and ``muz`` contains fitted
        spline z coordinates or ``np.nan`` for 2D models. ``info`` contains
        detection-aligned ``"amp"``, ``"offset"``, ``"sigma"``, ``"sigmax"``, and
        ``"sigmay"`` arrays plus a ``"models"`` list. Sigma arrays always match
        the localization length and contain ``np.nan`` where the parameter does
        not apply to the channel model.

    Raises
    ------
    ValueError
        If per-channel inputs have incompatible lengths or a model name is not
        one of ``"isogauss"``, ``"gauss"``, and ``"spline"``.
    SyntaxError
        If the optimizer, estimator, distribution, or required spline metadata
        is missing or unsupported.

    Notes
    -----
    1. Channel model names and per-channel parameters are normalized to match
       the crop list length.
    2. ``X0`` and ``Y0`` are split by ``ch`` so origins match each crop stack.
    3. Each channel selects ``IsoGaussian``, ``Gaussian2D``, or ``Spline3D`` and
       initializes local coordinates at the crop center.
    4. The selected optimizer updates model parameters in local nanometer
       coordinates before crop origins are added back.
    5. Coordinates and fitted parameter arrays are remapped to detection order.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = np.array([10, 20], dtype=np.float32)
    >>> y0 = np.array([30, 40], dtype=np.float32)
    >>> mux, muy, muz, info = locs_individual_fit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_fit_models=["isogauss"],
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ... )
    >>> mux.shape == muy.shape
    True

    >>> tx = [np.linspace(-300.0, 300.0, 8, dtype=np.float32)]
    >>> ty = [np.linspace(-300.0, 300.0, 8, dtype=np.float32)]
    >>> tz = [np.linspace(-300.0, 300.0, 8, dtype=np.float32)]
    >>> coeffs = [np.ones((4, 4, 4), dtype=np.float32)]
    >>> mux, muy, muz, info = locs_individual_fit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_fit_models=["spline"],
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ...     channels_psf_3d_xtangents=tx,
    ...     channels_psf_3d_ytangents=ty,
    ...     channels_psf_3d_ztangents=tz,
    ...     channels_psf_3d_spline_coeffs=coeffs,
    ... )
    >>> info["models"]
    ['spline']
    """
    n_channels = len(crops)
    X0, Y0, positions = split_channel_origins(crops, X0, Y0, ch, cuda=cuda)

    channels_fit_models = _normalize_channels_fit_models(
        channels_fit_models,
        n_channels,
    )
    channels_pixels_nm = _normalize_channels_pixels_nm(
        channels_pixels_nm,
        n_channels,
    )
    channels_gains = _normalize_channels_parameter(channels_gains, n_channels)
    channels_QE = _normalize_channels_parameter(channels_QE, n_channels)
    channels_psf_sigmas_nm = _normalize_channels_parameter(
        channels_psf_sigmas_nm,
        n_channels,
    )
    channels_psf_xsigmas_nm = _normalize_channels_parameter(
        channels_psf_xsigmas_nm,
        n_channels,
    )
    channels_psf_ysigmas_nm = _normalize_channels_parameter(
        channels_psf_ysigmas_nm,
        n_channels,
    )
    channels_psf_thetas_deg = _normalize_channels_parameter(
        channels_psf_thetas_deg,
        n_channels,
    )
    channels_fit_thetas = _normalize_channels_parameter(channels_fit_thetas, n_channels)

    needs_spline = "spline" in channels_fit_models
    channels_psf_3d_xtangents = _normalize_spline_parameter(
        channels_psf_3d_xtangents,
        n_channels,
        "channels_psf_3d_xtangents",
        required=needs_spline,
    )
    channels_psf_3d_ytangents = _normalize_spline_parameter(
        channels_psf_3d_ytangents,
        n_channels,
        "channels_psf_3d_ytangents",
        required=needs_spline,
    )
    channels_psf_3d_ztangents = _normalize_spline_parameter(
        channels_psf_3d_ztangents,
        n_channels,
        "channels_psf_3d_ztangents",
        required=needs_spline,
    )
    channels_psf_3d_spline_coeffs = _normalize_spline_parameter(
        channels_psf_3d_spline_coeffs,
        n_channels,
        "channels_psf_3d_spline_coeffs",
        required=needs_spline,
    )

    optimizer_cls = _resolve_optimizer(optimizer)
    distribution = _resolve_distribution(distribution)
    estimator = _resolve_estimator(estimator, distribution)

    xp = get_xp(cuda)
    mux_all = []
    muy_all = []
    amp_all = []
    offset_all = []
    sigma_all = []
    sigmax_all = []
    sigmay_all = []
    muz_all = []

    iterator = zip(
        crops,
        X0,
        Y0,
        channels_pixels_nm,
        channels_gains,
        channels_QE,
        channels_fit_models,
        channels_psf_sigmas_nm,
        channels_psf_xsigmas_nm,
        channels_psf_ysigmas_nm,
        channels_psf_thetas_deg,
        channels_fit_thetas,
        channels_psf_3d_xtangents,
        channels_psf_3d_ytangents,
        channels_psf_3d_ztangents,
        channels_psf_3d_spline_coeffs,
    )

    for values in iterator:
        (
            crop,
            x0,
            y0,
            pixel,
            gain,
            qe,
            model,
            sigma,
            sigx,
            sigy,
            theta,
            fit_theta,
            tx,
            ty,
            tz,
            coeffs,
        ) = values

        crop = xp.asarray(crop)
        _, height, width = crop.shape
        yy, xx = coordinates(shape=(height, width), pixel=pixel, cuda=cuda)

        x0 = xp.asarray(x0) * pixel[1]
        y0 = xp.asarray(y0) * pixel[0]

        if len(crop) == 0:
            empty = xp.empty(0, dtype=xp.float32)
            mux_all.append(empty)
            muy_all.append(empty)
            muz_all.append(empty)
            amp_all.append(empty)
            offset_all.append(empty)
            sigma_all.append(empty)
            sigmax_all.append(empty)
            sigmay_all.append(empty)
            continue

        mux = xp.full_like(x0, fill_value=(width - 1) / 2 * pixel[1])
        muy = xp.full_like(y0, fill_value=(height - 1) / 2 * pixel[0])
        amp = xp.max(crop, axis=(1, 2))
        offset = xp.min(crop, axis=(1, 2))

        function = _make_model(
            model=model,
            mux=mux,
            muy=muy,
            amp=amp,
            offset=offset,
            pixel=pixel,
            sigma=sigma,
            sigx=sigx,
            sigy=sigy,
            theta=theta,
            fit_theta=fit_theta,
            tx=tx,
            ty=ty,
            tz=tz,
            coeffs=coeffs,
            cuda=cuda,
        )

        fit = optimizer_cls(function, estimator)
        if model == "spline":
            zz = xp.zeros_like(xx)
            if cuda:
                fit(crop, xx, yy, zz)
            else:
                with nb_threads(parallel):
                    fit(crop, xx, yy, zz)
            muz = function.muz
        else:
            if cuda:
                fit(crop, xx, yy)
            else:
                with nb_threads(parallel):
                    fit(crop, xx, yy)
            muz = xp.full_like(x0, fill_value=np.nan, dtype=xp.float32)

        mux = function.mux + x0
        muy = function.muy + y0
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

        nan_sigma = xp.full_like(x0, fill_value=np.nan, dtype=xp.float32)
        if model == "isogauss":
            sig = function.sig
            sigx = nan_sigma
            sigy = nan_sigma
        elif model == "gauss":
            sigx = function.sigx
            sigy = function.sigy
            sig = xp.sqrt(sigx * sigy)
        else:
            sig = nan_sigma
            sigx = nan_sigma
            sigy = nan_sigma

        if cuda:
            sig = xp.asnumpy(sig)
            sigx = xp.asnumpy(sigx)
            sigy = xp.asnumpy(sigy)
        sigma_all.append(sig)
        sigmax_all.append(sigx)
        sigmay_all.append(sigy)

    info = {
        "amp": stack_channel_values(amp_all, positions),
        "offset": stack_channel_values(offset_all, positions),
        "sigma": stack_channel_values(sigma_all, positions),
        "sigmax": stack_channel_values(sigmax_all, positions),
        "sigmay": stack_channel_values(sigmay_all, positions),
        "models": channels_fit_models,
    }

    return stack_channel_values(mux_all, positions), stack_channel_values(muy_all, positions), stack_channel_values(muz_all, positions), info



def _normalize_channels_fit_models(values, n_channels):
    """Normalize and validate per-channel model names."""
    if isinstance(values, str):
        values = [values for _ in range(n_channels)]
    elif len(values) != n_channels:
        raise ValueError("channels_fit_models must have same length as crops")

    models = [str(value).lower() for value in values]
    invalid = [model for model in models if model not in FIT_MODELS]
    if invalid:
        raise ValueError(
            f"channels_fit_models contains unsupported values {invalid}; "
            f"expected one of {FIT_MODELS}"
        )
    return models



def _normalize_channels_pixels_nm(channels_pixels_nm, n_channels):
    """Normalize pixel sizes to one ``(py, px)`` tuple per channel."""
    try:
        if len(channels_pixels_nm) != n_channels:
            if len(channels_pixels_nm) == 2:
                channels_pixels_nm = [channels_pixels_nm for _ in range(n_channels)]
            else:
                raise ValueError("channels_pixels_nm must have same length as crops")
    except TypeError:
        channels_pixels_nm = [
            (channels_pixels_nm, channels_pixels_nm)
            for _ in range(n_channels)
        ]

    return channels_pixels_nm



def _normalize_channels_parameter(values, n_channels):
    """Normalize scalar/per-channel values to a per-channel sequence."""
    if isinstance(values, str):
        return [values for _ in range(n_channels)]

    try:
        if len(values) != n_channels:
            raise ValueError("parameter must have same length as crops")
    except TypeError:
        values = [values for _ in range(n_channels)]

    return values



def _normalize_spline_parameter(values, n_channels, name, required):
    """Normalize spline metadata to one object per channel."""
    if values is None:
        if required:
            raise SyntaxError(f"{name} must be specified for spline channels")
        return [None for _ in range(n_channels)]

    if isinstance(values, np.ndarray):
        if n_channels != 1:
            raise ValueError(f"{name} must have same length as crops")
        return [values]

    if len(values) != n_channels:
        raise ValueError(f"{name} must have same length as crops")
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



def _make_model(
    *,
    model,
    mux,
    muy,
    amp,
    offset,
    pixel,
    sigma,
    sigx,
    sigy,
    theta,
    fit_theta,
    tx,
    ty,
    tz,
    coeffs,
    cuda,
):
    """Instantiate the fit model selected for one channel."""
    if model == "isogauss":
        return IsoGaussian(
            mux=mux,
            muy=muy,
            amp=amp,
            offset=offset,
            sig=sigma,
            pixx=pixel[1],
            pixy=pixel[0],
            cuda=cuda,
        )
    if model == "gauss":
        return Gaussian2D(
            mux=mux,
            muy=muy,
            amp=amp,
            offset=offset,
            sigx=sigx,
            sigy=sigy,
            theta=theta,
            theta_fit=fit_theta,
            pixx=pixel[1],
            pixy=pixel[0],
            cuda=cuda,
        )
    if model == "spline":
        return Spline3D(
            mux=mux,
            muy=muy,
            muz=mux * 0,
            amp=amp,
            offset=offset,
            tx=tx,
            ty=ty,
            tz=tz,
            coeffs=coeffs,
            cuda=cuda,
        )

    raise ValueError(f"Unknown model: {model}")



if __name__ == "__main__":
    from corelp import test

    test(__file__)
