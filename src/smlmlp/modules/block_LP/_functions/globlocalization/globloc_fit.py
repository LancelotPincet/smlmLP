#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from funclp import LM, MLE, LSE, Poisson, Normal, Gaussian2D, IsoGaussian, Spline2D, JointFunction, JointChannel
from arrlp import get_xp, nb_threads, coordinates
import numpy as np

SIGMA = 0.21 * 670 / 1.5



@block()
def globloc_fit(
    crops,
    X0,
    Y0,
    /,
    channels_models,
    channels_fit_inits,
    *,
    optimizer="lm",
    estimator="mle",
    distribution="poisson",
    channels_pixels_nm=1.0,
    channels_gains=1.0,
    channels_QE=1.0,
    cuda=False,
    parallel=False,
):
    """Fit global localizations from channel crops using joint fitting.

    Uses :class:`funclp.JointFunction` to fit all crops simultaneously, sharing
    position parameters across channels.

    Parameters
    ----------
    crops : sequence of ndarray
        Crop stacks to fit, one per channel, shaped ``(N, Y, X)``.
    X0 : sequence of ndarray
        Crop x origins in pixels.
    Y0 : sequence of ndarray
        Crop y origins in pixels.
    channels_models : sequence of str
        Model per channel, one of ``"gauss"``, ``"isogauss"``, ``"spline"``.
    channels_fit_inits : sequence of dict
        Initial fit parameters per channel.
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
        A tuple ``(mux, muy, info)`` where:

        - ``mux`` is the concatenated x localization array in nanometers,
        - ``muy`` is the concatenated y localization array in nanometers,
        - ``info`` is a dictionary with fitted parameter arrays.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = [np.array([10, 20], dtype=np.float32)]
    >>> y0 = [np.array([30, 40], dtype=np.float32)]
    >>> models = ["gauss"]
    >>> inits = [{"sigx": 90.0, "sigy": 90.0, "theta": 0.0, "theta_fit": False}]
    >>> mux, muy, info = globloc_fit(
    ...     crops, x0, y0,
    ...     channels_models=models,
    ...     channels_fit_inits=inits,
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ... )
    >>> mux.shape == muy.shape
    True

    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = [np.array([10, 20], dtype=np.float32)]
    >>> y0 = [np.array([30, 40], dtype=np.float32)]
    >>> models = ["isogauss"]
    >>> inits = [{"sig": 90.0}]
    >>> mux, muy, info = globloc_fit(
    ...     crops, x0, y0,
    ...     channels_models=models,
    ...     channels_fit_inits=inits,
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ... )
    >>> info['sigma'].ndim
    1
    """
    n_channels = len(crops)

    channels_pixels_nm = _normalize_channels_pixels_nm(
        channels_pixels_nm,
        n_channels,
    )
    channels_gains = _normalize_channels_parameter(channels_gains, n_channels)
    channels_QE = _normalize_channels_parameter(channels_QE, n_channels)

    optimizer_cls = _resolve_optimizer(optimizer)
    distribution = _resolve_distribution(distribution)
    estimator = _resolve_estimator(estimator, distribution)

    if len(channels_models) != n_channels:
        raise ValueError("channels_models must have same length as crops")
    if len(channels_fit_inits) != n_channels:
        raise ValueError("channels_fit_inits must have same length as crops")

    xp = get_xp(cuda)

    functions = []
    function_data = []
    all_crop_data = []
    all_xy_data = []

    for ch_idx, (crop, x0, y0, pixel, model_name, fit_init) in enumerate(
        zip(crops, X0, Y0, channels_pixels_nm, channels_models, channels_fit_inits)
    ):
        crop = xp.asarray(crop)
        _, height, width = crop.shape

        yy, xx = coordinates(shape=(height, width), pixel=pixel, cuda=cuda)
        x0 = xp.asarray(x0) * pixel[1]
        y0 = xp.asarray(y0) * pixel[0]

        mux = xp.full_like(x0, fill_value=(width - 1) / 2 * pixel[1])
        muy = xp.full_like(y0, fill_value=(height - 1) / 2 * pixel[0])
        amp = xp.max(crop, axis=(1, 2))
        offset = xp.min(crop, axis=(1, 2))

        model_name = model_name.lower()
        if model_name == "gauss":
            function = Gaussian2D(
                mux=mux,
                muy=muy,
                amp=amp,
                offset=offset,
                cuda=cuda,
                sigx=fit_init.get("sigx", SIGMA),
                sigy=fit_init.get("sigy", SIGMA),
                theta=fit_init.get("theta", 0.0),
                pixx=pixel[1],
                pixy=pixel[0],
                theta_fit=fit_init.get("theta_fit", False),
            )
        elif model_name == "isogauss":
            function = IsoGaussian(
                mux=mux,
                muy=muy,
                amp=amp,
                offset=offset,
                cuda=cuda,
                sig=fit_init.get("sig", SIGMA),
                pixx=pixel[1],
                pixy=pixel[0],
            )
        elif model_name == "spline":
            function = Spline2D(
                mux=mux,
                muy=muy,
                amp=amp,
                offset=offset,
                cuda=cuda,
                tx=fit_init.get("tx"),
                ty=fit_init.get("ty"),
                coeffs=fit_init.get("coeffs"),
                pixx=pixel[1],
                pixy=pixel[0],
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        functions.append(function)
        function_data.append({"x0": x0, "y0": y0, "pixel": pixel})
        all_crop_data.append(crop)
        all_xy_data.append({"x": xx, "y": yy})

    if n_channels == 1:
        function = functions[0]
        data = function_data[0]
        fit = optimizer_cls(function, estimator)
        if cuda:
            fit(all_crop_data[0], all_xy_data[0]["x"], all_xy_data[0]["y"])
        else:
            with nb_threads(parallel):
                fit(all_crop_data[0], all_xy_data[0]["x"], all_xy_data[0]["y"])
    else:
        prefixes = [f"ch{i}" for i in range(n_channels)]
        joint_channels = [
            JointChannel(func, prefix=prefix)
            for func, prefix in zip(functions, prefixes)
        ]

        shared_vars = {
            "x": ["x"] * n_channels,
            "y": ["y"] * n_channels,
        }
        shared_params = {
            "mux": ["mux"] * n_channels,
            "muy": ["muy"] * n_channels,
        }

        joint_function = JointFunction(
            joint_channels,
            shared_variables=shared_vars,
            shared_parameters=shared_params,
        )

        fit = optimizer_cls(joint_function, estimator)
        with nb_threads(parallel):
            fit(all_crop_data, all_xy_data)

    mux_all = []
    muy_all = []
    amp_all = []
    offset_all = []
    sigmax_all = []
    sigmay_all = []
    sigma_all = []

    for ch_idx, (function, x0, y0, pixel, gain, qe, model_name) in enumerate(
        zip(functions, X0, Y0, channels_pixels_nm, channels_gains, channels_QE, channels_models)
    ):
        x0 = xp.asarray(x0) * pixel[1]
        y0 = xp.asarray(y0) * pixel[0]

        mux = function.mux + x0
        muy = function.muy + y0
        amp = function.amp / qe * gain
        offset = function.offset / qe * gain

        if cuda:
            mux = xp.asnumpy(mux)
            muy = xp.asnumpy(muy)
            amp = xp.asnumpy(amp)
            offset = xp.asnumpy(offset)

        mux_all.append(mux)
        muy_all.append(muy)
        amp_all.append(amp)
        offset_all.append(offset)

        model_name = model_name.lower()
        if model_name == "gauss":
            sigx = function.sigx
            sigy = function.sigy
            if cuda:
                sigx = xp.asnumpy(sigx)
                sigy = xp.asnumpy(sigy)
            sigmax_all.append(sigx)
            sigmay_all.append(sigy)
        elif model_name == "isogauss":
            sig = function.sig
            if cuda:
                sig = xp.asnumpy(sig)
            sigma_all.append(sig)
        elif model_name == "spline":
            pass

    info = {
        "amp": np.hstack(amp_all),
        "offset": np.hstack(offset_all),
    }
    if sigmax_all:
        info["sigmax"] = np.hstack(sigmax_all)
        info["sigmay"] = np.hstack(sigmay_all)
    if sigma_all:
        info["sigma"] = np.hstack(sigma_all)

    return np.hstack(mux_all), np.hstack(muy_all), info



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