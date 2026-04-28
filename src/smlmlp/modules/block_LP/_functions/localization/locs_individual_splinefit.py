#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block, locs_individual_barycenter
from funclp import LM, MLE, LSE, Poisson, Normal, Spline3D
from arrlp import get_xp, nb_threads, coordinates
from ._channel_values import split_channel_origins, stack_channel_values

SIGMA = 0.21 * 670 / 1.5



@block()
def locs_individual_splinefit(
    crops,
    X0,
    Y0,
    /,
    ch=None,
    *,
    optimizer="lm",
    estimator="mle",
    distribution="poisson",
    channels_pixels_nm=1.0,
    channels_gains=1.0,
    channels_QE=1.0,
    cuda=False,
    parallel=False,
    channels_psf_3d_xtangents=None,
    channels_psf_3d_ytangents=None,
    channels_psf_3d_ztangents=None,
    channels_psf_3d_spline_coeffs=None,
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
    X0 : array-like
        Detection-aligned 1D vector of x-origin pixel indices.
    Y0 : array-like
        Detection-aligned 1D vector of y-origin pixel indices.
    ch : array-like or None, optional
        One-based channel index for each detection. Required when ``crops`` has
        several channels.
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
    channels_psf_3d_xtangents : sequence
        Spline x tangents, one set per channel.
    channels_psf_3d_ytangents : sequence
        Spline y tangents, one set per channel.
    channels_psf_3d_ztangents : sequence
        Spline z tangents, one set per channel.
    channels_psf_3d_spline_coeffs : sequence
        Spline coefficients, one set per channel.

    Returns
    -------
    tuple
        A tuple ``(mux, muy, muz, info)`` where:

        - ``mux`` is the detection-aligned x localization array in nanometers,
        - ``muy`` is the detection-aligned y localization array in nanometers,
        - ``muz`` is the detection-aligned z localization array,
        - ``info`` is a dictionary with fitted parameter arrays.

        ``info`` contains:

        ``'amp'``
            Detection-aligned converted amplitudes.
        ``'offset'``
            Detection-aligned converted offsets.

    Notes
    -----
    1. ``X0`` and ``Y0`` are split by ``ch`` so each origin vector follows the
       crop order inside its channel stack.
    2. A local x/y grid and zero z grid are built for each channel, and spline
       models are initialized at the crop center with zero z.
    3. The optimizer updates local spline parameters, local coordinates are
       shifted by crop origins, and all outputs are remapped to detection order.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = np.array([10, 20], dtype=np.float32)
    >>> y0 = np.array([30, 40], dtype=np.float32)
    >>> tx = [np.linspace(-1.0, 1.0, 5, dtype=np.float32)]
    >>> ty = [np.linspace(-1.0, 1.0, 5, dtype=np.float32)]
    >>> tz = [np.linspace(-0.5, 0.5, 5, dtype=np.float32)]
    >>> coeffs = [np.ones((4, 4, 4), dtype=np.float32)]
    >>> mux, muy, muz, info = locs_individual_splinefit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ...     channels_psf_3d_xtangents=tx,
    ...     channels_psf_3d_ytangents=ty,
    ...     channels_psf_3d_ztangents=tz,
    ...     channels_psf_3d_spline_coeffs=coeffs,
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
    ...     channels_psf_3d_xtangents=tx,
    ...     channels_psf_3d_ytangents=ty,
    ...     channels_psf_3d_ztangents=tz,
    ...     channels_psf_3d_spline_coeffs=coeffs,
    ... )
    >>> info['amp'].ndim
    1
    """
    n_channels = len(crops)
    X0_input, Y0_input = X0, Y0

    channels_pixels_nm = _normalize_channels_pixels_nm(channels_pixels_nm, n_channels)
    channels_gains = _normalize_channels_parameter(channels_gains, n_channels)
    channels_QE = _normalize_channels_parameter(channels_QE, n_channels)

    optimizer = _resolve_optimizer(optimizer)
    distribution = _resolve_distribution(distribution)
    estimator = _resolve_estimator(estimator, distribution)

    if channels_psf_3d_xtangents is None:
        raise SyntaxError("channels_psf_3d_xtangents must be specified as a kwarg")
    if len(channels_psf_3d_xtangents) != n_channels:
        raise ValueError("channels_psf_3d_xtangents does not have the same length as crops")

    if channels_psf_3d_ytangents is None:
        raise SyntaxError("channels_psf_3d_ytangents must be specified as a kwarg")
    if len(channels_psf_3d_ytangents) != n_channels:
        raise ValueError("channels_psf_3d_ytangents does not have the same length as crops")

    if channels_psf_3d_ztangents is None:
        raise SyntaxError("channels_psf_3d_ztangents must be specified as a kwarg")
    if len(channels_psf_3d_ztangents) != n_channels:
        raise ValueError("channels_psf_3d_ztangents does not have the same length as crops")

    if channels_psf_3d_spline_coeffs is None:
        raise SyntaxError("channels_psf_3d_spline_coeffs must be specified as a kwarg")
    if len(channels_psf_3d_spline_coeffs) != n_channels:
        raise ValueError(
            "channels_psf_3d_spline_coeffs does not have the same length as crops"
        )

    X0, Y0, positions = split_channel_origins(crops, X0_input, Y0_input, ch, cuda=cuda)
    bary_x, bary_y, _ = locs_individual_barycenter(
        crops,
        X0_input,
        Y0_input,
        ch=ch,
        channels_pixels_nm=channels_pixels_nm,
        cuda=cuda,
        parallel=parallel,
    )
    bary_x, bary_y, _ = split_channel_origins(crops, bary_x, bary_y, ch, cuda=cuda)

    fit_kwargs = [
        dict(
            tx=tx,
            ty=ty,
            tz=tz,
            coeffs=coeffs,
        )
        for tx, ty, tz, coeffs in zip(
            channels_psf_3d_xtangents,
            channels_psf_3d_ytangents,
            channels_psf_3d_ztangents,
            channels_psf_3d_spline_coeffs,
        )
    ]

    xp = get_xp(cuda)
    mux_all = []
    muy_all = []
    muz_all = []
    amp_all = []
    offset_all = []
    converged_all = []

    for crop, x0, y0, bary_x_ch, bary_y_ch, pixel, gain, qe, function_kw in zip(
        crops,
        X0,
        Y0,
        bary_x,
        bary_y,
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

        if len(crop) == 0:
            empty = xp.empty(0, dtype=xp.float32)
            mux_all.append(empty)
            muy_all.append(empty)
            muz_all.append(empty)
            amp_all.append(empty)
            offset_all.append(empty)
            converged_all.append(xp.empty(0, dtype=xp.int8))
            continue

        center_x = (width - 1) / 2 * pixel[1]
        center_y = (height - 1) / 2 * pixel[0]
        mux_min = center_x - 1.5 * pixel[1]
        mux_max = center_x + 1.5 * pixel[1]
        muy_min = center_y - 1.5 * pixel[0]
        muy_max = center_y + 1.5 * pixel[0]
        bary_x_ch = xp.asarray(bary_x_ch)
        bary_y_ch = xp.asarray(bary_y_ch)
        mux = bary_x_ch - x0
        muy = bary_y_ch - y0
        mux = xp.where(xp.isfinite(mux), mux, center_x)
        muy = xp.where(xp.isfinite(muy), muy, center_y)
        mux = xp.clip(mux, mux_min, mux_max)
        muy = xp.clip(muy, muy_min, muy_max)
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
        function.mux_min = mux_min
        function.mux_max = mux_max
        function.muy_min = muy_min
        function.muy_max = muy_max

        fit = optimizer(function, estimator)
        if cuda:
            fit(crop, xx, yy, zz)
        else:
            with nb_threads(parallel):
                fit(crop, xx, yy, zz)
        converged = getattr(fit, "converged", xp.zeros(len(crop), dtype=xp.int8))

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
            converged = xp.asnumpy(converged)

        mux_all.append(mux)
        muy_all.append(muy)
        muz_all.append(muz)
        amp_all.append(amp)
        offset_all.append(offset)
        converged_all.append(converged)

    info = {
        "amp": stack_channel_values(amp_all, positions),
        "offset": stack_channel_values(offset_all, positions),
        "converged": stack_channel_values(converged_all, positions),
    }

    return stack_channel_values(mux_all, positions), stack_channel_values(muy_all, positions), stack_channel_values(muz_all, positions), info



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
