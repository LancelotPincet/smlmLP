#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block, locs_individual_barycenter
from funclp import LM, MLE, LSE, Poisson, Normal, Gaussian2D
from arrlp import get_xp, nb_threads, coordinates
from ._channel_values import split_channel_origins, stack_channel_values

SIGMA = 0.21 * 670 / 1.5



@block()
def locs_individual_gaussfit(
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
    channels_psf_xsigmas_nm=SIGMA,
    channels_psf_ysigmas_nm=SIGMA,
    channels_psf_thetas_deg=0.0,
    channels_fit_thetas=False,
):
    """
    Fit each crop independently with an anisotropic 2D Gaussian model.

    The function loops through channels, initializes a :class:`funclp.Gaussian2D`
    model per event, runs the selected optimizer/estimator combination, and
    returns localized coordinates with fitted parameters.

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
    channels_psf_xsigmas_nm : float or sequence, optional
        Initial/fixed PSF sigma along x for each channel.
    channels_psf_ysigmas_nm : float or sequence, optional
        Initial/fixed PSF sigma along y for each channel.
    channels_psf_thetas_deg : float or sequence, optional
        Initial/fixed PSF angle in degrees for each channel.
    channels_fit_thetas : bool, optional
        Whether to fit the PSF rotation angle.

    Returns
    -------
    tuple
        A tuple ``(mux, muy, info)`` where:

        - ``mux`` is the detection-aligned x localization array in nanometers,
        - ``muy`` is the detection-aligned y localization array in nanometers,
        - ``info`` is a dictionary with fitted parameter arrays.

        ``info`` contains:

        ``'amp'``
            Detection-aligned converted amplitudes.
        ``'offset'``
            Detection-aligned converted offsets.
        ``'sigmax'``
            Detection-aligned fitted x sigmas.
        ``'sigmay'``
            Detection-aligned fitted y sigmas.

    Notes
    -----
    1. ``X0`` and ``Y0`` are split by ``ch`` so each origin vector follows the
       crop order inside its channel stack.
    2. A local coordinate grid is built from the channel pixel size, and each
       anisotropic Gaussian is initialized at the crop center.
    3. The optimizer updates local Gaussian parameters, local coordinates are
       shifted by crop origins, and all outputs are remapped to detection order.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = np.array([10, 20], dtype=np.float32)
    >>> y0 = np.array([30, 40], dtype=np.float32)
    >>> mux, muy, info = locs_individual_gaussfit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ...     channels_psf_xsigmas_nm=[90.0],
    ...     channels_psf_ysigmas_nm=[90.0],
    ... )
    >>> mux.shape == muy.shape
    True
    >>> sorted(info)
    ['amp', 'offset', 'sigmax', 'sigmay']

    >>> mux, muy, info = locs_individual_gaussfit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 120.0)],
    ...     channels_psf_xsigmas_nm=[80.0],
    ...     channels_psf_ysigmas_nm=[95.0],
    ...     channels_psf_thetas_deg=[5.0],
    ...     channels_fit_thetas=True,
    ... )
    >>> info['sigmax'].ndim
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

    channels_psf_xsigmas_nm = _normalize_psf_parameter(
        channels_psf_xsigmas_nm,
        n_channels,
    )
    channels_psf_ysigmas_nm = _normalize_psf_parameter(
        channels_psf_ysigmas_nm,
        n_channels,
    )
    channels_psf_thetas_deg = _normalize_psf_parameter(
        channels_psf_thetas_deg,
        n_channels,
    )
    channels_fit_thetas = _normalize_psf_parameter(
        channels_fit_thetas,
        n_channels,
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
            sigx=sigx,
            sigy=sigy,
            theta=theta,
            pixx=pixel[1],
            pixy=pixel[0],
            theta_fit=fit_theta,
        )
        for pixel, sigx, sigy, theta, fit_theta in zip(
            channels_pixels_nm,
            channels_psf_xsigmas_nm,
            channels_psf_ysigmas_nm,
            channels_psf_thetas_deg,
            channels_fit_thetas,
        )
    ]

    xp = get_xp(cuda)
    mux_all = []
    muy_all = []
    amp_all = []
    offset_all = []
    sigmax_all = []
    sigmay_all = []
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
        x0 = xp.asarray(x0) * pixel[1]
        y0 = xp.asarray(y0) * pixel[0]

        if len(crop) == 0:
            empty = xp.empty(0, dtype=xp.float32)
            mux_all.append(empty)
            muy_all.append(empty)
            amp_all.append(empty)
            offset_all.append(empty)
            sigmax_all.append(empty)
            sigmay_all.append(empty)
            converged_all.append(xp.empty(0, dtype=xp.int8))
            continue

        center_x = (width - 1) / 2 * pixel[1]
        center_y = (height - 1) / 2 * pixel[0]
        bary_x_ch = xp.asarray(bary_x_ch)
        bary_y_ch = xp.asarray(bary_y_ch)
        mux_center = bary_x_ch - x0
        muy_center = bary_y_ch - y0
        mux_center = xp.where(xp.isfinite(mux_center), mux_center, center_x)
        muy_center = xp.where(xp.isfinite(muy_center), muy_center, center_y)
        mux_min = mux_center - 1.5 * pixel[1]
        mux_max = mux_center + 1.5 * pixel[1]
        muy_min = muy_center - 1.5 * pixel[0]
        muy_max = muy_center + 1.5 * pixel[0]
        mux = mux_center
        muy = muy_center
        mux = xp.clip(mux, mux_min, mux_max)
        muy = xp.clip(muy, muy_min, muy_max)
        amp = xp.max(crop, axis=(1, 2))
        offset = xp.min(crop, axis=(1, 2))

        function = Gaussian2D(
            mux=mux,
            muy=muy,
            amp=amp,
            offset=offset,
            cuda=cuda,
            **function_kw,
        )
        function.sigx_min = SIGMA / 3
        function.sigx_max = SIGMA * 3
        function.sigy_min = SIGMA / 3
        function.sigy_max = SIGMA * 3
        function.mux_min = mux_min
        function.mux_max = mux_max
        function.muy_min = muy_min
        function.muy_max = muy_max

        fit = optimizer(function, estimator)
        if cuda:
            fit(crop, xx, yy)
        else:
            with nb_threads(parallel):
                fit(crop, xx, yy)
        converged = getattr(fit, "converged", xp.zeros(len(crop), dtype=xp.int8))

        mux, muy = function.mux, function.muy
        mux += x0
        muy += y0
        amp = function.amp / qe * gain
        offset = function.offset / qe * gain

        if cuda:
            mux = xp.asnumpy(mux)
            muy = xp.asnumpy(muy)
            amp = xp.asnumpy(amp)
            offset = xp.asnumpy(offset)
            converged = xp.asnumpy(converged)

        mux_all.append(mux)
        muy_all.append(muy)
        amp_all.append(amp)
        offset_all.append(offset)
        converged_all.append(converged)

        sigx = function.sigx
        sigy = function.sigy
        if cuda:
            sigx = xp.asnumpy(sigx)
            sigy = xp.asnumpy(sigy)
        sigmax_all.append(sigx)
        sigmay_all.append(sigy)

    info = {
        "amp": stack_channel_values(amp_all, positions),
        "offset": stack_channel_values(offset_all, positions),
        "sigmax": stack_channel_values(sigmax_all, positions),
        "sigmay": stack_channel_values(sigmay_all, positions),
        "converged": stack_channel_values(converged_all, positions),
    }

    return stack_channel_values(mux_all, positions), stack_channel_values(muy_all, positions), info



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



def _normalize_psf_parameter(values, n_channels):
    """Normalize PSF parameters to one value per channel."""
    try:
        if len(values) != n_channels:
            raise ValueError(
                "channel_mean_radius_pix does not have the same length as channels"
            )
    except TypeError:
        values = [values for _ in range(n_channels)]

    return values
