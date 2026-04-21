#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from funclp import LM, MLE, LSE, Poisson, Normal, Gaussian2D
from arrlp import get_xp, nb_threads, coordinates
import numpy as np

SIGMA = 0.21 * 670 / 1.5



# %% Function
@block()
def locs_individual_gaussfit(
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
    channels_psf_xsigmas_nm=SIGMA,
    channels_psf_ysigmas_nm=SIGMA,
    channels_psf_theta_deg=0.0,
    fit_theta=False,
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
    channels_psf_xsigmas_nm : float or sequence, optional
        Initial/fixed PSF sigma along x for each channel.
    channels_psf_ysigmas_nm : float or sequence, optional
        Initial/fixed PSF sigma along y for each channel.
    channels_psf_theta_deg : float or sequence, optional
        Initial/fixed PSF angle in degrees for each channel.
    fit_theta : bool, optional
        Whether to fit the PSF rotation angle.

    Returns
    -------
    tuple
        A tuple ``(mux, muy, output)`` where:

        - ``mux`` is the concatenated x localization array in nanometers,
        - ``muy`` is the concatenated y localization array in nanometers,
        - ``output`` is a dictionary with fitted parameter arrays.

        ``output`` contains:

        ``'amp'``
            Concatenated converted amplitudes.
        ``'offset'``
            Concatenated converted offsets.
        ``'sigmax'``
            Concatenated fitted x sigmas.
        ``'sigmay'``
            Concatenated fitted y sigmas.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    >>> x0 = [np.array([10, 20], dtype=np.float32)]
    >>> y0 = [np.array([30, 40], dtype=np.float32)]
    >>> mux, muy, output = locs_individual_gaussfit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 100.0)],
    ...     channels_psf_xsigmas_nm=[90.0],
    ...     channels_psf_ysigmas_nm=[90.0],
    ... )
    >>> mux.shape == muy.shape
    True
    >>> sorted(output)
    ['amp', 'offset', 'sigmax', 'sigmay']

    >>> mux, muy, output = locs_individual_gaussfit(
    ...     crops,
    ...     x0,
    ...     y0,
    ...     channels_pixels_nm=[(100.0, 120.0)],
    ...     channels_psf_xsigmas_nm=[80.0],
    ...     channels_psf_ysigmas_nm=[95.0],
    ...     channels_psf_theta_deg=[5.0],
    ...     fit_theta=True,
    ... )
    >>> output['sigmax'].ndim
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

    channels_psf_xsigmas_nm = _normalize_psf_parameter(
        channels_psf_xsigmas_nm,
        n_channels,
    )
    channels_psf_ysigmas_nm = _normalize_psf_parameter(
        channels_psf_ysigmas_nm,
        n_channels,
    )
    channels_psf_theta_deg = _normalize_psf_parameter(
        channels_psf_theta_deg,
        n_channels,
    )

    fit_kwargs = [
        dict(
            sigx=sigx,
            sigy=sigy,
            theta=theta,
            pixx=pixel[1],
            pixy=pixel[0],
            theta_fit=fit_theta,
        )
        for pixel, sigx, sigy, theta in zip(
            channels_pixels_nm,
            channels_psf_xsigmas_nm,
            channels_psf_ysigmas_nm,
            channels_psf_theta_deg,
        )
    ]

    xp = get_xp(cuda)
    mux_all = []
    muy_all = []
    amp_all = []
    offset_all = []
    sigmax_all = []
    sigmay_all = []

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
        x0 = xp.asarray(x0) * pixel[1]
        y0 = xp.asarray(y0) * pixel[0]

        mux = xp.full_like(x0, fill_value=(width - 1) / 2 * pixel[1])
        muy = xp.full_like(y0, fill_value=(height - 1) / 2 * pixel[0])
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

        fit = optimizer(function, estimator)
        if cuda:
            fit(crop, xx, yy)
        else:
            with nb_threads(parallel):
                fit(crop, xx, yy)

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

        mux_all.append(mux)
        muy_all.append(muy)
        amp_all.append(amp)
        offset_all.append(offset)

        sigx = function.sigx
        sigy = function.sigy
        if cuda:
            sigx = xp.asnumpy(sigx)
            sigy = xp.asnumpy(sigy)
        sigmax_all.append(sigx)
        sigmay_all.append(sigy)

    output = {
        "amp": np.hstack(amp_all),
        "offset": np.hstack(offset_all),
        "sigmax": np.hstack(sigmax_all),
        "sigmay": np.hstack(sigmay_all),
    }

    return np.hstack(mux_all), np.hstack(muy_all), output



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

