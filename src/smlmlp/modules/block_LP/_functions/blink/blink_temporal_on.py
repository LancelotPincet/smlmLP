#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, get_xp, img_gaussianfilter
from funclp import Exponential1
from stacklp import temporal_autocorr as stack_autocorr
import numpy as np
from scipy.optimize import curve_fit



# %% Function
@block()
def blink_temporal_on(
    channels,
    crop_fr=None,
    /,
    psf_sigma_nm=100.0,
    *,
    exposure_ms=50.0,
    channels_pixels_nm=100.0,
    cuda=False,
    parallel=False,
):
    """
    Estimate the blinking on-time from temporal autocorrelations.

    This function computes a temporal autocorrelation curve for each channel
    after subtracting a smooth background estimated from the PSF scale. The
    channel autocorrelations are then averaged and fitted with a 1D
    exponential model in order to estimate the characteristic on-time.

    The returned ``info`` dictionary contains the main intermediate results,
    including the per-channel autocorrelations, their average, the fitted
    curve, and the associated time axis.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel. Each channel is expected to
        have shape ``(n_frames, height, width)``.
    crop_fr : int or None, optional
        Number of temporal lags to keep from the autocorrelation. If ``None``,
        it is set to ``int(len(channels) // 2 - 1)`` following the original
        implementation.
    psf_sigma_nm : float or sequence, optional
        PSF sigma in nanometers, one value per channel or a scalar shared
        across channels. This value is used to define the Gaussian smoothing
        scale for the background estimation.
    exposure_ms : float, optional
        Exposure time in milliseconds.
    channels_pixels_nm : float or sequence, optional
        Pixel size in nanometers for each channel. This value is normalized
        through :class:`smlmlp.Config` so that one pixel size pair is available
        for each channel.
    cuda : bool, optional
        Whether to use CUDA-enabled array operations when supported.
    parallel : bool, optional
        Whether to enable parallel execution in the backend functions.

    Returns
    -------
    tuple
        A tuple ``(on_time, info)`` where:

        - ``on_time`` is the fitted characteristic on-time in milliseconds,
        - ``info`` is a dictionary containing the main intermediate results.

        The dictionary contains the following keys:

        ``'ac'``
            Temporal autocorrelation curve for each channel.
        ``'average'``
            Average temporal autocorrelation across channels.
        ``'fit'``
            Fitted exponential decay evaluated on the lag grid.
        ``'time'``
            Time axis in milliseconds corresponding to the fitted curve.

    Notes
    -----
    For each channel, the background is estimated with a Gaussian filter using
    a scale of ``psf_sigma_nm[i] * 3``. After subtraction, the temporal
    autocorrelation is computed, cropped, spatially averaged, shifted by its
    last value, and normalized by its first value.

    The averaged autocorrelation is fitted with :class:`funclp.Exponential1`,
    and the returned on-time is:

    .. math::

        \\tau_\\mathrm{on} = \\tau \\times \\mathrm{exposure\\_ms}

    where :math:`\\tau` is the fitted exponential parameter in frame units.

    Examples
    --------
    Estimate the on-time for one channel:

    >>> import numpy as np
    >>> channel = np.random.rand(50, 32, 32).astype(np.float32)
    >>> on_time, info = blink_temporal_on([channel], psf_sigma_nm=[120.0])
    >>> isinstance(on_time, float)
    True
    >>> "average" in info
    True

    Estimate the on-time for two channels with explicit exposure time:

    >>> channels = [
    ...     np.random.rand(50, 32, 32).astype(np.float32),
    ...     np.random.rand(50, 32, 32).astype(np.float32),
    ... ]
    >>> on_time, info = blink_temporal_on(
    ...     channels,
    ...     crop_fr=10,
    ...     psf_sigma_nm=[120.0, 140.0],
    ...     exposure_ms=25.0,
    ...     channels_pixels_nm=[(100.0, 100.0), (110.0, 110.0)],
    ... )
    >>> len(info["ac"])
    2
    >>> info["time"].shape[0]
    10
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Normalize the pixel size input so that one pixel size pair is available
    # for each channel.
    try:
        if len(channels_pixels_nm) != len(channels):
            if len(channels_pixels_nm) == 2:
                channels_pixels_nm = [
                    channels_pixels_nm for _ in range(len(channels))
                ]
            else:
                raise ValueError(
                    "channels_pixels_nm does not have the same length as channels"
                )
    except TypeError:
        channels_pixels_nm = [
            (channels_pixels_nm, channels_pixels_nm)
            for _ in range(len(channels))
        ]

    # Build the temporal lag coordinates used for averaging and fitting.
    if crop_fr is None:
        crop_fr = int(len(channels) // 2 - 1)

    T = np.arange(crop_fr)
    t = T * exposure_ms

    info = {"ac": []}

    for i, (channel, pix) in enumerate(zip(channels, channels_pixels_nm)):
        # Estimate a smooth background from the PSF scale and subtract it
        # before computing the temporal autocorrelation.
        gc()
        bkgd = img_gaussianfilter(
            channel,
            sigma=psf_sigma_nm[i] * 3,
            pixel=pix,
            stacks=True,
            cuda=cuda,
            parallel=parallel,
        )
        bkgd = xp.minimum(bkgd, channel)
        channel = channel - bkgd

        # Compute the temporal autocorrelation, keep the requested positive
        # lags, and average spatially.
        f0 = int(channel.shape[0] // 2)
        ac = stack_autocorr(channel, cuda=cuda, parallel=parallel)[
            f0 + 1 : f0 + 1 + crop_fr
        ]
        ac = ac.mean(axis=(1, 2))

        # Shift and normalize the autocorrelation according to the original
        # implementation.
        ac -= ac[-1]
        ac /= ac[0]

        if cuda:
            ac = xp.asnumpy(ac)

        info["ac"].append(ac)

    # Average the autocorrelation curves across channels.
    gc()
    y = np.zeros_like(info["ac"][0])
    for ac in info["ac"]:
        y += ac / len(info["ac"])

    info["average"] = y

    # Fit the averaged autocorrelation with a 1D exponential model.
    p0 = [1.0, -0.75]  # tau, offset
    bounds = ([0.0, -1.0], [len(y), 1.0])

    expodecay = Exponential1()
    func2fit = lambda t, tau, offset: expodecay(t, tau=tau, offset=offset)

    popt, _ = curve_fit(func2fit, T, y, p0=p0, bounds=bounds)
    tau, offset = popt

    info["fit"] = func2fit(T, *popt)
    info["time"] = t

    # Convert the fitted decay constant from frames to milliseconds.
    on_time = tau * exposure_ms

    return on_time, info