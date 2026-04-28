#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

import numpy as np
from arrlp import coordinates, get_xp
from funclp import Gaussian2D, IsoGaussian, Spline3D

from smlmlp import block
from smlmlp import crop_remove_bkgd

SIGMA = 0.21 * 670 / 1.5
FIT_MODELS = ("isogauss", "gauss", "spline")


@block()
def integrate_crop(
    crops, X0=None, Y0=None, /,
    ch=None, *,
    x_fit=None, y_fit=None, z_fit=None, os_fit=None,
    sigma_fit=None, sigma_x_fit=None, sigma_y_fit=None, sigma_angle=None,
    channels_weighted_integrations=True, channels_fit_models="isogauss",
    channels_pixels_nm=1.0, channels_gains=1.0, channels_QE=1.0,
    channels_psf_sigmas_nm=SIGMA, channels_psf_xsigmas_nm=SIGMA, channels_psf_ysigmas_nm=SIGMA,
    channels_psf_thetas_deg=0.0,
    channels_psf_3d_xtangents=None, channels_psf_3d_ytangents=None, channels_psf_3d_ztangents=None,
    channels_psf_3d_spline_coeffs=None,
    cuda=False, parallel=False,
):
    """
    Integrate photon counts from per-channel crop stacks.

    Parameters
    ----------
    crops : sequence of array-like
        Crop stacks, one stack per channel, shaped ``(N, Y, X)``. The order
        inside each channel must match the detections where ``ch == channel``.
    X0, Y0 : array-like or sequence of array-like, optional
        Crop x/y origins in pixels. Detection-aligned arrays are split with
        ``ch``. Per-channel arrays are also accepted. These origins are required
        for weighted channels because fitted coordinates are stored globally.
    ch : array-like or None, optional
        One-based channel index for each detection. When provided, integrated
        values are remapped to this detection order.
    x_fit, y_fit : array-like, optional
        Fitted x/y coordinates in nanometers. Required for weighted channels.
    z_fit : array-like, optional
        Fitted z coordinates for spline channels. Missing values default to zero.
    os_fit : array-like, optional
        Fitted offset in photons per pixel. Required for weighted channels and
        converted back to raw crop units before subtraction.
    sigma_fit : array-like, optional
        Fitted isotropic sigma in nanometers for ``"isogauss"`` channels.
    sigma_x_fit, sigma_y_fit : array-like, optional
        Fitted x/y sigmas in nanometers for ``"gauss"`` channels.
    sigma_angle : array-like, optional
        Per-detection Gaussian angle. If omitted, ``channels_psf_thetas_deg`` is
        used for each channel.
    channels_weighted_integrations : bool or sequence of bool, optional
        Per-channel switch selecting weighted PSF projection when true and
        border-background summation when false.
    channels_fit_models : str or sequence of str, optional
        Per-channel fit model used to reconstruct weighted PSFs. Supported
        values are ``"isogauss"``, ``"gauss"``, and ``"spline"``.
    channels_pixels_nm : float, tuple, or sequence, optional
        Pixel sizes in ``(y, x)`` nanometers. A scalar or one tuple is broadcast
        to all channels.
    channels_gains : float or sequence, optional
        Gain values used to convert raw crop counts to photons.
    channels_QE : float or sequence, optional
        Quantum efficiencies used to convert raw crop counts to photons.
    channels_psf_sigmas_nm, channels_psf_xsigmas_nm, channels_psf_ysigmas_nm : float or sequence, optional
        Channel PSF defaults used when fitted sigma columns are not supplied.
    channels_psf_thetas_deg : float or sequence, optional
        Channel Gaussian angle defaults.
    channels_psf_3d_xtangents, channels_psf_3d_ytangents, channels_psf_3d_ztangents : sequence, optional
        Spline tangents used by weighted ``"spline"`` channels.
    channels_psf_3d_spline_coeffs : sequence, optional
        Spline coefficients used by weighted ``"spline"`` channels.
    cuda : bool, optional
        Whether to run array operations on CUDA-compatible arrays.
    parallel : bool, optional
        Whether to parallelize the background-removal step for unweighted
        channels.

    Returns
    -------
    tuple
        A tuple ``(intensity, info)`` where ``intensity`` is a one-dimensional
        detection-aligned photon count array. ``info`` contains normalized
        channel settings and intermediate weighted denominators.

    Raises
    ------
    ValueError
        If channel lengths are inconsistent, channel indices are invalid, or a
        weighted channel is missing required fitted coordinates or offsets.
    SyntaxError
        If spline metadata is required but missing.

    Notes
    -----
    Unweighted channels subtract the border-median background with
    :func:`crop_remove_bkgd`, sum all corrected pixels, and convert raw counts as
    ``raw * gain / QE``.

    Weighted channels reconstruct the fitted PSF shape with the matching
    :mod:`funclp` function and zero offset. The fitted offset is subtracted from
    the raw crop after conversion back to raw units. The normalized PSF
    ``p = psf_model / psf_model.sum()`` is then used as a matched filter:
    ``N = sum(p * crop_no_bkgd) / sum(p**2)``. The resulting raw count is
    converted with the same ``gain / QE`` factor.

    Examples
    --------
    >>> import numpy as np
    >>> crops = [np.ones((2, 3, 3), dtype=np.float32)]
    >>> crops[0][:, 1, 1] = [11.0, 21.0]
    >>> intensity, info = integrate_crop(crops, channels_weighted_integrations=False)
    >>> intensity.shape
    (2,)

    >>> ch = np.array([2, 1], dtype=np.uint8)
    >>> crops = [np.ones((1, 3, 3), dtype=np.float32), np.ones((1, 3, 3), dtype=np.float32)]
    >>> crops[1][0, 1, 1] = 6.0
    >>> intensity, info = integrate_crop(
    ...     crops,
    ...     np.array([0, 0]),
    ...     np.array([0, 0]),
    ...     ch=ch,
    ...     x_fit=np.array([np.nan, 1.0], dtype=np.float32),
    ...     y_fit=np.array([np.nan, 1.0], dtype=np.float32),
    ...     os_fit=np.array([0.0, 1.0], dtype=np.float32),
    ...     channels_weighted_integrations=[True, False],
    ... )
    >>> intensity.ndim
    1
    """
    # Normalize and validate inputs
    n_channels = len(crops)
    cuda = bool(cuda)
    xp = get_xp(cuda)
    positions = _channel_positions(crops, ch)

    channels_weighted_integrations = [
        bool(value) for value in _normalize_channels_parameter(
            channels_weighted_integrations, n_channels, "channels_weighted_integrations")
    ]
    channels_fit_models = _normalize_channels_fit_models(channels_fit_models, n_channels)
    channels_pixels_nm = _normalize_channels_pixels_nm(channels_pixels_nm, n_channels)
    channels_gains = _normalize_channels_parameter(channels_gains, n_channels, "channels_gains")
    channels_QE = _normalize_channels_parameter(channels_QE, n_channels, "channels_QE")
    channels_psf_sigmas_nm = _normalize_channels_parameter(channels_psf_sigmas_nm, n_channels, "channels_psf_sigmas_nm")
    channels_psf_xsigmas_nm = _normalize_channels_parameter(channels_psf_xsigmas_nm, n_channels, "channels_psf_xsigmas_nm")
    channels_psf_ysigmas_nm = _normalize_channels_parameter(channels_psf_ysigmas_nm, n_channels, "channels_psf_ysigmas_nm")
    channels_psf_thetas_deg = _normalize_channels_parameter(channels_psf_thetas_deg, n_channels, "channels_psf_thetas_deg")

    needs_spline = any(weighted and model == "spline"
                       for weighted, model in zip(channels_weighted_integrations, channels_fit_models))
    channels_psf_3d_xtangents = _normalize_spline_parameter(
        channels_psf_3d_xtangents, n_channels, "channels_psf_3d_xtangents", required=needs_spline)
    channels_psf_3d_ytangents = _normalize_spline_parameter(
        channels_psf_3d_ytangents, n_channels, "channels_psf_3d_ytangents", required=needs_spline)
    channels_psf_3d_ztangents = _normalize_spline_parameter(
        channels_psf_3d_ztangents, n_channels, "channels_psf_3d_ztangents", required=needs_spline)
    channels_psf_3d_spline_coeffs = _normalize_spline_parameter(
        channels_psf_3d_spline_coeffs, n_channels, "channels_psf_3d_spline_coeffs", required=needs_spline)

    any_weighted = any(channels_weighted_integrations)
    any_unweighted = any(not w for w in channels_weighted_integrations)

    # Split fitted parameters per channel
    if any_weighted:
        X0_channels, Y0_channels = _split_origins(crops, X0, Y0, positions, cuda=cuda)
        x_fit_channels = _split_detection_values(x_fit, crops, positions, cuda=cuda, name="x_fit", required=True)
        y_fit_channels = _split_detection_values(y_fit, crops, positions, cuda=cuda, name="y_fit", required=True)
        z_fit_channels = _split_detection_values(z_fit, crops, positions, cuda=cuda)
        os_fit_channels = _split_detection_values(os_fit, crops, positions, cuda=cuda, name="os_fit", required=True)
        sigma_fit_channels = _split_detection_values(sigma_fit, crops, positions, cuda=cuda)
        sigma_x_fit_channels = _split_detection_values(sigma_x_fit, crops, positions, cuda=cuda)
        sigma_y_fit_channels = _split_detection_values(sigma_y_fit, crops, positions, cuda=cuda)
        sigma_angle_channels = _split_detection_values(sigma_angle, crops, positions, cuda=cuda)
    else:
        X0_channels = Y0_channels = [None for _ in range(n_channels)]
        x_fit_channels = y_fit_channels = z_fit_channels = [None for _ in range(n_channels)]
        os_fit_channels = sigma_fit_channels = [None for _ in range(n_channels)]
        sigma_x_fit_channels = sigma_y_fit_channels = [None for _ in range(n_channels)]
        sigma_angle_channels = [None for _ in range(n_channels)]

    # Remove background for unweighted channels
    bkgd_crops, bkgd_info = (None, {})
    if any_unweighted:
        bkgd_crops, bkgd_info = crop_remove_bkgd(crops, cuda=cuda, parallel=parallel)

    # Integrate each channel
    intensities, psf_sums, weighted_denominators = [], [], []
    for channel_index, values in enumerate(zip(
        crops, channels_weighted_integrations, channels_fit_models, channels_pixels_nm,
        channels_gains, channels_QE, channels_psf_sigmas_nm, channels_psf_xsigmas_nm,
        channels_psf_ysigmas_nm, channels_psf_thetas_deg, channels_psf_3d_xtangents,
        channels_psf_3d_ytangents, channels_psf_3d_ztangents, channels_psf_3d_spline_coeffs,
        X0_channels, Y0_channels, x_fit_channels, y_fit_channels, z_fit_channels,
        os_fit_channels, sigma_fit_channels, sigma_x_fit_channels, sigma_y_fit_channels, sigma_angle_channels,
    )):
        (crop, weighted, model, pixel, gain, qe, psf_sigma, psf_xsigma, psf_ysigma,
         psf_theta, tx, ty, tz, coeffs, x0, y0,
         x_fit_ch, y_fit_ch, z_fit_ch, os_fit_ch,
         sigma_fit_ch, sigma_x_fit_ch, sigma_y_fit_ch, sigma_angle_ch) = values

        crop = xp.asarray(crop)
        if len(crop) == 0:
            intensity = xp.empty(0, dtype=xp.float32)
            psf_sums.append(np.empty(0, dtype=np.float32))
            weighted_denominators.append(np.empty(0, dtype=np.float32))
        elif weighted:
            intensity, psf_sum, denominator = _integrate_weighted_channel(
                crop, x0, y0, x_fit_ch, y_fit_ch, z_fit_ch, os_fit_ch,
                sigma_fit_ch, sigma_x_fit_ch, sigma_y_fit_ch, sigma_angle_ch,
                model=model, pixel=pixel, gain=gain, qe=qe,
                psf_sigma=psf_sigma, psf_xsigma=psf_xsigma, psf_ysigma=psf_ysigma, psf_theta=psf_theta,
                tx=tx, ty=ty, tz=tz, coeffs=coeffs, cuda=cuda,
            )
            psf_sums.append(_asnumpy(psf_sum))
            weighted_denominators.append(_asnumpy(denominator))
        else:
            crop_no_bkgd = xp.asarray(bkgd_crops[channel_index])
            intensity = xp.sum(crop_no_bkgd, axis=(1, 2)) * gain / qe
            psf_sums.append(np.full(len(crop), np.nan, dtype=np.float32))
            weighted_denominators.append(np.full(len(crop), np.nan, dtype=np.float32))

        intensities.append(_asnumpy(intensity).astype(np.float32, copy=False))

    info = {
        "channels_weighted_integrations": channels_weighted_integrations,
        "channels_fit_models": channels_fit_models,
        "channels_pixels_nm": channels_pixels_nm,
        "psf_sums": psf_sums,
        "weighted_denominators": weighted_denominators,
    }
    info.update(bkgd_info)
    return stack_channel_values(intensities, positions), info


def _integrate_weighted_channel(
    crop, x0, y0, x_fit, y_fit, z_fit, os_fit, sigma_fit, sigma_x_fit, sigma_y_fit, sigma_angle, /,
    *, model, pixel, gain, qe, psf_sigma, psf_xsigma, psf_ysigma, psf_theta, tx, ty, tz, coeffs, cuda,
):
    """Integrate one channel by projecting crops on fitted PSF shapes."""
    xp = get_xp(cuda)
    _, height, width = crop.shape

    # Cast to float32 arrays
    x0, y0 = xp.asarray(x0, dtype=xp.float32), xp.asarray(y0, dtype=xp.float32)
    x_fit, y_fit = xp.asarray(x_fit, dtype=xp.float32), xp.asarray(y_fit, dtype=xp.float32)
    os_fit = xp.asarray(os_fit, dtype=xp.float32)

    # Build PSF grid relative to crop origins
    yy, xx = coordinates(shape=(height, width), pixel=pixel, cuda=cuda)
    mux = x_fit - x0 * pixel[1]
    muy = y_fit - y0 * pixel[0]
    amp, offset = xp.ones_like(mux, dtype=xp.float32), xp.zeros_like(mux, dtype=xp.float32)

    # Reconstruct PSF model
    function = _make_model(
        model=model, mux=mux, muy=muy, amp=amp, offset=offset,
        z_fit=z_fit, sigma_fit=sigma_fit, sigma_x_fit=sigma_x_fit, sigma_y_fit=sigma_y_fit,
        sigma_angle=sigma_angle, pixel=pixel,
        psf_sigma=psf_sigma, psf_xsigma=psf_xsigma, psf_ysigma=psf_ysigma, psf_theta=psf_theta,
        tx=tx, ty=ty, tz=tz, coeffs=coeffs, cuda=cuda,
    )
    psf_model = function(xx, yy, xp.zeros_like(xx)) if model == "spline" else function(xx, yy)

    # Matched-filter integration with fitted PSF
    crop_no_bkgd = crop - (os_fit * qe / gain)[:, None, None]
    psf_sum = xp.sum(psf_model, axis=(1, 2))
    psf = psf_model / psf_sum[:, None, None]
    denominator = xp.sum(psf * psf, axis=(1, 2))
    raw_intensity = xp.sum(psf * crop_no_bkgd, axis=(1, 2)) / denominator
    valid = xp.isfinite(psf_sum) & xp.isfinite(denominator) & (psf_sum != 0) & (denominator != 0)
    raw_intensity = xp.where(valid, raw_intensity, xp.nan)
    return raw_intensity * gain / qe, psf_sum, denominator


def _make_model(
    *, model, mux, muy, amp, offset,
    z_fit, sigma_fit, sigma_x_fit, sigma_y_fit, sigma_angle,
    pixel, psf_sigma, psf_xsigma, psf_ysigma, psf_theta, tx, ty, tz, coeffs, cuda,
):
    """Instantiate the fitted PSF model for one weighted channel."""
    xp = get_xp(cuda)

    if model == "isogauss":
        sigma = _parameter_or_default(sigma_fit, mux, psf_sigma, cuda=cuda)
        return IsoGaussian(mux=mux, muy=muy, amp=amp, offset=offset, sig=sigma, pixx=pixel[1], pixy=pixel[0], cuda=cuda)

    if model == "gauss":
        sigx = _parameter_or_default(sigma_x_fit, mux, psf_xsigma, cuda=cuda)
        sigy = _parameter_or_default(sigma_y_fit, mux, psf_ysigma, cuda=cuda)
        if sigma_x_fit is None and sigma_fit is not None:
            sigx = xp.asarray(sigma_fit, dtype=xp.float32)
        if sigma_y_fit is None and sigma_fit is not None:
            sigy = xp.asarray(sigma_fit, dtype=xp.float32)
        theta = _parameter_or_default(sigma_angle, mux, psf_theta, cuda=cuda)
        return Gaussian2D(mux=mux, muy=muy, amp=amp, offset=offset, sigx=sigx, sigy=sigy, theta=theta, pixx=pixel[1], pixy=pixel[0], cuda=cuda)

    if model == "spline":
        muz = xp.zeros_like(mux) if z_fit is None else xp.asarray(z_fit, dtype=xp.float32)
        return Spline3D(mux=mux, muy=muy, muz=muz, amp=amp, offset=offset, tx=tx, ty=ty, tz=tz, coeffs=coeffs, cuda=cuda)

    raise ValueError(f"Unknown model: {model}")


def _channel_positions(crops, ch):
    """Return per-channel detection positions from one-based channel labels."""
    n_channels = len(crops)
    crop_lengths = [len(crop) for crop in crops]
    total = sum(crop_lengths)

    if ch is None:
        return None

    ch_np = _asnumpy(ch)
    if len(ch_np) != total:
        raise ValueError("ch must match the total number of crops")
    if total and (ch_np.min() < 1 or ch_np.max() > n_channels):
        raise ValueError("Channel indices must be one-based and within crops.")

    positions = []
    for channel_index, crop_length in enumerate(crop_lengths, start=1):
        pos = np.flatnonzero(ch_np == channel_index)
        if len(pos) != crop_length:
            raise ValueError("ch channel counts must match crop stack lengths")
        positions.append(pos)
    return positions


def _split_origins(crops, X0, Y0, positions, *, cuda=False):
    """Split crop origins into one x/y pair per channel."""
    if X0 is None or Y0 is None:
        raise ValueError("X0 and Y0 are required for weighted crop integration")

    n_channels = len(crops)
    crop_lengths = [len(crop) for crop in crops]
    if isinstance(X0, (list, tuple)) or isinstance(Y0, (list, tuple)):
        if not isinstance(X0, (list, tuple)) or not isinstance(Y0, (list, tuple)):
            raise ValueError("X0 and Y0 must use the same format")
        if len(X0) != n_channels or len(Y0) != n_channels:
            raise ValueError("X0 and Y0 must have same length as crops")
        _validate_channel_lengths(crop_lengths, X0, Y0)
        xp = get_xp(cuda)
        return [xp.asarray(x0) for x0 in X0], [xp.asarray(y0) for y0 in Y0]

    X0_channels = _split_detection_values(X0, crops, positions, cuda=cuda, name="X0")
    Y0_channels = _split_detection_values(Y0, crops, positions, cuda=cuda, name="Y0")
    return X0_channels, Y0_channels


def _split_detection_values(values, crops, positions, *, cuda=False, name="value", required=False):
    """Split a detection-aligned array or per-channel sequence by channel."""
    n_channels = len(crops)
    crop_lengths = [len(crop) for crop in crops]
    if values is None:
        if required:
            raise ValueError(f"{name} is required for weighted crop integration")
        return [None for _ in range(n_channels)]

    xp = get_xp(cuda)
    if isinstance(values, (list, tuple)) and len(values) == n_channels:
        for crop_length, value in zip(crop_lengths, values):
            if len(value) != crop_length:
                raise ValueError(f"{name} channel counts must match crop stack lengths")
        return [xp.asarray(value) for value in values]

    total = sum(crop_lengths)
    values = xp.asarray(values)
    if len(values) != total:
        raise ValueError(f"{name} must match the total number of crops")

    if positions is not None:
        return [values[xp.asarray(pos)] for pos in positions]

    channel_values = []
    start = 0
    for crop_length in crop_lengths:
        stop = start + crop_length
        channel_values.append(values[start:stop])
        start = stop
    return channel_values


def _normalize_channels_fit_models(values, n_channels):
    """Normalize and validate per-channel fit model names."""
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
        channels_pixels_nm = [(channels_pixels_nm, channels_pixels_nm) for _ in range(n_channels)]
    return channels_pixels_nm


def _normalize_channels_parameter(values, n_channels, name):
    """Normalize scalar/per-channel values to a per-channel sequence."""
    if isinstance(values, str):
        return [values for _ in range(n_channels)]
    try:
        if len(values) != n_channels:
            raise ValueError(f"{name} must have same length as crops")
    except TypeError:
        values = [values for _ in range(n_channels)]
    return values


def _normalize_spline_parameter(values, n_channels, name, required):
    """Normalize spline metadata to one object per channel."""
    if values is None:
        if required:
            raise SyntaxError(f"{name} must be specified for weighted spline channels")
        return [None for _ in range(n_channels)]
    if isinstance(values, np.ndarray):
        if n_channels != 1:
            raise ValueError(f"{name} must have same length as crops")
        return [values]
    if len(values) != n_channels:
        raise ValueError(f"{name} must have same length as crops")
    return values


def _parameter_or_default(values, reference, default, *, cuda=False):
    """Return fitted parameter values or a per-event default array."""
    xp = get_xp(cuda)
    if values is None:
        return xp.full_like(reference, fill_value=default, dtype=xp.float32)
    return xp.asarray(values, dtype=xp.float32)


def _validate_channel_lengths(crop_lengths, X0, Y0):
    """Validate per-channel origin array lengths."""
    for crop_length, x0, y0 in zip(crop_lengths, X0, Y0):
        if len(x0) != crop_length or len(y0) != crop_length:
            raise ValueError("X0/Y0 channel counts must match crop stack lengths")


def _asnumpy(array):
    """Return a NumPy view/copy for CPU-side remapping."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def stack_channel_values(values, positions=None, *, fill_value=np.nan):
    """Stack channel values or remap them to detection order."""
    if positions is None:
        return np.hstack(values)
    total = sum(len(pos) for pos in positions)
    dtype = np.result_type(*[np.asarray(value).dtype for value in values], np.float32)
    output = np.full(total, fill_value, dtype=dtype)
    for value, pos in zip(values, positions):
        output[pos] = value
    return output


if __name__ == "__main__":
    from corelp import test
    test(__file__)