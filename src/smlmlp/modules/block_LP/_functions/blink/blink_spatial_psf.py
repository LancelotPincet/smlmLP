#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, get_xp, img_autocorr, img_fft, img_ifft, coordinates
import bottleneck as bn
from funclp import Gaussian2D, Spline2D
import numpy as np
from scipy.optimize import curve_fit



# %% Function
@block()
def blink_spatial_psf(
    channels,
    /,
    crop_pix=41,
    *,
    channels_pixels_nm=100.0,
    cuda=False,
    parallel=False,
):
    """
    Estimate a global PSF for each channel from spatial autocorrelations.

    This function computes, for each channel, a mean spatial autocorrelation,
    derives a PSF image from it, fits a 2D Gaussian model, and builds a 2D
    spline representation of the normalized PSF.

    The returned ``info`` dictionary gathers the main intermediate results for
    each channel, including the autocorrelation, PSF, Gaussian fit, spline
    reconstruction, and fitted parameters.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel. Each channel is expected to
        have shape ``(n_frames, height, width)``.
    crop_pix : int, optional
        Size of the square crop used around the autocorrelation center. The
        effective crop shape is ``(crop_pix, crop_pix)``.
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
        A tuple ``(psf_sigma, info)`` where:

        - ``psf_sigma`` is a list containing one effective PSF width per
          channel, computed as ``sqrt(sigx * sigy)``,
        - ``info`` is a dictionary containing the main intermediate results.

        The dictionary contains the following keys:

        ``'ac'``
            Mean cropped autocorrelation image for each channel.
        ``'psf'``
            Normalized PSF image for each channel.
        ``'fit'``
            Normalized Gaussian fit image for each channel.
        ``'spline'``
            Spline-interpolated PSF image for each channel.
        ``'sigx'``
            Fitted Gaussian sigma along x, in nanometers.
        ``'sigy'``
            Fitted Gaussian sigma along y, in nanometers.
        ``'theta'``
            Fitted Gaussian rotation angle, in degrees.
        ``'tx'``
            Spline knot positions along x.
        ``'ty'``
            Spline knot positions along y.
        ``'coeffs'``
            Spline coefficients.

    Notes
    -----
    The PSF image is computed from the autocorrelation using:

    .. math::

        \\mathrm{PSF} = \\mathcal{F}^{-1}\\left(\\sqrt{|\\mathcal{F}(AC)|}\\right)

    followed by centering, real-part extraction, and normalization.

    The Gaussian fit is performed away from the central row and column
    intersection defined by the initial mask ``(Y != 0) & (X != 0)``.

    Examples
    --------
    Estimate the PSF for one channel:

    >>> import numpy as np
    >>> channel = np.random.rand(20, 64, 64).astype(np.float32)
    >>> psf_sigma, info = blink_spatial_psf([channel], crop_pix=41)
    >>> len(psf_sigma)
    1
    >>> sorted(info.keys())
    ['ac', 'coeffs', 'fit', 'psf', 'sigx', 'sigy', 'spline', 'theta', 'tx', 'ty']

    Estimate the PSF for multiple channels with different pixel sizes:

    >>> channels = [
    ...     np.random.rand(20, 64, 64).astype(np.float32),
    ...     np.random.rand(20, 64, 64).astype(np.float32),
    ... ]
    >>> psf_sigma, info = blink_spatial_psf(
    ...     channels,
    ...     crop_pix=31,
    ...     channels_pixels_nm=[(100.0, 100.0), (110.0, 110.0)],
    ... )
    >>> len(info["psf"])
    2
    >>> len(info["sigx"])
    2
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

    # Build centered coordinate grids for the cropped region.
    cp = crop_pix // 2
    Y, X = coordinates((2 * cp + 1, 2 * cp + 1), grid=True)

    # Exclude the central horizontal and vertical axes during the Gaussian fit.
    mask = np.logical_and(Y != 0, X != 0)

    psf_sigma = []
    info = {
        "ac": [],
        "psf": [],
        "fit": [],
        "spline": [],
        "sigx": [],
        "sigy": [],
        "theta": [],
        "tx": [],
        "ty": [],
        "coeffs": [],
    }

    for channel, pix in zip(channels, channels_pixels_nm):
        # Estimate and remove a per-pixel temporal background before computing
        # the spatial autocorrelation.
        gc()
        if cuda:
            bkgd = xp.median(channel, axis=(0,), keepdims=True)
        else:
            bkgd = bn.median(channel, axis=0).reshape((1, *channel.shape[1:]))

        bkgd = xp.minimum(bkgd, channel)
        channel = channel - bkgd

        # Compute the mean cropped spatial autocorrelation around the image
        # center and normalize it while preserving the center value handling.
        y0 = int(channel.shape[1] // 2)
        x0 = int(channel.shape[2] // 2)

        ac = img_autocorr(channel, stacks=True, cuda=cuda, parallel=parallel)
        ac = ac[:, y0 - cp : y0 + cp + 1, x0 - cp : x0 + cp + 1]
        ac = ac.mean(axis=0)

        if cuda:
            ac = xp.asnumpy(ac)

        mid, ac[cp, cp] = ac[cp, cp], np.nan
        ac -= np.nanmin(ac)
        maxi = np.nanmax(ac)
        ac[cp, cp] = mid
        ac /= maxi

        # Recover a PSF image from the autocorrelation and normalize it.
        gc()
        psf = np.fft.fftshift(np.real(img_ifft(np.sqrt(np.abs(img_fft(ac))))))

        mid, psf[cp, cp] = psf[cp, cp], np.nan
        psf -= np.nanmin(psf)
        maxi = np.nanmax(psf)
        psf[cp, cp] = mid
        psf /= maxi

        # Fit a rotated 2D Gaussian model to the PSF away from the central
        # horizontal and vertical axes.
        gc()
        _psf = psf[mask]
        yy = Y[mask] * pix[0] / min(pix)
        xx = X[mask] * pix[1] / min(pix)

        # Initial parameters:
        # sigx, sigy, amp, offset, theta [deg]
        p0 = [1.0, 1.0, 1.0, 0.0, 0.0]

        # Lower and upper bounds for the Gaussian parameters.
        bounds = ([0.0, 0.0, 0.75, -0.25, 0.0], [cp, cp, 1.25, 0.25, 90.0])

        gaus = Gaussian2D(pixx=1.0, pixy=1.0, sigx=2.0, sigy=2.0)
        func2fit = (
            lambda xy, sigx, sigy, amp, offset, theta:
            gaus(
                xy[0],
                xy[1],
                sigx=sigx,
                sigy=sigy,
                amp=amp,
                offset=offset,
                theta=theta,
            )
        )

        popt, _ = curve_fit(func2fit, (xx, yy), _psf, p0=p0, bounds=bounds)
        sigx, sigy, amp, offset, theta = popt

        # Convert the fitted sigmas back to nanometers.
        sigx, sigy = sigx * min(pix), sigy * min(pix)

        # Rebuild the fitted Gaussian image on the full crop grid, then
        # normalize both the fit and the PSF using the fitted amplitude and
        # offset.
        _X = X * pix[1] / min(pix)
        _Y = Y * pix[0] / min(pix)
        fit = func2fit((_X, _Y), *popt)
        fit = (fit - offset) / amp
        psf = (psf - offset) / amp

        # Build a 2D spline model of the normalized PSF while excluding the
        # central row and column.
        yy = Y * pix[0]
        xx = X * pix[1]

        spline_mask = np.ones(psf.shape, dtype=bool)
        spline_mask[cp, :] = False
        spline_mask[:, cp] = False

        _psf = psf[spline_mask].reshape(2 * cp, 2 * cp)
        _yy = yy[spline_mask].reshape(2 * cp, 2 * cp)
        _xx = xx[spline_mask].reshape(2 * cp, 2 * cp)

        spline_model = Spline2D(_psf, _xx, _yy)
        spline = spline_model(xx, yy)

        # Store the effective PSF width and the main intermediate outputs.
        psf_sigma.append(np.sqrt(sigx * sigy))
        info["ac"].append(ac)
        info["psf"].append(psf)
        info["fit"].append(fit)
        info["spline"].append(spline)
        info["sigx"].append(sigx)
        info["sigy"].append(sigy)
        info["theta"].append(theta)
        info["tx"].append(spline_model.tx)
        info["ty"].append(spline_model.ty)
        info["coeffs"].append(spline_model.coeffs)

    return psf_sigma, info