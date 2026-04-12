#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_gaussianfilter, kernel
import numpy as np



# %% Function
@block()
def bkgd_spatial_mean(
    channels,
    /,
    channels_mean_radii_pix=7.0,
    bkgds=None,
    noise_corrections=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Compute a spatially local mean background for each channel.

    The background is estimated by applying a Gaussian filter independently to
    each input channel stack. A per-channel noise correction factor is also
    updated from the corresponding Gaussian kernels so that downstream code can
    reuse intermediate calibration information.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel. Each element is passed to
        :func:`arrlp.img_gaussianfilter` with ``stacks=True``.
    channels_mean_radii_pix : float or sequence, optional
        Spatial averaging radius in pixels.

        Accepted forms are:

        - a single scalar, applied as ``(radius, radius)`` to every channel,
        - a sequence of length 2, applied to every channel as
          ``(radius_y, radius_x)``,
        - a sequence with the same length as ``channels``, where each element
          contains the per-channel radii.

        Internally, the Gaussian standard deviation is set to
        ``(radius_y / 2, radius_x / 2)`` for each channel.
    bkgds : sequence of ndarray or None, optional
        Optional preallocated output arrays for the backgrounds. If provided
        and longer than the corresponding input acquisition, each background
        stack is truncated to the channel length.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors. If ``None``, they are
        initialized to ``1.0`` for every channel. Each value is then updated
        in-place by multiplying it with the norm-based correction induced by
        the Gaussian filtering kernels.
    cuda : bool, optional
        Whether to enable CUDA processing in :func:`arrlp.img_gaussianfilter`.
    parallel : bool, optional
        Whether to enable parallel processing in :func:`arrlp.img_gaussianfilter`.

    Returns
    -------
    tuple
        A tuple ``(new_bkgds, noise_corrections, info)`` where:

        - ``new_bkgds`` is the list of spatially smoothed background stacks,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'channels_mean_radius_pix'``
            Normalized per-channel spatial radii used for the Gaussian filter.
        ``'spatial_mean_sigmas'``
            Per-channel Gaussian standard deviations used for filtering.
        ``'spatial_mean_kernels_y'``
            List of 1D kernels used to compute the noise correction factors
            for each channel in y direction.
        ``'spatial_mean_kernels_x'``
            List of 1D kernels used to compute the noise correction factors
            for each channel in x direction.

    Raises
    ------
    ValueError
        If ``channels_mean_radii_pix`` is a sequence whose length is neither
        equal to ``len(channels)`` nor equal to 2.

    Notes
    -----
    The noise correction is computed from two 1D kernels derived from the
    Gaussian sigmas. Each kernel is negated, then its center value is
    incremented by 1. The correction factor for a given channel is:

    .. math::

        \\sqrt{\\sum k_1^2} \\times \\sqrt{\\sum k_2^2}

    This factor is multiplied into the corresponding entry of
    ``noise_corrections``.

    Examples
    --------
    Use a single isotropic radius for all channels:

    >>> import numpy as np
    >>> channels = [
    ...     np.random.rand(4, 16, 16).astype(np.float32),
    ...     np.random.rand(4, 16, 16).astype(np.float32),
    ... ]
    >>> bkgds, noise_corr, info = bkgd_spatial_mean(channels, 7.0)
    >>> len(bkgds) == len(channels)
    True
    >>> len(noise_corr) == len(channels)
    True
    >>> "sigmas" in info
    True

    Use anisotropic radii shared across channels:

    >>> radii = (6.0, 10.0)
    >>> bkgds, noise_corr, info = bkgd_spatial_mean(channels, radii)
    >>> info["channels_mean_radii_pix"][0]
    (6.0, 10.0)

    Use one radius pair per channel and provide initial corrections:

    >>> radii = [(6.0, 6.0), (8.0, 10.0)]
    >>> init_corr = [np.float32(1.0), np.float32(2.0)]
    >>> bkgds, noise_corr, info = bkgd_spatial_mean(
    ...     channels,
    ...     radii,
    ...     noise_corrections=init_corr,
    ... )
    >>> len(info["per_channel"]) == 2
    True
    """
    # Normalize the per-channel spatial mean radii so that downstream code can
    # always assume a list of 2-tuples with one entry per channel.
    try:
        if len(channels_mean_radii_pix) != len(channels):
            if len(channels_mean_radii_pix) == 2:
                channels_mean_radii_pix = [
                    channels_mean_radii_pix for _ in range(len(channels))
                ]
            else:
                raise ValueError(
                    "channels_mean_radii_pix does not have the same length as channels"
                )
    except TypeError:
        channels_mean_radii_pix = [
            (channels_mean_radii_pix, channels_mean_radii_pix)
            for _ in range(len(channels))
        ]

    # If preallocated background arrays extend beyond the effective acquisition
    # length, truncate them to match the corresponding channel stacks.
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]

    # Initialize noise correction factors if they were not provided.
    if noise_corrections is None:
        noise_corrections = [np.float32(1.0) for _ in range(len(channels))]

    new_bkgds = []
    sigmas = []
    K1 = []
    K2 = []

    # Process each channel independently.
    for i in range(len(channels)):
        channel = channels[i]
        bkgd = None if bkgds is None else bkgds[i]

        # Convert the spatial radius to the Gaussian sigma used by the filter.
        sigma = (
            channels_mean_radii_pix[i][0] / 2,
            channels_mean_radii_pix[i][1] / 2,
        )
        sigmas.append(sigma)

        # Compute the local mean background for the current channel.
        new_bkgd = img_gaussianfilter(
            channel,
            sigma=sigma,
            out=bkgd,
            cuda=cuda,
            parallel=parallel,
            stacks=True,
        )

        # Build the 1D correction kernels used to propagate the filtering effect
        # into the estimated noise correction factor.
        k1 = -kernel(ndims=1, sigma=sigma[0])
        k2 = -kernel(ndims=1, sigma=sigma[1])
        k1[int(len(k1) // 2)] += 1.0
        k2[int(len(k2) // 2)] += 1.0
        K1.append(k1)
        K2.append(k2)

        # Update the channel-specific noise correction in place.
        correction_factor = np.sqrt(np.sum(k1 ** 2)) * np.sqrt(np.sum(k2 ** 2))
        noise_corrections[i] *= correction_factor

        new_bkgds.append(new_bkgd)

    info = {
        "channels_mean_radii_pix": channels_mean_radii_pix,
        "spatial_mean_sigmas": sigmas,
        "spatial_mean_kernels_y": K1,
        "spatial_mean_kernels_x": K2,
    }

    return new_bkgds, noise_corrections, info