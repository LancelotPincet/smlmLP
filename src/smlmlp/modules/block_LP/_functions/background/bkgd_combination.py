#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, bkgd_spatial_opening, bkgd_temporal_median, bkgd_spatial_mean
from arrlp import gc, get_xp



# %% Function
@block(timeit=False)
def bkgd_combination(
    channels,
    /,
    bkgds=None,
    noise_corrections=None,
    *,
    do_spatial_opening=False,
    channels_opening_radii_pix=3.0,
    do_temporal_median=True,
    median_window_fr=25,
    do_spatial_mean=True,
    channels_mean_radii_pix=7.0,
    cuda=False,
    parallel=False,
):
    """
    Compute a background by combining several background estimation steps.

    This function applies up to three background estimation methods in
    sequence:

    1. spatial opening,
    2. temporal median,
    3. spatial mean.

    After each enabled step, the estimated background is subtracted from the
    current working channels. At the end of the pipeline, the final background
    is reconstructed from the difference between the original input channels
    and the final residual channels.

    The function preserves the original computation logic. The returned
    ``info`` dictionary is built by updating a single dictionary with the
    intermediate ``info`` outputs returned by the sub-functions that were
    actually executed.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of input channel stacks to process.
    bkgds : sequence of ndarray or None, optional
        Optional output arrays used to store the background estimates.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors propagated through the
        enabled background estimation steps.
    do_spatial_opening : bool, optional
        Whether to apply :func:`bkgd_spatial_opening` first.
    channels_opening_radii_pix : float or sequence, optional
        Opening radii parameter forwarded to
        :func:`bkgd_spatial_opening`.
    do_temporal_median : bool, optional
        Whether to apply :func:`bkgd_temporal_median`.
    median_window_fr : int, optional
        Temporal median window, in frames, forwarded to
        :func:`bkgd_temporal_median`.
    do_spatial_mean : bool, optional
        Whether to apply :func:`bkgd_spatial_mean`.
    channels_mean_radii_pix : float or sequence, optional
        Spatial mean radii parameter forwarded to
        :func:`bkgd_spatial_mean`.
    cuda : bool, optional
        Whether to use CUDA-enabled array operations when supported.
    parallel : bool, optional
        Whether to enable parallel execution in the called sub-functions.

    Returns
    -------
    tuple
        A tuple ``(bkgds, noise_corrections, info)`` where:

        - ``bkgds`` is the final reconstructed background for each channel,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary aggregating intermediate results from all
          executed background estimation steps.

        The dictionary is built by updating a single dictionary with the
        ``info`` outputs of each enabled step. Depending on the activated
        methods, it may contain:

        From spatial opening:

        ``'channels_opening_radii_pix'``
            Per-channel opening radii.
        ``'footprints'``
            Structuring elements used for each channel.

        From temporal median:

        ``'median_window_fr'``
            Temporal window size used for the median filter.

        From spatial mean:

        ``'channels_mean_radii_pix'``
            Per-channel spatial radii.
        ``'sigmas'``
            Gaussian standard deviations used for filtering.
        ``'kernels'``
            1D kernels used for noise correction estimation.

    Notes
    -----
    The working channels are updated in-place whenever possible through the
    backend returned by :func:`arrlp.get_xp`.

    If no background step is enabled:

    - ``bkgds`` is created as zeros if it was initially ``None``,
    - otherwise the provided output backgrounds are filled with zeros.

    Examples
    --------
    Apply the default combination of temporal median and spatial mean:

    >>> import numpy as np
    >>> channels = [
    ...     np.random.rand(10, 16, 16).astype(np.float32),
    ...     np.random.rand(10, 16, 16).astype(np.float32),
    ... ]
    >>> bkgds, noise_corr, info = bkgd_combination(channels)
    >>> len(bkgds)
    2
    >>> len(noise_corr)
    2
    >>> isinstance(info, dict)
    True

    Apply only spatial opening:

    >>> bkgds, noise_corr, info = bkgd_combination(
    ...     channels,
    ...     do_spatial_opening=True,
    ...     do_temporal_median=False,
    ...     do_spatial_mean=False,
    ...     channels_opening_radii_pix=4.0,
    ... )
    >>> "footprints" in info
    True

    Disable all background estimation steps:

    >>> bkgds, noise_corr, info = bkgd_combination(
    ...     channels,
    ...     do_spatial_opening=False,
    ...     do_temporal_median=False,
    ...     do_spatial_mean=False,
    ... )
    >>> len(bkgds) == len(channels)
    True
    """
    # Keep a reference to the original inputs so the final background can be
    # reconstructed after the successive subtraction steps.
    raws = channels

    # Working buffer used to store the progressively background-corrected
    # channels.
    buffers = None

    # Select the array backend matching the current execution mode.
    xp = get_xp(cuda)

    # Collect reusable intermediate outputs from the executed background
    # estimation steps in a single dictionary.
    info = {}

    if do_spatial_opening:
        gc()
        bkgds, noise_corrections, opening_info = bkgd_spatial_opening(
            channels,
            channels_opening_radii_pix=channels_opening_radii_pix,
            noise_corrections=noise_corrections,
            bkgds=bkgds,
            cuda=cuda,
            parallel=parallel,
        )
        info.update(opening_info)

        # Subtract the estimated background from the current working channels.
        if buffers is None:
            channels = [channel - bkgd for channel, bkgd in zip(channels, bkgds)]
            buffers = channels
        else:
            for i in range(len(channels)):
                xp.subtract(channels[i], bkgds[i], buffers[i])

    if do_temporal_median:
        gc()
        bkgds, noise_corrections, median_info = bkgd_temporal_median(
            channels,
            median_window_fr=median_window_fr,
            noise_corrections=noise_corrections,
            bkgds=bkgds,
            cuda=cuda,
            parallel=parallel,
        )
        info.update(median_info)

        # Subtract the estimated background from the current working channels.
        if buffers is None:
            channels = [channel - bkgd for channel, bkgd in zip(channels, bkgds)]
            buffers = channels
        else:
            for i in range(len(channels)):
                xp.subtract(channels[i], bkgds[i], buffers[i])

    if do_spatial_mean:
        gc()
        bkgds, noise_corrections, mean_info = bkgd_spatial_mean(
            channels,
            channels_mean_radii_pix=channels_mean_radii_pix,
            noise_corrections=noise_corrections,
            bkgds=bkgds,
            cuda=cuda,
            parallel=parallel,
        )
        info.update(mean_info)

        # Subtract the estimated background from the current working channels.
        if buffers is None:
            channels = [channel - bkgd for channel, bkgd in zip(channels, bkgds)]
            buffers = channels
        else:
            for i in range(len(channels)):
                xp.subtract(channels[i], bkgds[i], buffers[i])

    gc()

    # If no processing step produced residual buffers, the final background is
    # simply zero.
    if buffers is None:
        if bkgds is None:
            bkgds = [xp.zeros_like(channel) for channel in channels]
        else:
            for bkgd in bkgds:
                bkgd[:] = 0
    else:
        # Reconstruct the full background from the original channels and the
        # final residual channels.
        for i in range(len(channels)):
            xp.subtract(raws[i], channels[i], bkgds[i])

    return bkgds, noise_corrections, info