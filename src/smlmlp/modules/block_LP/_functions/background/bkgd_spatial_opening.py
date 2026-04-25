#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import img_greyopening, kernel
import numpy as np



@block()
def bkgd_spatial_opening(
    channels,
    /,
    channels_opening_radii_pix=3.0,
    bkgds=None,
    noise_corrections=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Compute a spatial background using morphological grey opening.

    This function estimates a background for each input channel by applying a
    grey opening with a channel-specific footprint. It also propagates the
    ``noise_corrections`` values so the output keeps a uniform return format
    across background estimation functions.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel. Each channel is processed
        independently with ``stacks=True``.
    channels_opening_radii_pix : float or sequence, optional
        Opening radius in pixels.

        Accepted forms are:

        - a single scalar, applied as ``(radius, radius)`` to every channel,
        - a sequence of length 2, applied to every channel as
          ``(radius_y, radius_x)``,
        - a sequence with the same length as ``channels``, where each element
          gives the radius pair for one channel.

        For each channel, the footprint window is built as
        ``(2 * radius_y, 2 * radius_x)``.
    bkgds : sequence of ndarray or None, optional
        Optional preallocated output arrays for the computed backgrounds. If
        provided and longer than the corresponding acquisition, each background
        stack is truncated to the channel length.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors. If ``None``, they are
        initialized to ``1.0`` for each channel. In this function, the values
        are preserved because the original logic multiplies them by ``1.0``.
    cuda : bool, optional
        Whether to enable CUDA processing in :func:`arrlp.img_greyopening`.
    parallel : bool, optional
        Whether to enable parallel processing in
        :func:`arrlp.img_greyopening`.

    Returns
    -------
    tuple
        A tuple ``(new_bkgds, noise_corrections, info)`` where:

        - ``new_bkgds`` is the list of computed background stacks,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'channels_opening_radius_pix'``
            Normalized per-channel opening radii.
        ``'footprints'``
            List of structuring elements (footprints) used for each channel.

    Raises
    ------
    ValueError
        If ``channels_opening_radii_pix`` is a sequence whose length is
        neither equal to ``len(channels)`` nor equal to 2.

    Notes
    -----
    The footprint for each channel is generated with :func:`arrlp.kernel`
    using:

    .. math::

        (2 r_y, 2 r_x)

    where :math:`r_y` and :math:`r_x` are the opening radii in pixels for that
    channel.

    Examples
    --------
    Use a single isotropic radius for all channels:

    >>> import numpy as np
    >>> channels = [
    ...     np.random.rand(3, 16, 16).astype(np.float32),
    ...     np.random.rand(3, 16, 16).astype(np.float32),
    ... ]
    >>> bkgds, noise_corr, info = bkgd_spatial_opening(channels, 3.0)
    >>> len(bkgds)
    2
    >>> len(noise_corr)
    2
    >>> len(info["footprints"])
    2

    Use one anisotropic radius shared across all channels:

    >>> bkgds, noise_corr, info = bkgd_spatial_opening(channels, (3.0, 5.0))
    >>> len(info["footprints"]) == len(channels)
    True

    Use one radius pair per channel:

    >>> radii = [(3.0, 3.0), (4.0, 6.0)]
    >>> bkgds, noise_corr, info = bkgd_spatial_opening(
    ...     channels,
    ...     channels_opening_radii_pix=radii,
    ... )
    >>> len(info["channels_opening_radii_pix"])
    2
    """
    # Normalize the per-channel opening radii so that the rest of the function
    # can always work with one 2-tuple per channel.
    try:
        if len(channels_opening_radii_pix) != len(channels):
            if len(channels_opening_radii_pix) == 2:
                channels_opening_radii_pix = [
                    channels_opening_radii_pix for _ in range(len(channels))
                ]
            else:
                raise ValueError(
                    "channel_mean_radius_pix does not have the same length as channels"
                )
    except TypeError:
        channels_opening_radii_pix = [
            (channels_opening_radii_pix, channels_opening_radii_pix)
            for _ in range(len(channels))
        ]

    # If preallocated background arrays are longer than the corresponding input
    # acquisition, truncate them to match the channel lengths.
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]

    # Initialize noise correction factors when none are provided.
    if noise_corrections is None:
        noise_corrections = [np.float32(1.0) for _ in range(len(channels))]

    # Build one morphological footprint per channel.
    footprints = [
        kernel(window=(2 * rad_pix[0], 2 * rad_pix[1]))
        for rad_pix in channels_opening_radii_pix
    ]

    new_bkgds = []
    for i in range(len(channels)):
        channel = channels[i]
        bkgd = None if bkgds is None else bkgds[i]

        # Compute the grey-opening background for the current channel.
        new_bkgd = img_greyopening(
            channel,
            footprint=footprints[i],
            out=bkgd,
            cuda=cuda,
            parallel=parallel,
            stacks=True,
        )

        # Preserve the original behavior for the noise correction update.
        noise_corrections[i] *= np.float32(1.0)
        new_bkgds.append(new_bkgd)

    info = {
        "channels_opening_radii_pix": channels_opening_radii_pix,
        "footprints": footprints,
    }

    return new_bkgds, noise_corrections, info
