#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from stacklp import temporal_median
import numpy as np



# %% Function
@block()
def bkgd_temporal_median(
    channels,
    /,
    median_window_fr=25,
    bkgds=None,
    noise_corrections=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Compute a temporal local median background for each channel.

    This function estimates the background of each input channel by applying a
    temporal median filter along the frame axis. It also propagates the
    ``noise_corrections`` values so the output format remains consistent with
    other background estimation functions.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of channel stacks. Each element is expected to be a temporal
        image stack and is processed independently.
    median_window_fr : int, optional
        Temporal median window size, in frames.
    bkgds : sequence of ndarray or None, optional
        Optional preallocated output arrays for the computed backgrounds. If
        provided and longer than the corresponding acquisition, each background
        stack is truncated to the channel length.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors. If ``None``, they are
        initialized to ``1.0`` for each channel. In this function, the values
        are preserved because the original logic multiplies them by ``1.0``.
    cuda : bool, optional
        Whether to enable CUDA processing in :func:`stacklp.temporal_median`.
    parallel : bool, optional
        Whether to enable parallel processing in
        :func:`stacklp.temporal_median`.

    Returns
    -------
    tuple
        A tuple ``(new_bkgds, noise_corrections, info)`` where:

        - ``new_bkgds`` is the list of computed background stacks,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'median_window_fr'``
            Temporal window size (in frames) used for the median filter.

    Examples
    --------
    Use the same temporal median window for all channels:

    >>> import numpy as np
    >>> channels = [
    ...     np.random.rand(10, 16, 16).astype(np.float32),
    ...     np.random.rand(10, 16, 16).astype(np.float32),
    ... ]
    >>> bkgds, noise_corr, info = bkgd_temporal_median(channels, median_window_fr=25)
    >>> len(bkgds)
    2
    >>> len(noise_corr)
    2
    >>> info["median_window_fr"]
    25

    Reuse preallocated output arrays:

    >>> out = [np.empty_like(ch) for ch in channels]
    >>> bkgds, noise_corr, info = bkgd_temporal_median(
    ...     channels,
    ...     median_window_fr=11,
    ...     bkgds=out,
    ... )
    >>> len(bkgds) == len(channels)
    True

    Provide existing noise corrections:

    >>> init_corr = [np.float32(1.0), np.float32(2.0)]
    >>> bkgds, noise_corr, info = bkgd_temporal_median(
    ...     channels,
    ...     median_window_fr=9,
    ...     noise_corrections=init_corr,
    ... )
    >>> noise_corr[1]
    2.0
    """
    # If preallocated background arrays are longer than the corresponding input
    # acquisition, truncate them to match the channel lengths.
    if bkgds is not None and len(bkgds[0]) > len(channels[0]):
        bkgds = [bkgd[:len(channel)] for channel, bkgd in zip(channels, bkgds)]

    # Initialize noise correction factors when none are provided.
    if noise_corrections is None:
        noise_corrections = [np.float32(1.0) for _ in range(len(channels))]

    new_bkgds = []
    for i in range(len(channels)):
        channel = channels[i]
        bkgd = None if bkgds is None else bkgds[i]

        # Compute the temporal median background for the current channel.
        new_bkgd = temporal_median(
            channel,
            median_window_fr,
            out=bkgd,
            cuda=cuda,
            parallel=parallel,
        )

        # Preserve the original behavior for the noise correction update.
        noise_corrections[i] *= np.float32(1.0)
        new_bkgds.append(new_bkgd)

    info = {
        "median_window_fr": median_window_fr,
    }

    return new_bkgds, noise_corrections, info