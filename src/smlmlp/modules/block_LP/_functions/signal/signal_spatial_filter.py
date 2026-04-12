#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_correlate
import numpy as np



# %% Function
@block()
def signal_spatial_filter(
    channels,
    /,
    channels_spatial_kernels,
    signals=None,
    bkgds=None,
    noise_corrections=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Apply a spatial filter to enhance signals.

    This function applies a spatial correlation filter independently to each
    channel. If backgrounds are provided, they are subtracted before filtering.
    The noise correction factors are updated according to the norm of each
    spatial kernel.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of input channel stacks, one per channel.
    channels_spatial_kernels : sequence of ndarray
        Spatial kernels used by :func:`arrlp.img_correlate`, one per channel.
    signals : sequence of ndarray or None, optional
        Optional preallocated output arrays for the filtered signals. If
        provided and longer than the corresponding channel acquisition, each
        signal stack is truncated to the channel length.
    bkgds : sequence of ndarray or None, optional
        Optional background arrays to subtract before spatial filtering.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors. If ``None``, they are
        initialized to ``1.0`` for each channel.
    cuda : bool, optional
        Whether to enable CUDA processing.
    parallel : bool, optional
        Whether to enable parallel processing.
    
    Returns
    -------
    tuple
        A tuple ``(new_signals, noise_corrections, info)`` where:

        - ``new_signals`` is the list of spatially filtered signal stacks,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'channels_spatial_kernels'``
            Spatial kernels used for the correlation, one per channel.
        ``'kernel_factors'``
            Multiplicative noise-correction factors derived from the norm of
            each spatial kernel.

    Notes
    -----
    For each channel, the noise correction factor is updated using:

    .. math::

        \\sqrt{\\sum k^2}

    where :math:`k` is the spatial kernel for that channel.

    Examples
    --------
    >>> import numpy as np
    >>> channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    >>> kernels = [np.ones((3, 3), dtype=np.float32)]
    >>> signals, noise_corr, info = signal_spatial_filter(channels, kernels)
    >>> len(signals)
    1
    >>> len(info["kernel_factors"])
    1

    >>> bkgds = [np.zeros_like(channels[0])]
    >>> signals, noise_corr, info = signal_spatial_filter(
    ...     channels,
    ...     kernels,
    ...     bkgds=bkgds,
    ...     noise_corrections=[np.float32(2.0)],
    ... )
    >>> len(noise_corr)
    1
    """
    # If preallocated signal arrays are longer than the corresponding input
    # acquisition, truncate them to match the channel lengths.
    if signals is not None and len(signals[0]) > len(channels[0]):
        signals = [signal[:len(channel)] for channel, signal in zip(channels, signals)]

    # Initialize noise correction factors when none are provided.
    if noise_corrections is None:
        noise_corrections = [np.float32(1.0) for _ in range(len(channels))]

    kernel_factors = []
    new_signals = []

    for i in range(len(channels)):
        bkgd = None if bkgds is None else bkgds[i]
        signal = None if signals is None else signals[i]
        channel = channels[i]
        kernel = channels_spatial_kernels[i]

        # Subtract the background before filtering when available.
        if bkgd is not None:
            channel = channel - bkgd

        # Avoid reusing the same array as both input and output.
        if signal is channel:
            signal = None

        # Apply the spatial correlation filter.
        new_signal = img_correlate(
            channel,
            kernel=kernel,
            out=signal,
            cuda=cuda,
            parallel=parallel,
            stacks=True,
        )

        # Update the per-channel noise correction factor.
        factor = np.sqrt(np.sum(kernel ** 2))
        noise_corrections[i] *= factor
        kernel_factors.append(factor)
        new_signals.append(new_signal)

    info = {
        "channels_spatial_kernels": channels_spatial_kernels,
        "kernel_factors": kernel_factors,
    }

    return new_signals, noise_corrections, info