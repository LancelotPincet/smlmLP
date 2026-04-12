#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from stacklp import temporal_correlate
import numpy as np



# %% Function
@block()
def signal_temporal_filter(
    channels,
    /,
    temporal_kernel,
    signals=None,
    bkgds=None,
    noise_corrections=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    Apply a temporal filter to enhance signals.

    This function applies a temporal correlation filter independently to each
    channel. If backgrounds are provided, they are subtracted before filtering.
    The noise correction factors are updated according to the norm of the
    temporal kernel.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of input channel stacks, one per channel.
    temporal_kernel : array-like
        Temporal kernel used by :func:`stacklp.temporal_correlate`.
    signals : sequence of ndarray or None, optional
        Optional preallocated output arrays for the filtered signals. If
        provided and longer than the corresponding channel acquisition, each
        signal stack is truncated to the channel length.
    bkgds : sequence of ndarray or None, optional
        Optional background arrays to subtract before temporal filtering.
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

        - ``new_signals`` is the list of temporally filtered signal stacks,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'temporal_kernel'``
            Temporal kernel used for the correlation.
        ``'kernel_factor'``
            Multiplicative noise-correction factor derived from the kernel norm.

    Notes
    -----
    The noise correction factor is updated using:

    .. math::

        \\sqrt{\\sum k^2}

    where :math:`k` is the temporal kernel.

    Examples
    --------
    >>> import numpy as np
    >>> channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    >>> kernel = np.array([1.0, -1.0], dtype=np.float32)
    >>> signals, noise_corr, info = signal_temporal_filter(channels, kernel)
    >>> len(signals)
    1
    >>> "kernel_factor" in info
    True

    >>> bkgds = [np.zeros_like(channels[0])]
    >>> signals, noise_corr, info = signal_temporal_filter(
    ...     channels,
    ...     kernel,
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

    # Precompute the kernel-dependent correction factor.
    kernel = temporal_kernel
    factor = np.sqrt(np.sum(kernel ** 2))

    new_signals = []
    for i in range(len(channels)):
        bkgd = None if bkgds is None else bkgds[i]
        signal = None if signals is None else signals[i]
        channel = channels[i]

        # Subtract the background before filtering when available.
        if bkgd is not None:
            channel = channel - bkgd

        # Avoid reusing the same array as both input and output.
        if signal is channel:
            signal = None

        # Apply the temporal correlation filter.
        new_signal = temporal_correlate(
            channel,
            kernel=kernel,
            out=signal,
            cuda=cuda,
            parallel=parallel,
        )

        # Update the per-channel noise correction factor.
        noise_corrections[i] *= factor
        new_signals.append(new_signal)

    info = {
        "temporal_kernel": kernel,
        "kernel_factor": factor,
    }

    return new_signals, noise_corrections, info