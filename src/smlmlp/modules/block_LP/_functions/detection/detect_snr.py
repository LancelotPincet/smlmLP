#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import get_xp, nb_threads
import numba as nb
from numba import cuda as nb_cuda
import math



@block()
def detect_snr(
    signals,
    bkgds,
    noise_corrections=None,
    channels_gains=0.25,
    *,
    cuda=False,
    parallel=False,
):
    """
    Normalize signals into signal-to-noise ratio values.

    This function converts each signal array into SNR units using the provided
    background, noise correction factor, and channel gain. The computation is
    performed in-place on each signal array, either on CPU or GPU.

    Parameters
    ----------
    signals : sequence of array-like
        Sequence of signal arrays, one per channel.
    bkgds : sequence of array-like
        Sequence of background arrays matching ``signals``.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors. If ``None``, a value of
        ``1.0`` is used for every channel.
    channels_gains : float or sequence, optional
        Experimental gain value(s) used to normalize the backgrounds. This
        value is normalized through :class:`smlmlp.Config` so that one gain is
        available for each channel.
    cuda : bool, optional
        Whether to use GPU acceleration.
    parallel : bool, optional
        Whether to enable CPU parallelization.

    Returns
    -------
    tuple
        A tuple ``(snrs, info)`` where:

        - ``snrs`` is the list of SNR arrays, one per channel,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'noise_corrections'``
            Per-channel noise correction factors effectively used.
        ``'channels_gains'``
            Per-channel gain values effectively used.

    Notes
    -----
    The normalization is performed in-place using:

    .. math::

        \\mathrm{SNR} = \\frac{\\mathrm{signal}}
        {\\mathrm{noise\\_correction} \\times
        \\sqrt{\\mathrm{bkgd} / \\mathrm{gain}}}

    Examples
    --------
    >>> import numpy as np
    >>> signals = [np.array([[10., 20.], [30., 40.]], dtype=np.float32)]
    >>> bkgds = [np.array([[4., 4.], [4., 4.]], dtype=np.float32)]
    >>> snrs, info = detect_snr(signals, bkgds, channels_gains=1.0)
    >>> len(snrs)
    1
    >>> snrs[0].shape
    (2, 2)
    >>> "channels_gains" in info
    True

    >>> signals = [
    ...     np.array([[10., 20.]], dtype=np.float32),
    ...     np.array([[5., 15.]], dtype=np.float32),
    ... ]
    >>> bkgds = [
    ...     np.array([[4., 4.]], dtype=np.float32),
    ...     np.array([[9., 9.]], dtype=np.float32),
    ... ]
    >>> snrs, info = detect_snr(
    ...     signals,
    ...     bkgds,
    ...     noise_corrections=[1.0, 2.0],
    ...     channels_gains=[0.5, 1.0],
    ... )
    >>> len(snrs)
    2
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Normalize the gain input so that one gain value is available per channel.
    try:
        if len(channels_gains) != len(signals):
            raise ValueError(
                "channels_gains does not have the same length as channels"
            )
    except TypeError:
        channels_gains = [
            channels_gains
            for _ in range(len(signals))
        ]

    # Initialize the noise correction factors when they are not provided.
    if noise_corrections is None:
        noise_corrections = [xp.float32(1.0) for _ in range(len(signals))]

    snrs = []

    for i in range(len(signals)):
        signal = xp.asarray(signals[i], dtype=xp.float32)
        bkgd = xp.asarray(bkgds[i], dtype=xp.float32)
        noise_correction = xp.float32(noise_corrections[i])
        gain = xp.float32(channels_gains[i])

        # Convert the signal array into SNR units in-place.
        if cuda:
            threads_per_block = 256
            blocks_per_grid = (
                signal.size + threads_per_block - 1
            ) // threads_per_block
            snr_gpu[
                blocks_per_grid,
                threads_per_block,
            ](signal.ravel(), bkgd.ravel(), noise_correction, gain)
        else:
            with nb_threads(parallel):
                snr_cpu(signal.ravel(), bkgd.ravel(), noise_correction, gain)

        snrs.append(signal)

    info = {
        "noise_corrections": noise_corrections,
        "channels_gains": channels_gains,
    }

    return snrs, info



@nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
def snr_cpu(signal, bkgd, noise_correction, gain):
    """Convert a flattened signal array into SNR values on CPU."""
    for i in nb.prange(len(signal)):
        signal[i] = nb.float32(
            signal[i] / noise_correction / math.sqrt(bkgd[i] / gain)
        )



@nb_cuda.jit(fastmath=True, cache=True)
def snr_gpu(signal, bkgd, noise_correction, gain):
    """Convert a flattened signal array into SNR values on GPU."""
    i = nb_cuda.grid(1)
    if i < len(signal):
        signal[i] = nb.float32(
            signal[i] / noise_correction / math.sqrt(bkgd[i] / gain)
        )
