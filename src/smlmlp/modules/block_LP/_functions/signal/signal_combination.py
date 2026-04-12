#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, signal_spatial_filter, signal_temporal_filter
from arrlp import gc



# %% Function
@block(timeit=False)
def signal_combination(
    channels,
    /,
    channels_spatial_kernels=None,
    temporal_kernel=None,
    signals=None,
    bkgds=None,
    noise_corrections=None,
    *,
    do_spatial_filter=True,
    do_temporal_filter=False,
    cuda=False,
    parallel=False,
):
    """
    Apply a sequence of signal-enhancement filters.

    This function combines the spatial and temporal signal filters into a
    single processing pipeline. Each enabled step is applied in sequence, and
    the output of one step becomes the input of the next one.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of input channel stacks, one per channel.
    channels_spatial_kernels : sequence of ndarray or None, optional
        Spatial kernels forwarded to :func:`signal_spatial_filter`.
    temporal_kernel : array-like or None, optional
        Temporal kernel forwarded to :func:`signal_temporal_filter`.
    signals : sequence of ndarray or None, optional
        Optional preallocated output arrays reused across filtering steps.
    bkgds : sequence of ndarray or None, optional
        Optional background arrays subtracted before the first enabled
        filtering step.
    noise_corrections : sequence of float or None, optional
        Optional per-channel noise correction factors propagated through the
        enabled filtering steps.
    do_spatial_filter : bool, optional
        Whether to apply :func:`signal_spatial_filter`.
    do_temporal_filter : bool, optional
        Whether to apply :func:`signal_temporal_filter`.
    cuda : bool, optional
        Whether to enable CUDA processing.
    parallel : bool, optional
        Whether to enable parallel processing.

    Returns
    -------
    tuple
        A tuple ``(signals, noise_corrections, info)`` where:

        - ``signals`` is the list of filtered signal stacks,
        - ``noise_corrections`` is the updated list of correction factors,
        - ``info`` is a dictionary aggregating intermediate results from all
          executed filtering steps.

        The dictionary is built by updating a single dictionary with the
        ``info`` outputs of each enabled step. Depending on the activated
        methods, it may contain:

        From spatial filtering:

        ``'channels_spatial_kernels'``
            Spatial kernels used for the correlation, one per channel.
        ``'kernel_factors'``
            Multiplicative noise-correction factors derived from the norm of
            each spatial kernel.

        From temporal filtering:

        ``'temporal_kernel'``
            Temporal kernel used for the correlation.
        ``'kernel_factor'``
            Multiplicative noise-correction factor derived from the temporal
            kernel norm.

    Notes
    -----
    After each enabled filtering step, the resulting signals are copied into
    internal buffers so they can safely be reused as input for the next step.

    Examples
    --------
    Apply only spatial filtering:

    >>> import numpy as np
    >>> channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    >>> spatial_kernels = [np.ones((3, 3), dtype=np.float32)]
    >>> signals, noise_corr, info = signal_combination(
    ...     channels,
    ...     channels_spatial_kernels=spatial_kernels,
    ...     do_spatial_filter=True,
    ...     do_temporal_filter=False,
    ... )
    >>> len(signals)
    1
    >>> "channels_spatial_kernels" in info
    True

    Apply spatial and temporal filtering in sequence:

    >>> temporal_kernel = np.array([1.0, -1.0], dtype=np.float32)
    >>> signals, noise_corr, info = signal_combination(
    ...     channels,
    ...     channels_spatial_kernels=spatial_kernels,
    ...     temporal_kernel=temporal_kernel,
    ...     do_spatial_filter=True,
    ...     do_temporal_filter=True,
    ... )
    >>> "temporal_kernel" in info
    True
    """
    # Working buffers used to safely pass the output of one filtering step to
    # the next one.
    buffers = None

    # Collect reusable intermediate outputs from the executed filtering steps
    # in a single dictionary.
    info = {}

    if do_spatial_filter:
        assert channels_spatial_kernels is not None

        gc()
        signals, noise_corrections, spatial_info = signal_spatial_filter(
            channels,
            channels_spatial_kernels,
            signals=signals,
            noise_corrections=noise_corrections,
            bkgds=bkgds,
            cuda=cuda,
            parallel=parallel,
        )
        info.update(spatial_info)

        # Copy the filtered signals into reusable buffers.
        if buffers is None:
            buffers = [signal.copy() for signal in signals]
        else:
            for i in range(len(channels)):
                buffers[i][:] = signals[i]

        # The next step uses the filtered signals as input and should not
        # subtract the original backgrounds again.
        channels = buffers
        bkgds = None

    if do_temporal_filter:
        assert temporal_kernel is not None

        gc()
        signals, noise_corrections, temporal_info = signal_temporal_filter(
            channels,
            temporal_kernel,
            signals=signals,
            noise_corrections=noise_corrections,
            bkgds=bkgds,
            cuda=cuda,
            parallel=parallel,
        )
        info.update(temporal_info)

        # Copy the filtered signals into reusable buffers.
        if buffers is None:
            buffers = [signal.copy() for signal in signals]
        else:
            for i in range(len(channels)):
                buffers[i][:] = signals[i]

        # The output of this step becomes the current working signal.
        channels = buffers
        bkgds = None

    gc()

    return signals, noise_corrections, info