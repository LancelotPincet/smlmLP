#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_transform, get_xp, transform_matrix, compress
import numpy as np



# %% Function
@block()
def registrate_optimize_images(
    channels,
    /,
    channels_x_shifts_nm=None,
    channels_y_shifts_nm=None,
    channels_rotations_deg=None,
    channels_x_shears=None,
    channels_y_shears=None,
    optimized=None,
    *,
    channels_pixels_nm=1.0,
    cuda=False,
    parallel=False,
):
    """
    Transform and normalize channel images to facilitate registration.

    This function rescales each channel to a common reference pixel size,
    applies geometric transformations, and compresses the intensity range in
    order to produce registration-friendly images.

    Parameters
    ----------
    channels : sequence of ndarray
        Sequence of image stacks, one per channel.
    channels_x_shifts_nm : sequence of float
        Per-channel shifts along x, in nanometers.
    channels_y_shifts_nm : sequence of float
        Per-channel shifts along y, in nanometers.
    channels_rotations_deg : sequence of float
        Per-channel rotations, in degrees.
    channels_x_shears : sequence of float
        Per-channel shear values along x.
    channels_y_shears : sequence of float
        Per-channel shear values along y.
    optimized : sequence of ndarray or None, optional
        Optional preallocated output arrays for the transformed images.
    channels_pixels_nm : float or sequence, optional
        Pixel size in nanometers. Can be scalar, ``(y, x)``, or per-channel.
    cuda : bool, optional
        Whether to enable CUDA processing.
    parallel : bool, optional
        Whether to enable parallel processing.

    Returns
    -------
    tuple
        A tuple ``(new_optimized, ref_pix, info)`` where:

        - ``new_optimized`` is the list of transformed and compressed images,
        - ``ref_pix`` is the reference pixel size used for the rescaling,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'channels_pixels_nm'``
            Normalized per-channel pixel sizes.
        ``'ref_pix'``
            Reference pixel size used for the common scaling.
        ``'scales_x'``
            Per-channel x scaling factors.
        ``'scales_y'``
            Per-channel y scaling factors.

    Examples
    --------
    >>> import numpy as np
    >>> channels = [np.random.rand(5, 16, 16).astype(np.float32)]
    >>> optimized, ref_pix, info = registrate_optimize_images(
    ...     channels,
    ...     channels_x_shifts_nm=[0.0],
    ...     channels_y_shifts_nm=[0.0],
    ...     channels_rotations_deg=[0.0],
    ...     channels_x_shears=[0.0],
    ...     channels_y_shears=[0.0],
    ... )
    >>> len(optimized)
    1
    >>> ref_pix
    (1.0, 1.0)

    >>> channels = [
    ...     np.random.rand(5, 16, 16).astype(np.float32),
    ...     np.random.rand(5, 16, 16).astype(np.float32),
    ... ]
    >>> optimized, ref_pix, info = registrate_optimize_images(
    ...     channels,
    ...     channels_x_shifts_nm=[0.0, 20.0],
    ...     channels_y_shifts_nm=[0.0, -10.0],
    ...     channels_rotations_deg=[0.0, 1.0],
    ...     channels_x_shears=[0.0, 0.01],
    ...     channels_y_shears=[0.0, -0.01],
    ...     channels_pixels_nm=[(100.0, 100.0), (110.0, 120.0)],
    ... )
    >>> len(info["scales_x"])
    2
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Normalize the per-channel pixel sizes so the rest of the function can
    # always work with one (y, x) tuple per channel.
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

    # Use the smallest pixel sizes across channels as the common reference.
    ref_pix = (
        min([pix[0] for pix in channels_pixels_nm]),
        min([pix[1] for pix in channels_pixels_nm]),
    )
    scales_x = [ref_pix[1] / pix[1] for pix in channels_pixels_nm]
    scales_y = [ref_pix[0] / pix[0] for pix in channels_pixels_nm]

    # Initialize transformations
    channels_x_shifts_nm = np.zeros(len(channels), dtype=np.float32) if channels_x_shifts_nm is None else channels_x_shifts_nm
    channels_y_shifts_nm = np.zeros(len(channels), dtype=np.float32) if channels_y_shifts_nm is None else channels_y_shifts_nm
    channels_rotations_deg = np.zeros(len(channels), dtype=np.float32) if channels_rotations_deg is None else channels_rotations_deg
    channels_x_shears = np.zeros(len(channels), dtype=np.float32) if channels_x_shears is None else channels_x_shears
    channels_y_shears = np.zeros(len(channels), dtype=np.float32) if channels_y_shears is None else channels_y_shears

    new_optimized = []
    for i in range(len(channels)):
        channel = xp.asarray(channels[i])
        optimize = None if optimized is None else optimized[i]

        # Build the rescaling and registration transforms, then combine them.
        matrix1 = transform_matrix(
            channel,
            scalex=scales_x[i],
            scaley=scales_y[i],
        )
        matrix2 = transform_matrix(
            channel,
            shiftx=channels_x_shifts_nm[i] / ref_pix[1],
            shifty=channels_y_shifts_nm[i] / ref_pix[0],
            angle=channels_rotations_deg[i],
            shearx=channels_x_shears[i],
            sheary=channels_y_shears[i],
        )
        matrix = matrix1 @ matrix2

        # Apply the geometric transform and then compress the intensity range.
        optimize = img_transform(
            channel,
            matrix=matrix,
            out=optimize,
            cuda=cuda,
            parallel=parallel,
            stacks=True,
        )
        optimize = compress(
            optimize,
            out=optimize,
            white_percent=1,
            black_percent=1,
            saturate=True,
            stacks=True,
        )
        new_optimized.append(optimize)

    info = {
        "channels_pixels_nm": channels_pixels_nm,
        "ref_pix": ref_pix,
        "scales_x": scales_x,
        "scales_y": scales_y,
    }

    return new_optimized, info