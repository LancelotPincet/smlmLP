#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block, Config
from arrlp import get_xp, nb_threads
import numba as nb
from numba import cuda as nb_cuda
import numpy as np
import math



@block()
def detect_spatial_maxima(
    snrs,
    /,
    snr_thresh,
    channels_spatial_kernels,
    *,
    f0=0,
    channels_pixels_nm=1.0,
    cuda=False,
    parallel=False,
):
    """
    Detect local spatial maxima in thresholded SNR images.

    This function detects local maxima in SNR stacks using a channel-specific
    spatial footprint derived from the provided kernels. Detected maxima are
    refined to subpixel coordinates through a local center-of-mass estimate,
    then converted to physical units using the channel pixel sizes.

    Parameters
    ----------
    snrs : sequence of ndarray
        Sequence of SNR stacks, one per channel. Each stack is expected to have
        shape ``(n_frames, height, width)``.
    snr_thresh : float
        Detection threshold applied to the SNR values.
    channels_spatial_kernels : sequence of ndarray
        Spatial kernels used to define the local-maxima footprint for each
        channel.
    f0 : int, optional
        Frame offset added to the detected frame indices before converting them
        back to the 1-based convention.
    channels_pixels_nm : float or sequence, optional
        Pixel size in nanometers for each channel. This value is normalized
        through :class:`smlmlp.Config` so that one pixel size pair is available
        for each channel.
    cuda : bool, optional
        Whether to use GPU acceleration.
    parallel : bool, optional
        Whether to enable CPU parallelization.

    Returns
    -------
    tuple
        A tuple ``(fr, x, y, ch, info)`` where:

        - ``fr`` is the concatenated array of detected frame indices,
        - ``x`` is the concatenated array of detected x coordinates in
          nanometers,
        - ``y`` is the concatenated array of detected y coordinates in
          nanometers,
        - ``ch`` is the concatenated array of detected channel indices,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'footprints'``
            Boolean detection footprints derived from the spatial kernels.
        ``'channels_pixels_nm'``
            Normalized per-channel pixel sizes used for coordinate conversion.

    Notes
    -----
    The footprint is obtained by thresholding each spatial kernel using the
    Sparrow-limit-based factor:

    .. math::

        \\exp\\left(-\\frac{(0.47 / 0.21)^2}{2}\\right)

    On CPU, local maxima are first detected on the integer grid and then
    refined with a 5x5 center-of-mass estimate. On GPU, both operations are
    combined in a single kernel.

    Examples
    --------
    >>> import numpy as np
    >>> snr = np.random.rand(5, 16, 16).astype(np.float32)
    >>> kernel = np.ones((3, 3), dtype=np.float32)
    >>> fr, x, y, ch, info = detect_spatial_maxima(
    ...     [snr],
    ...     0.9,
    ...     [kernel],
    ... )
    >>> fr.ndim == x.ndim == y.ndim == ch.ndim == 1
    True
    >>> "footprints" in info
    True

    >>> snrs = [
    ...     np.random.rand(4, 16, 16).astype(np.float32),
    ...     np.random.rand(4, 16, 16).astype(np.float32),
    ... ]
    >>> kernels = [
    ...     np.ones((3, 3), dtype=np.float32),
    ...     np.ones((5, 5), dtype=np.float32),
    ... ]
    >>> fr, x, y, ch, info = detect_spatial_maxima(
    ...     snrs,
    ...     1.2,
    ...     kernels,
    ...     channels_pixels_nm=[(100.0, 100.0), (110.0, 110.0)],
    ... )
    >>> len(info["footprints"])
    2
    """
    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Build boolean detection footprints from the spatial kernels.
    k_fp = math.exp(-(0.47 / 0.21) ** 2 / 2)
    footprints = [
        kernel > kernel.max() * k_fp
        for kernel in channels_spatial_kernels
    ]

    # Normalize the pixel size input so that one pixel size pair is available
    # for each channel.
    channels_pixels_nm = Config(
        ncameras=len(snrs),
        channels_cameras_nm=channels_pixels_nm,
    ).channels_cameras_nm

    F, X, Y, C = [], [], [], []

    for pos, (snr, footprint, pixel) in enumerate(
        zip(snrs, footprints, channels_pixels_nm)
    ):
        snr = xp.asarray(snr)
        footprint = xp.asarray(footprint)

        # GPU implementation: detect maxima and compute the local center of
        # mass directly in the CUDA kernel.
        if cuda:
            max_points = int(snr.size / xp.sum(footprint))
            fr_out = xp.empty(max_points, dtype=xp.int32)
            y_out = xp.empty(max_points, dtype=xp.float32)
            x_out = xp.empty(max_points, dtype=xp.float32)
            counter = xp.zeros(1, dtype=xp.int32)

            shape = snr.shape
            threads_per_block = (2, 16, 16)
            blocks_per_grid = (
                (shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                (shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
                (shape[2] + threads_per_block[2] - 1) // threads_per_block[2],
            )

            det_gpu[blocks_per_grid, threads_per_block](
                snr,
                snr_thresh,
                footprint,
                fr_out,
                y_out,
                x_out,
                counter,
            )

            n = xp.asnumpy(counter)[0]
            fr = xp.asnumpy(fr_out[:n])
            y = xp.asnumpy(y_out[:n])
            x = xp.asnumpy(x_out[:n])

        # CPU implementation: detect integer-grid maxima first, then refine
        # them with a local center of mass.
        else:
            mask = xp.zeros_like(snr, dtype=xp.bool_)

            with nb_threads(parallel):
                maxi_cpu(mask, snr, snr_thresh, footprint)

            fr, y_int, x_int = xp.nonzero(mask)
            y = xp.empty_like(y_int, dtype=xp.float32)
            x = xp.empty_like(x_int, dtype=xp.float32)

            if len(fr):
                with nb_threads(parallel):
                    com_cpu(snr, fr, y_int, x_int, y, x)

        # Convert detections back to the external frame convention and to
        # physical coordinates, then sort them lexicographically.
        fr += f0 + 1
        y *= pixel[0]
        x *= pixel[1]

        c = np.full_like(fr, fill_value=pos + 1, dtype=np.uint8)
        argsort = np.lexsort((x, y, fr))

        F.append(fr[argsort])
        Y.append(y[argsort])
        X.append(x[argsort])
        C.append(c)

    info = {
        "footprints": footprints,
        "channels_pixels_nm": channels_pixels_nm,
    }

    return np.hstack(F), np.hstack(X), np.hstack(Y), np.hstack(C), info



@nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
def maxi_cpu(mask, snr, snr_thresh, footprint):
    """Mark thresholded local maxima on CPU."""
    n_frames, height, width = snr.shape
    fp_height, fp_width = footprint.shape
    cy, cx = fp_height // 2, fp_width // 2

    for fr in nb.prange(n_frames):
        frame = snr[fr]
        frame_mask = mask[fr]

        for y in range(height):
            for x in range(width):
                val = frame[y, x]
                if val < snr_thresh:
                    continue

                is_max = True

                for dy in range(fp_height):
                    for dx in range(fp_width):
                        if not footprint[dy, dx]:
                            continue

                        ny = y + dy - cy
                        nx = x + dx - cx

                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue

                        if frame[ny, nx] > val:
                            is_max = False
                            break

                    if not is_max:
                        break

                if is_max:
                    frame_mask[y, x] = True



@nb.njit(parallel=True, nogil=True, cache=True, fastmath=True)
def com_cpu(snr, fr_idx, y_idx, x_idx, y_out, x_out):
    """Refine detected maxima with a 5x5 center of mass on CPU."""
    n = len(fr_idx)
    _, height, width = snr.shape

    for i in nb.prange(n):
        fr = fr_idx[i]
        y = y_idx[i]
        x = x_idx[i]

        xnum = 0.0
        ynum = 0.0
        denom = 0.0

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny = y + dy
                nx = x + dx

                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue

                val = snr[fr, ny, nx]
                xnum += nx * val
                ynum += ny * val
                denom += val

        if denom > 0:
            x_out[i] = xnum / denom
            y_out[i] = ynum / denom
        else:
            x_out[i] = x
            y_out[i] = y



@nb_cuda.jit(fastmath=True, cache=True)
def det_gpu(snr, snr_thresh, footprint, fr_out, y_out, x_out, counter):
    """Detect and refine local maxima on GPU."""
    fr, y, x = nb_cuda.grid(3)

    n_frames, height, width = snr.shape
    fp_height, fp_width = footprint.shape

    if fr >= n_frames or y >= height or x >= width:
        return

    val = snr[fr, y, x]
    if val < snr_thresh:
        return

    cy = fp_height // 2
    cx = fp_width // 2

    # Local-maximum test inside the footprint.
    is_max = True

    for dy in range(fp_height):
        for dx in range(fp_width):
            if not footprint[dy, dx]:
                continue

            ny = y + dy - cy
            nx = x + dx - cx

            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue

            if snr[fr, ny, nx] > val:
                is_max = False
                break

        if not is_max:
            break

    if not is_max:
        return

    # 5x5 center-of-mass refinement.
    xnum = 0.0
    ynum = 0.0
    denom = 0.0

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue

            v = snr[fr, ny, nx]
            xnum += nx * v
            ynum += ny * v
            denom += v

    if denom > 0:
        xf = xnum / denom
        yf = ynum / denom
    else:
        xf = x
        yf = y

    # Atomically append the detection to the output buffers.
    idx = nb_cuda.atomic.add(counter, 0, 1)

    fr_out[idx] = fr
    y_out[idx] = yf
    x_out[idx] = xf
