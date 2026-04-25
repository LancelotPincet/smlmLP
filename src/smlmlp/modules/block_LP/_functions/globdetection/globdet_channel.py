#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

from smlmlp import block
from arrlp import get_xp, img_transform, transform_matrix
import numpy as np



@block()
def globdet_channels(
    channels,
    /,
    mode="mean",
    channels_x_shifts_nm=None,
    channels_y_shifts_nm=None,
    channels_rotations_deg=None,
    channels_x_shears=None,
    channels_y_shears=None,
    global_channels=None,
    *,
    channels_pixels_nm=1.0,
    cuda=False,
    parallel=False,
):
    """
    Create a global channel for detection.

    This function geometrically transforms each input channel stack into the
    global detection frame, then merges all transformed channels into a single
    channel stack using either a mean or standard deviation projection across
    channels.

    Parameters
    ----------
    channels : sequence of ndarray
        Input channel stacks.
    mode : {"mean", "std"}, optional
        Aggregation used to merge transformed channels.
    channels_x_shifts_nm, channels_y_shifts_nm : sequence of float or None, optional
        Per-channel translations in nanometers. If ``None``, zeros are used.
    channels_rotations_deg : sequence of float or None, optional
        Per-channel rotations, in degrees. If ``None``, zeros are used.
    channels_x_shears, channels_y_shears : sequence of float or None, optional
        Per-channel shear values. If ``None``, zeros are used.
    global_channels : sequence of ndarray or None, optional
        Optional preallocated output list for the merged global detection
        channel. If provided with larger arrays, centered spatial views are
        reused when possible.
    channels_pixels_nm : float or sequence, optional
        Pixel size in nanometers. Can be scalar, ``(y, x)``, or per-channel.
    cuda : bool, optional
        Whether to use CUDA execution.
    parallel : bool, optional
        Whether to use parallel execution.

    Returns
    -------
    tuple
        A tuple ``(new_channels, info)`` where:

        - ``new_channels`` is a one-element list containing the merged global
          detection channel,
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
        ``'channels_x_shifts_nm'``
            Normalized per-channel x translations.
        ``'channels_y_shifts_nm'``
            Normalized per-channel y translations.
        ``'channels_rotations_deg'``
            Normalized per-channel rotations.
        ``'channels_x_shears'``
            Normalized per-channel x shears.
        ``'channels_y_shears'``
            Normalized per-channel y shears.
        ``'transform_matrices'``
            Transform matrices applied to each input channel.
        ``'crop_shape'``
            Common centered crop shape applied after transformation.
        ``'crop_bboxes'``
            Per-channel centered crop boxes as ``(x0, y0, x1, y1)``.

    Raises
    ------
    ValueError
        If no channel is provided or if a per-channel parameter length does not
        match ``len(channels)``.
    SyntaxError
        If ``mode`` is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> channels = [
    ...     np.ones((2, 4, 4), dtype=np.float32),
    ...     np.ones((2, 4, 4), dtype=np.float32) * 3,
    ... ]
    >>> global_channels, info = globdet_channels(channels)
    >>> len(global_channels)
    1
    >>> global_channels[0].shape
    (2, 4, 4)

    >>> global_channels, info = globdet_channels(channels, mode="std")
    >>> np.allclose(global_channels[0], 1.0)
    True
    """
    if len(channels) == 0:
        raise ValueError("channels must contain at least one channel")

    # Select the array backend matching the requested execution mode.
    xp = get_xp(cuda)

    # Normalize the per-channel pixel sizes so shifts can be converted from
    # nanometers to pixels before building image-space transforms.
    channels_pixels_nm = _normalize_channels_pixels_nm(
        channels_pixels_nm,
        len(channels),
    )
    ref_pix = (
        min([pix[0] for pix in channels_pixels_nm]),
        min([pix[1] for pix in channels_pixels_nm]),
    )
    scales_x = [ref_pix[1] / pix[1] for pix in channels_pixels_nm]
    scales_y = [ref_pix[0] / pix[0] for pix in channels_pixels_nm]

    # Initialize and normalize registration parameters to one value per channel.
    channels_x_shifts_nm = _normalize_channels_parameter(
        np.zeros(len(channels), dtype=np.float32)
        if channels_x_shifts_nm is None
        else channels_x_shifts_nm,
        len(channels),
        "channels_x_shifts_nm",
    )
    channels_y_shifts_nm = _normalize_channels_parameter(
        np.zeros(len(channels), dtype=np.float32)
        if channels_y_shifts_nm is None
        else channels_y_shifts_nm,
        len(channels),
        "channels_y_shifts_nm",
    )
    channels_rotations_deg = _normalize_channels_parameter(
        np.zeros(len(channels), dtype=np.float32)
        if channels_rotations_deg is None
        else channels_rotations_deg,
        len(channels),
        "channels_rotations_deg",
    )
    channels_x_shears = _normalize_channels_parameter(
        np.zeros(len(channels), dtype=np.float32)
        if channels_x_shears is None
        else channels_x_shears,
        len(channels),
        "channels_x_shears",
    )
    channels_y_shears = _normalize_channels_parameter(
        np.zeros(len(channels), dtype=np.float32)
        if channels_y_shears is None
        else channels_y_shears,
        len(channels),
        "channels_y_shears",
    )

    match mode:
        case "mean":
            agg_func = xp.mean
        case "std":
            agg_func = xp.std
        case _:
            raise SyntaxError(f"Aggregation mode {mode} is not recognized")

    transformed = []
    valid_masks = []
    matrices = []

    # Transform each channel stack into the global detection frame.
    for i in range(len(channels)):
        channel = xp.asarray(channels[i])
        matrix1 = transform_matrix(
            channel,
            scalex=scales_x[i],
            scaley=scales_y[i],
            stacks=True,
        )
        matrix2 = transform_matrix(
            channel,
            shiftx=channels_x_shifts_nm[i] / ref_pix[1],
            shifty=channels_y_shifts_nm[i] / ref_pix[0],
            angle=channels_rotations_deg[i],
            shearx=channels_x_shears[i],
            sheary=channels_y_shears[i],
            stacks=True,
        )
        matrix = matrix1 @ matrix2
        transformed.append(
            img_transform(
                channel,
                matrix=matrix,
                cuda=cuda,
                parallel=parallel,
                stacks=True,
            )
        )
        valid_masks.append(
            _transform_valid_mask(
                channel.shape[1:3],
                matrix,
                cuda=cuda,
            )
        )
        matrices.append(matrix)

    # Merge channels into the single global detection channel requested by the
    # downstream detection pipeline.
    crop_shape = _largest_centered_valid_crop(valid_masks)
    transformed, crop_shape, crop_bboxes = _center_crop_arrays(
        transformed,
        crop_shape,
        stacks=True,
    )
    transformed = xp.stack(transformed, axis=0)
    merged = agg_func(transformed, axis=0, dtype=xp.float32)
    new_channels = [
        _copy_to_output(
            merged,
            global_channels,
            0,
            stacks=True,
            name="global_channels",
        )
    ]

    info = {
        "channels_pixels_nm": channels_pixels_nm,
        "ref_pix": ref_pix,
        "scales_x": scales_x,
        "scales_y": scales_y,
        "channels_x_shifts_nm": channels_x_shifts_nm,
        "channels_y_shifts_nm": channels_y_shifts_nm,
        "channels_rotations_deg": channels_rotations_deg,
        "channels_x_shears": channels_x_shears,
        "channels_y_shears": channels_y_shears,
        "transform_matrices": matrices,
        "crop_shape": crop_shape,
        "crop_bboxes": crop_bboxes,
    }

    return new_channels, info



def _normalize_channels_pixels_nm(channels_pixels_nm, n_channels):
    """Normalize pixel sizes to one ``(py, px)`` tuple per channel."""
    try:
        n_pixels = len(channels_pixels_nm)
    except TypeError:
        channels_pixels_nm = [
            (channels_pixels_nm, channels_pixels_nm)
            for _ in range(n_channels)
        ]
    else:
        try:
            len(channels_pixels_nm[0])
        except TypeError:
            if n_pixels == 2:
                channels_pixels_nm = [
                    channels_pixels_nm
                    for _ in range(n_channels)
                ]
            elif n_pixels == n_channels:
                channels_pixels_nm = [
                    (pix, pix)
                    for pix in channels_pixels_nm
                ]
            else:
                raise ValueError(
                    "channels_pixels_nm does not have the same length as channels"
                )
        else:
            if n_pixels != n_channels:
                raise ValueError(
                    "channels_pixels_nm does not have the same length as channels"
                )

    return channels_pixels_nm



def _normalize_channels_parameter(values, n_channels, name):
    """Normalize scalar/per-channel values to a per-channel sequence."""
    try:
        if len(values) != n_channels:
            raise ValueError(f"{name} does not have the same length as channels")
    except TypeError:
        values = [values for _ in range(n_channels)]

    return values



def _transform_valid_mask(shape, matrix, cuda=False):
    """Transform a mask that tracks pixels not filled by affine borders."""
    xp = get_xp(cuda)
    mask = xp.ones(shape, dtype=xp.float32)
    mask = img_transform(
        mask,
        matrix=matrix,
        cuda=cuda,
        parallel=False,
        order=0,
        cval=0.0,
    )

    return mask > 0.5



def _largest_centered_valid_crop(valid_masks):
    """Find the largest centered crop valid in every mask."""
    base_shape = np.asarray(
        (
            min([mask.shape[0] for mask in valid_masks]),
            min([mask.shape[1] for mask in valid_masks]),
        ),
        dtype=int,
    )

    lo, hi = 1, int(base_shape.min())
    best_shape = None

    while lo <= hi:
        mid = (lo + hi) // 2
        scale = mid / base_shape.min()
        crop_shape = np.maximum(1, np.floor(base_shape * scale).astype(int))

        if _valid_centered_crop(valid_masks, crop_shape):
            best_shape = tuple(int(size) for size in crop_shape)
            lo = mid + 1
        else:
            hi = mid - 1

    if best_shape is None:
        raise ValueError("No centered crop without transformed border pixels found")

    return best_shape



def _valid_centered_crop(valid_masks, crop_shape):
    """Return whether the centered crop is fully valid in every mask."""
    for mask in valid_masks:
        y0 = (mask.shape[0] - crop_shape[0]) // 2
        x0 = (mask.shape[1] - crop_shape[1]) // 2
        y1 = y0 + crop_shape[0]
        x1 = x0 + crop_shape[1]

        valid = mask[y0:y1, x0:x1].all()
        if hasattr(valid, "item"):
            valid = valid.item()
        if not valid:
            return False

    return True



def _center_crop_arrays(arrays, crop_shape, stacks=False):
    """Center-crop arrays to a shared spatial shape."""
    spatial_axis = int(stacks)

    cropped = []
    crop_bboxes = []
    for array in arrays:
        y0 = (array.shape[spatial_axis] - crop_shape[0]) // 2
        x0 = (array.shape[spatial_axis + 1] - crop_shape[1]) // 2
        y1 = y0 + crop_shape[0]
        x1 = x0 + crop_shape[1]

        slices = [slice(None) for _ in range(array.ndim)]
        slices[spatial_axis] = slice(y0, y1)
        slices[spatial_axis + 1] = slice(x0, x1)

        cropped.append(array[tuple(slices)])
        crop_bboxes.append((x0, y0, x1, y1))

    return cropped, crop_shape, crop_bboxes



def _copy_to_output(array, outputs, index, stacks=False, name="outputs"):
    """Copy an array into a reusable output buffer when one is available."""
    if outputs is None:
        return array
    if len(outputs) <= index:
        raise ValueError(f"{name} does not have enough output arrays")

    out = _output_view(outputs[index], array.shape, stacks=stacks)
    if out is None:
        return array

    out[...] = array
    return out



def _output_view(output, shape, stacks=False):
    """Return a compatible view into an output buffer, if possible."""
    if output.shape == shape:
        return output
    if output.ndim != len(shape):
        return None
    if any(out_size < size for out_size, size in zip(output.shape, shape)):
        return None

    slices = []
    for axis, (out_size, size) in enumerate(zip(output.shape, shape)):
        start = 0 if stacks and axis == 0 else (out_size - size) // 2
        slices.append(slice(start, start + size))

    return output[tuple(slices)]
