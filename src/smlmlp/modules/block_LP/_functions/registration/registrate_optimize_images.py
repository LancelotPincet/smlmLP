#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import img_transform, get_xp, transform_matrix, compress
import numpy as np



@block(timeit=False)
def registrate_optimize_images(
    channels,
    /,
    mode="mean",
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
    mode : {"mean", "std"}, optional
        Projection used to reduce each channel stack before registration.
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
        Optional preallocated output arrays for the transformed images. If
        provided with larger arrays, centered spatial views are reused when
        possible.
    channels_pixels_nm : float or sequence, optional
        Pixel size in nanometers. Can be scalar, ``(y, x)``, or per-channel.
    cuda : bool, optional
        Whether to enable CUDA processing.
    parallel : bool, optional
        Whether to enable parallel processing.

    Returns
    -------
    tuple
        A tuple ``(new_optimized, info)`` where:

        - ``new_optimized`` is the list of transformed and compressed images,
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
        ``'crop_shape'``
            Common centered crop shape applied after transformation.
        ``'crop_bboxes'``
            Per-channel centered crop boxes as ``(x0, y0, x1, y1)``.

    Examples
    --------
    >>> import numpy as np
    >>> channels = [np.random.rand(5, 16, 16).astype(np.float32)]
    >>> optimized, info = registrate_optimize_images(
    ...     channels,
    ...     channels_x_shifts_nm=[0.0],
    ...     channels_y_shifts_nm=[0.0],
    ...     channels_rotations_deg=[0.0],
    ...     channels_x_shears=[0.0],
    ...     channels_y_shears=[0.0],
    ... )
    >>> len(optimized)
    1
    >>> info["ref_pix"]
    (1.0, 1.0)

    >>> channels = [
    ...     np.random.rand(5, 16, 16).astype(np.float32),
    ...     np.random.rand(5, 16, 16).astype(np.float32),
    ... ]
    >>> optimized, info = registrate_optimize_images(
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
    valid_masks = []

    match mode:
        case "mean":
            agg_func = xp.mean
        case "std":
            agg_func = xp.std
        case _:
            raise SyntaxError(f"Aggregation mode {mode} is not recognized")

    for i in range(len(channels)):
        channel = xp.asarray(channels[i])
        projection = agg_func(channel, axis=0, dtype=np.float32)

        # Build the rescaling and registration transforms, then combine them.
        matrix1 = transform_matrix(
            projection,
            scalex=scales_x[i],
            scaley=scales_y[i],
        )
        matrix2 = transform_matrix(
            projection,
            shiftx=channels_x_shifts_nm[i] / ref_pix[1],
            shifty=channels_y_shifts_nm[i] / ref_pix[0],
            angle=channels_rotations_deg[i],
            shearx=channels_x_shears[i],
            sheary=channels_y_shears[i],
        )
        matrix = matrix1 @ matrix2

        # Apply the geometric transform and keep a transformed validity mask so
        # the final centered crop can avoid affine border pixels.
        optimize = img_transform(
            projection,
            matrix=matrix,
            cuda=cuda,
            parallel=False,
        )
        new_optimized.append(optimize)
        valid_masks.append(
            _transform_valid_mask(
                projection.shape,
                matrix,
                cuda=cuda,
            )
        )

    # Crop transformed images around their centers so registration images share
    # the same valid spatial support after rescaling to the reference pixel size.
    crop_shape = _largest_centered_valid_crop(valid_masks)
    new_optimized, crop_shape, crop_bboxes = _center_crop_arrays(
        new_optimized,
        crop_shape,
        outputs=optimized,
    )

    for i, optimize in enumerate(new_optimized):
        new_optimized[i] = compress(
            optimize,
            out=optimize,
            white_percent=1,
            black_percent=1,
            saturate=True,
        )

    info = {
        "channels_pixels_nm": channels_pixels_nm,
        "ref_pix": ref_pix,
        "scales_x": scales_x,
        "scales_y": scales_y,
        "crop_shape": crop_shape,
        "crop_bboxes": crop_bboxes,
    }

    return new_optimized, info



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



def _center_crop_arrays(arrays, crop_shape, outputs=None):
    """Center-crop arrays to a shared spatial shape."""

    cropped = []
    crop_bboxes = []
    for i, array in enumerate(arrays):
        y0 = (array.shape[0] - crop_shape[0]) // 2
        x0 = (array.shape[1] - crop_shape[1]) // 2
        y1 = y0 + crop_shape[0]
        x1 = x0 + crop_shape[1]

        crop = array[y0:y1, x0:x1]
        crop = _copy_to_output(crop, outputs, i, name="optimized")
        cropped.append(crop)
        crop_bboxes.append((x0, y0, x1, y1))

    return cropped, crop_shape, crop_bboxes



def _copy_to_output(array, outputs, index, name="outputs"):
    """Copy an array into a reusable output buffer when one is available."""
    if outputs is None:
        return array
    if len(outputs) <= index:
        raise ValueError(f"{name} does not have enough output arrays")

    out = _output_view(outputs[index], array.shape)
    if out is None:
        return array

    out[...] = array
    return out



def _output_view(output, shape):
    """Return a compatible centered view into an output buffer, if possible."""
    if output.shape == shape:
        return output
    if output.ndim != len(shape):
        return None
    if any(out_size < size for out_size, size in zip(output.shape, shape)):
        return None

    slices = []
    for out_size, size in zip(output.shape, shape):
        start = (out_size - size) // 2
        slices.append(slice(start, start + size))

    return output[tuple(slices)]
