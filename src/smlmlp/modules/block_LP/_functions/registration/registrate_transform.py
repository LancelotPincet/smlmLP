#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from arrlp import img_transform, transform_matrix
import numpy as np



@block(timeit=False)
def registrate_transform(
    channels, /,
    channels_x_shifts_nm=None, channels_y_shifts_nm=None,
    channels_rotations_deg=None, channels_x_shears=None, channels_y_shears=None,
    *,
    channels_pixels_nm=1.0,
    cuda=False, parallel=False,
):
    """Apply per-channel affine transforms in place to channel stacks."""
    # Normalize per-channel pixel sizes to (y, x) tuples.
    channels_pixels_nm = _normalize_channels_pixels_nm(channels_pixels_nm, len(channels))

    # Normalize per-channel transformation parameters.
    channels_x_shifts_nm = _normalize_channels_parameter(channels_x_shifts_nm, len(channels), "channels_x_shifts_nm")
    channels_y_shifts_nm = _normalize_channels_parameter(channels_y_shifts_nm, len(channels), "channels_y_shifts_nm")
    channels_rotations_deg = _normalize_channels_parameter(channels_rotations_deg, len(channels), "channels_rotations_deg")
    channels_x_shears = _normalize_channels_parameter(channels_x_shears, len(channels), "channels_x_shears")
    channels_y_shears = _normalize_channels_parameter(channels_y_shears, len(channels), "channels_y_shears")

    # Build one transform matrix per channel and apply each transform in place.
    transformed = []
    for i, channel in enumerate(channels):
        matrix = transform_matrix(
            channel,
            shiftx=channels_x_shifts_nm[i] / channels_pixels_nm[i][1],
            shifty=channels_y_shifts_nm[i] / channels_pixels_nm[i][0],
            angle=channels_rotations_deg[i],
            shearx=channels_x_shears[i],
            sheary=channels_y_shears[i],
        )
        transformed_channel = img_transform(
            channel,
            matrix=matrix,
            stacks=True,
            out=channel,
            cuda=cuda,
            parallel=parallel,
        )
        transformed.append(transformed_channel)

    return transformed



def _normalize_channels_pixels_nm(channels_pixels_nm, n_channels):
    """Normalize pixel sizes to one (y, x) tuple per channel."""
    try:
        if len(channels_pixels_nm) != n_channels:
            if len(channels_pixels_nm) == 2:
                channels_pixels_nm = [channels_pixels_nm for _ in range(n_channels)]
            else:
                raise ValueError("channels_pixels_nm does not have the same length as channels")
    except TypeError:
        channels_pixels_nm = [(channels_pixels_nm, channels_pixels_nm) for _ in range(n_channels)]

    return channels_pixels_nm



def _normalize_channels_parameter(values, n_channels, name):
    """Normalize one per-channel transform parameter to float32 values."""
    if values is None:
        return np.zeros(n_channels, dtype=np.float32)

    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        return np.full(n_channels, values.item(), dtype=np.float32)
    if len(values) != n_channels:
        raise ValueError(f"{name} does not have the same length as channels")

    return values
