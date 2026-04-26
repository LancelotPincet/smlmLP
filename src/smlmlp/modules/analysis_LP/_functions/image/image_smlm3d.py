#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


import math

import numpy as np

from smlmlp import analysis
from smlmlp.modules.analysis_LP._functions.image.image_smlm import (
    _as_float_array,
    _auto_padding_nm,
    _prepare_xy,
    _resolve_pixel,
    _resolve_render_geometry,
    _render_gaussians_stack_njit,
)


@analysis(df_name="blinks")
def image_smlm3d(
    col,
    /,
    xx,
    yy,
    crlb=None,
    weight=1.0,
    x_shape=None,
    y_shape=None,
    shape=None,
    *,
    image_sigma=None,
    pixel_sr_nm=15.0,
    z_pixel=None,
    col_pixel=None,
    z_bins=None,
    z_min=None,
    z_max=None,
    x0=None,
    y0=None,
    crop_sigma=4.0,
    max_radius_pixels=0,
    cuda=False,
    parallel=False,
):
    """
    Render localizations into a 3D SMLM volume using ``col`` as z coordinate.

    Each localization contributes a 2D integrated Gaussian in the z plane
    selected from ``col``. This keeps the expensive Gaussian integration in xy
    while avoiding artificial blur along arbitrary metadata dimensions.

    Parameters
    ----------
    col : array-like
        Values defining the third dimension.
    xx, yy : array-like
        Localization coordinates in nm.
    crlb, image_sigma : float or array-like, optional
        Gaussian sigma in nm. ``image_sigma`` takes precedence over ``crlb``.
    weight : float or array-like, optional
        Integrated weight of each localization.
    z_pixel, col_pixel : float, optional
        Bin size for the third dimension. Defaults to the render pixel size.
    z_bins : int or array-like, optional
        Explicit number of z bins or bin edges. If provided, overrides
        ``z_pixel`` for z indexing.
    z_min, z_max : float, optional
        z range limits. Defaults to finite extrema of ``col``.

    Returns
    -------
    volume : ndarray
        ``float32`` array with shape ``(z, y, x)``.
    info : dict
        Rendering and z-bin metadata.
    """

    del cuda, parallel

    xx, yy = _prepare_xy(xx, yy)
    n = len(xx)
    z_values = np.ascontiguousarray(np.asarray(col, dtype=np.float32).ravel())
    if len(z_values) != n:
        raise ValueError(f"col must have length {n}.")

    pixel = _resolve_pixel(pixel_sr_nm)
    sigma_source = crlb if image_sigma is None else image_sigma
    sigma = _as_float_array(sigma_source, n, "image_sigma", default=pixel)
    weight = _as_float_array(weight, n, "weight", default=1.0)

    z_index, z_info = _resolve_z_index(
        z_values,
        pixel,
        z_pixel=z_pixel,
        col_pixel=col_pixel,
        z_bins=z_bins,
        z_min=z_min,
        z_max=z_max,
    )

    padding_nm = _auto_padding_nm(sigma, pixel, crop_sigma)
    y_size, x_size, x0, y0 = _resolve_render_geometry(
        xx,
        yy,
        pixel,
        shape=shape,
        x_shape=x_shape,
        y_shape=y_shape,
        x0=x0,
        y0=y0,
        padding_nm=padding_nm,
    )

    volume = np.zeros((z_info["z_shape"], y_size, x_size), dtype=np.float32)
    _render_gaussians_stack_njit(
        xx,
        yy,
        z_index,
        sigma,
        weight,
        volume,
        float(pixel),
        float(x0),
        float(y0),
        float(crop_sigma),
        int(max_radius_pixels),
    )

    info = {
        "pixel_sr_nm": float(pixel),
        "x0": float(x0),
        "y0": float(y0),
        "crop_sigma": float(crop_sigma),
        **z_info,
    }
    return volume, info


def _resolve_z_index(
    z_values,
    xy_pixel,
    *,
    z_pixel=None,
    col_pixel=None,
    z_bins=None,
    z_min=None,
    z_max=None,
):
    """Return z plane indices and metadata."""
    finite = np.isfinite(z_values)
    if not np.any(finite):
        return np.full(len(z_values), -1, dtype=np.int64), {
            "z_shape": 1,
            "z_min": np.nan,
            "z_max": np.nan,
            "z_pixel": float(xy_pixel),
        }

    if z_bins is not None:
        return _resolve_z_index_from_bins(z_values, finite, z_bins, z_min, z_max)

    z_pixel = col_pixel if z_pixel is None and col_pixel is not None else z_pixel
    z_pixel = float(xy_pixel if z_pixel is None else z_pixel)
    if not np.isfinite(z_pixel) or z_pixel <= 0:
        raise ValueError("z_pixel must be a finite positive value.")

    if z_min is None:
        z_min = math.floor(float(np.nanmin(z_values[finite])) / z_pixel) * z_pixel
    else:
        z_min = float(z_min)
    z_max = float(np.nanmax(z_values[finite]) if z_max is None else z_max)
    z_shape = int(math.floor((z_max - z_min) / z_pixel)) + 1
    z_shape = max(z_shape, 1)

    z_index = np.full(len(z_values), -1, dtype=np.int64)
    z_index[finite] = np.floor((z_values[finite] - z_min) / z_pixel).astype(np.int64)
    z_index[(z_index < 0) | (z_index >= z_shape)] = -1

    return np.ascontiguousarray(z_index), {
        "z_shape": int(z_shape),
        "z_min": float(z_min),
        "z_max": float(z_min + (z_shape - 1) * z_pixel),
        "z_pixel": float(z_pixel),
    }


def _resolve_z_index_from_bins(z_values, finite, z_bins, z_min, z_max):
    """Return z plane indices from explicit bin count or edges."""
    if np.ndim(z_bins) == 0:
        z_shape = int(z_bins)
        if z_shape <= 0:
            raise ValueError("z_bins must be positive.")
        z_min = float(np.nanmin(z_values[finite]) if z_min is None else z_min)
        z_max = float(np.nanmax(z_values[finite]) if z_max is None else z_max)
        if z_min == z_max:
            z_max = z_min + 1.0
        edges = np.linspace(z_min, z_max, z_shape + 1, dtype=np.float64)
    else:
        edges = np.asarray(z_bins, dtype=np.float64).ravel()
        if len(edges) < 2:
            raise ValueError("z_bins edges must contain at least two values.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("z_bins edges must be strictly increasing.")
        z_shape = len(edges) - 1

    z_index = np.full(len(z_values), -1, dtype=np.int64)
    idx = np.searchsorted(edges, z_values[finite], side="right") - 1
    idx[z_values[finite] == edges[-1]] = z_shape - 1
    z_index[finite] = idx.astype(np.int64)
    z_index[(z_index < 0) | (z_index >= z_shape)] = -1

    return np.ascontiguousarray(z_index), {
        "z_shape": int(z_shape),
        "z_min": float(edges[0]),
        "z_max": float(edges[-1]),
        "z_edges": edges,
    }
