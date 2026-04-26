#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


import math

import numba as nb
import numpy as np

from smlmlp import analysis


@analysis(df_name="blinks")
def image_smlm(
    xx,
    yy,
    image_sigma=None,
    weight=1.0,
    x_shape=None,
    y_shape=None,
    shape=None,
    *,
    pixel_sr_nm=15.0,
    x0=None,
    y0=None,
    crop_sigma=4.0,
    max_radius_pixels=0,
    cuda=False,
    parallel=False,
):
    """
    Render localizations as an integrated Gaussian SMLM image.

    Parameters
    ----------
    xx, yy : array-like
        Localization coordinates in nm.
    image_sigma : float or array-like, optional
        Gaussian sigma in nm. Missing values default to the render pixel size.
    weight : float or array-like, optional
        Integrated weight of each localization. Default is one count per point.
    x_shape, y_shape, shape : int or tuple, optional
        Output image dimensions. ``shape`` is ``(y, x)`` and takes precedence.
        Without a shape, the image is tightly fitted around finite coordinates.
    pixel_sr_nm : float, optional
        Render pixel size in nm. 
    x0, y0 : float, optional
        Coordinate of the pixel-center origin in nm. Defaults to zero when an
        explicit shape is supplied, otherwise to the fitted lower bound.
    crop_sigma : float, optional
        Gaussian support radius in sigma units.
    max_radius_pixels : int, optional
        Optional cap for very large Gaussian radii. Zero disables the cap.
    cuda, parallel : bool, optional
        Accepted for analysis API consistency. CPU rendering is deterministic;
        overlapping Gaussian writes are intentionally not parallelized.

    Returns
    -------
    image : ndarray
        ``float32`` image with shape ``(y, x)``.
    info : dict
        Rendering metadata.
    """

    del cuda, parallel

    xx, yy = _prepare_xy(xx, yy)
    pixel = _resolve_pixel(pixel_sr_nm)
    n = len(xx)

    sigma = _as_float_array(image_sigma, n, "image_sigma", default=pixel)
    weight = _as_float_array(weight, n, "weight", default=1.0)

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

    image = np.zeros((y_size, x_size), dtype=np.float32)
    _render_gaussians_njit(
        xx,
        yy,
        sigma,
        weight,
        image,
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
    }
    return image, info


def _prepare_xy(xx, yy):
    """Return finite-ready x and y arrays with matching lengths."""
    xx = np.ascontiguousarray(np.asarray(xx, dtype=np.float32).ravel())
    yy = np.ascontiguousarray(np.asarray(yy, dtype=np.float32).ravel())
    if len(xx) != len(yy):
        raise ValueError("xx and yy must have the same length.")
    return xx, yy


def _resolve_pixel(pixel_sr_nm):
    """Validate and return render pixel size."""
    pixel = float(pixel_sr_nm)
    if not np.isfinite(pixel) or pixel <= 0:
        raise ValueError("pixel_sr_nm must be a finite positive value.")
    return pixel





def _as_float_array(value, n, name, default=None):
    """Broadcast a scalar or validate a one-dimensional float array."""
    if value is None:
        value = default
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float32)
    arr = np.ascontiguousarray(arr.ravel())
    if len(arr) != n:
        raise ValueError(f"{name} must be scalar or have length {n}.")
    return arr


def _shape_value(value):
    """Return an integer image shape value from a scalar or repeated column."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    size = int(np.nanmax(arr))
    if size <= 0:
        raise ValueError("Image shape values must be positive.")
    return size


def _resolve_shape(shape, x_shape, y_shape):
    """Resolve optional image shape as ``(y, x)``."""
    if shape is not None:
        shape_arr = np.asarray(shape).ravel()
        if len(shape_arr) != 2:
            raise ValueError("shape must contain two values: (y, x).")
        y_size, x_size = int(shape_arr[0]), int(shape_arr[1])
        if y_size <= 0 or x_size <= 0:
            raise ValueError("shape values must be positive.")
        return y_size, x_size

    x_size = _shape_value(x_shape)
    y_size = _shape_value(y_shape)
    if (x_size is None) != (y_size is None):
        raise ValueError("x_shape and y_shape must be provided together.")
    if x_size is None:
        return None, None
    return y_size, x_size


def _auto_padding_nm(sigma, pixel, crop_sigma):
    """Return enough automatic padding to keep fitted images from clipping."""
    finite = sigma[np.isfinite(sigma) & (sigma > 0)]
    if len(finite) == 0:
        return pixel
    return max(float(np.nanmax(finite)) * float(crop_sigma), pixel)


def _resolve_render_geometry(
    xx,
    yy,
    pixel,
    *,
    shape=None,
    x_shape=None,
    y_shape=None,
    x0=None,
    y0=None,
    padding_nm=0.0,
):
    """Resolve image geometry and origin for coordinate rendering."""
    y_size, x_size = _resolve_shape(shape, x_shape, y_shape)
    explicit_shape = y_size is not None

    finite = np.isfinite(xx) & np.isfinite(yy)
    if not np.any(finite):
        if y_size is None:
            y_size, x_size = 1, 1
        if x0 is None:
            x0 = 0.0
        if y0 is None:
            y0 = 0.0
        return y_size, x_size, float(x0), float(y0)

    if x0 is None:
        if explicit_shape:
            x0 = 0.0
        else:
            x0 = math.floor(float(np.nanmin(xx[finite]) - padding_nm) / pixel) * pixel
    if y0 is None:
        if explicit_shape:
            y0 = 0.0
        else:
            y0 = math.floor(float(np.nanmin(yy[finite]) - padding_nm) / pixel) * pixel

    if x_size is None:
        x_max = float(np.nanmax(xx[finite]) + padding_nm)
        x_size = int(math.ceil((x_max - float(x0)) / pixel)) + 1
    if y_size is None:
        y_max = float(np.nanmax(yy[finite]) + padding_nm)
        y_size = int(math.ceil((y_max - float(y0)) / pixel)) + 1

    return max(y_size, 1), max(x_size, 1), float(x0), float(y0)


@nb.njit(cache=True)
def _render_gaussians_njit(
    xx,
    yy,
    sigma,
    weight,
    image,
    pixel,
    x0,
    y0,
    crop_sigma,
    max_radius_pixels,
):
    """Rasterize integrated 2D Gaussians into an image."""
    sqrt2 = math.sqrt(2.0)
    half_pixel = pixel * 0.5
    y_size, x_size = image.shape

    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]
        sig = sigma[i]
        w = weight[i]

        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if not math.isfinite(w) or w == 0.0:
            continue

        x_pix = (x - x0) / pixel
        y_pix = (y - y0) / pixel
        x_center = int(math.floor(x_pix + 0.5))
        y_center = int(math.floor(y_pix + 0.5))

        if not math.isfinite(sig) or sig <= 0.0:
            if 0 <= x_center < x_size and 0 <= y_center < y_size:
                image[y_center, x_center] += w
            continue

        crop = int(math.ceil(crop_sigma * sig / pixel + 0.5))
        if max_radius_pixels > 0 and crop > max_radius_pixels:
            crop = max_radius_pixels

        norm = sqrt2 * sig
        for iy in range(y_center - crop, y_center + crop + 1):
            if iy < 0 or iy >= y_size:
                continue
            py = y0 + iy * pixel
            gy_min = (py - y - half_pixel) / norm
            gy_max = (py - y + half_pixel) / norm
            gy = 0.5 * (math.erf(gy_max) - math.erf(gy_min))
            if gy == 0.0:
                continue

            for ix in range(x_center - crop, x_center + crop + 1):
                if ix < 0 or ix >= x_size:
                    continue
                px = x0 + ix * pixel
                gx_min = (px - x - half_pixel) / norm
                gx_max = (px - x + half_pixel) / norm
                gx = 0.5 * (math.erf(gx_max) - math.erf(gx_min))
                image[iy, ix] += w * gx * gy

    return image


@nb.njit(cache=True)
def _render_gaussians_stack_njit(
    xx,
    yy,
    z_index,
    sigma,
    weight,
    volume,
    pixel,
    x0,
    y0,
    crop_sigma,
    max_radius_pixels,
):
    """Rasterize integrated 2D Gaussians into indexed z planes."""
    sqrt2 = math.sqrt(2.0)
    half_pixel = pixel * 0.5
    z_size, y_size, x_size = volume.shape

    for i in range(len(xx)):
        z = z_index[i]
        if z < 0 or z >= z_size:
            continue

        x = xx[i]
        y = yy[i]
        sig = sigma[i]
        w = weight[i]

        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if not math.isfinite(w) or w == 0.0:
            continue

        x_pix = (x - x0) / pixel
        y_pix = (y - y0) / pixel
        x_center = int(math.floor(x_pix + 0.5))
        y_center = int(math.floor(y_pix + 0.5))

        if not math.isfinite(sig) or sig <= 0.0:
            if 0 <= x_center < x_size and 0 <= y_center < y_size:
                volume[z, y_center, x_center] += w
            continue

        crop = int(math.ceil(crop_sigma * sig / pixel + 0.5))
        if max_radius_pixels > 0 and crop > max_radius_pixels:
            crop = max_radius_pixels

        norm = sqrt2 * sig
        for iy in range(y_center - crop, y_center + crop + 1):
            if iy < 0 or iy >= y_size:
                continue
            py = y0 + iy * pixel
            gy_min = (py - y - half_pixel) / norm
            gy_max = (py - y + half_pixel) / norm
            gy = 0.5 * (math.erf(gy_max) - math.erf(gy_min))
            if gy == 0.0:
                continue

            for ix in range(x_center - crop, x_center + crop + 1):
                if ix < 0 or ix >= x_size:
                    continue
                px = x0 + ix * pixel
                gx_min = (px - x - half_pixel) / norm
                gx_max = (px - x + half_pixel) / norm
                gx = 0.5 * (math.erf(gx_max) - math.erf(gx_min))
                volume[z, iy, ix] += w * gx * gy

    return volume
