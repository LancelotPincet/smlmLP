#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


import math

import numba as nb
import numpy as np

from smlmlp import analysis
from smlmlp.modules.analysis_LP._functions.image.image_smlm import (
    _as_float_array,
    _prepare_xy,
    _resolve_pixel,
    _resolve_render_geometry,
)


@analysis(df_name="pixels")
def image_vectors(
    col,
    /,
    xx,
    yy,
    azimuth,
    x_shape=None,
    y_shape=None,
    shape=None,
    *,
    pixel_sr_nm=15.0,
    line_length_nm=100.0,
    line_width_nm=None,
    color_limits=None,
    azimuth_unit="deg",
    x0=None,
    y0=None,
    cuda=False,
    parallel=False,
):
    """
    Render one colored orientation segment per localization.

    Parameters
    ----------
    col : array-like
        Values encoded as hue, from blue at the lower color limit to red at the
        upper color limit.
    xx, yy : array-like
        Segment centers in nm.
    azimuth : array-like
        Segment angles. Degrees are expected by default to match dataframe
        metadata.
    line_length_nm, line_width_nm : float, optional
        Segment length and thickness in nm.
    color_limits : tuple, optional
        ``(min, max)`` used for color normalization. Finite data extrema are
        used by default.

    Returns
    -------
    image : ndarray
        RGB ``float32`` image with shape ``(y, x, 3)`` and values in ``[0, 1]``.
    info : dict
        Rendering metadata.
    """

    del cuda, parallel

    xx, yy = _prepare_xy(xx, yy)
    n = len(xx)
    values = _as_float_array(col, n, "col")
    azimuth = _as_float_array(azimuth, n, "azimuth")

    pixel = _resolve_pixel(pixel_sr_nm)
    line_length_nm = float(line_length_nm)
    line_width_nm = pixel if line_width_nm is None else float(line_width_nm)
    if line_length_nm <= 0 or line_width_nm <= 0:
        raise ValueError("line_length_nm and line_width_nm must be positive.")

    padding_nm = 0.5 * line_length_nm + line_width_nm
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

    cmin, cmax = _resolve_color_limits(values, color_limits)
    angle_factor = math.pi / 180.0 if azimuth_unit.lower().startswith("deg") else 1.0

    rgb_sum = np.zeros((y_size, x_size, 3), dtype=np.float32)
    counts = np.zeros((y_size, x_size), dtype=np.float32)
    _render_vectors_njit(
        values,
        xx,
        yy,
        azimuth,
        rgb_sum,
        counts,
        float(pixel),
        float(x0),
        float(y0),
        line_length_nm,
        line_width_nm,
        float(cmin),
        float(cmax),
        float(angle_factor),
    )

    mask = counts > 0
    image = np.zeros_like(rgb_sum)
    image[mask] = rgb_sum[mask] / counts[mask][:, None]

    info = {
        "pixel_sr_nm": float(pixel),
        "x0": float(x0),
        "y0": float(y0),
        "color_limits": (float(cmin), float(cmax)),
        "line_length_nm": float(line_length_nm),
        "line_width_nm": float(line_width_nm),
    }
    return image, info


def _resolve_color_limits(values, color_limits):
    """Return finite color normalization limits."""
    finite = values[np.isfinite(values)]
    if color_limits is None:
        if len(finite) == 0:
            return 0.0, 1.0
        cmin = float(np.nanmin(finite))
        cmax = float(np.nanmax(finite))
    else:
        cmin, cmax = [float(v) for v in color_limits]
    if not np.isfinite(cmin) or not np.isfinite(cmax):
        raise ValueError("color_limits must be finite.")
    if cmin == cmax:
        cmin -= 0.5
        cmax += 0.5
    if cmin > cmax:
        cmin, cmax = cmax, cmin
    return cmin, cmax


@nb.njit(cache=True)
def _render_vectors_njit(
    values,
    xx,
    yy,
    azimuth,
    rgb_sum,
    counts,
    pixel,
    x0,
    y0,
    line_length_nm,
    line_width_nm,
    cmin,
    cmax,
    angle_factor,
):
    """Rasterize colored line segments."""
    y_size, x_size = counts.shape
    half_length = 0.5 * line_length_nm / pixel
    half_width = max(0.5 * line_width_nm / pixel, 0.5)
    radius = int(math.ceil(half_length + half_width + 1.0))

    for i in range(len(values)):
        value = values[i]
        x = xx[i]
        y = yy[i]
        angle = azimuth[i]
        if not math.isfinite(value) or not math.isfinite(x) or not math.isfinite(y):
            continue
        if not math.isfinite(angle):
            continue

        norm_value = (value - cmin) / (cmax - cmin)
        if norm_value < 0.0:
            norm_value = 0.0
        elif norm_value > 1.0:
            norm_value = 1.0
        red, green, blue = _blue_red_hsv(norm_value)

        theta = angle * angle_factor
        ux = math.cos(theta)
        uy = math.sin(theta)
        cx = (x - x0) / pixel
        cy = (y - y0) / pixel
        ix_center = int(math.floor(cx + 0.5))
        iy_center = int(math.floor(cy + 0.5))

        for iy in range(iy_center - radius, iy_center + radius + 1):
            if iy < 0 or iy >= y_size:
                continue
            dy = iy - cy
            for ix in range(ix_center - radius, ix_center + radius + 1):
                if ix < 0 or ix >= x_size:
                    continue
                dx = ix - cx
                along = dx * ux + dy * uy
                if along < -half_length or along > half_length:
                    continue
                perp = abs(-dx * uy + dy * ux)
                if perp > half_width:
                    continue

                rgb_sum[iy, ix, 0] += red
                rgb_sum[iy, ix, 1] += green
                rgb_sum[iy, ix, 2] += blue
                counts[iy, ix] += 1.0

    return rgb_sum, counts


@nb.njit(cache=True)
def _blue_red_hsv(value):
    """Map 0..1 to HSV blue-cyan-green-yellow-red RGB."""
    hue = (2.0 / 3.0) * (1.0 - value)
    h6 = hue * 6.0
    sector = int(math.floor(h6))
    fraction = h6 - sector
    x = 1.0 - abs(fraction * 2.0 - 1.0)

    if sector == 0:
        return 1.0, x, 0.0
    if sector == 1:
        return x, 1.0, 0.0
    if sector == 2:
        return 0.0, 1.0, x
    if sector == 3:
        return 0.0, x, 1.0
    if sector == 4:
        return x, 0.0, 1.0
    return 1.0, 0.0, x
