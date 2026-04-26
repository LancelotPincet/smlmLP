#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


import numpy as np

from smlmlp import analysis
from smlmlp.modules.analysis_LP._functions.image.image_smlm import image_smlm


@analysis(df_name="points")
def metric_squirrel(
    widefield_image,
    sr_image=None,
    /,
    xx=None,
    yy=None,
    image_sigma=None,
    mask=None,
    *,
    pixel_sr_nm=15.0,
    fit_intercept=True,
    ignore_zero=True,
    cuda=False,
    parallel=False,
):
    """
    Compare a wide-field image with a super-resolution image.

    The super-resolution image is linearly matched to the wide-field image
    before errors are measured, following the practical SQUIRREL idea without
    coupling the metric to plotting or file IO.

    Parameters
    ----------
    widefield_image : array-like
        Reference 2D wide-field image.
    sr_image : array-like, optional
        Super-resolution image to compare. If omitted, ``xx`` and ``yy`` are
        rendered into an SMLM image with the same shape as ``widefield_image``.
    xx, yy, image_sigma : array-like, optional
        Localizations used only when ``sr_image`` is omitted.
    mask : array-like of bool, optional
        Pixels included in the fit and metric calculation.
    fit_intercept : bool, optional
        Fit both scale and offset when true; otherwise fit scale only.
    ignore_zero : bool, optional
        Exclude zero-valued pixels in either image from the fit.

    Returns
    -------
    error_map : ndarray
        Absolute residual image after intensity matching.
    info : dict
        Contains ``scale``, ``offset``, ``rse``, ``rsp`` and ``n_pixels``.
    """

    wf = _as_2d_float(widefield_image, "widefield_image")
    render_info = None
    if sr_image is None:
        if xx is None or yy is None:
            raise ValueError("xx and yy are required when sr_image is omitted.")
        sr_image, render_info = image_smlm(
            xx,
            yy,
            image_sigma=image_sigma,
            shape=wf.shape,
            pixel_sr_nm=pixel_sr_nm,
            cuda=cuda,
            parallel=parallel,
        )

    sr = _as_2d_float(sr_image, "sr_image")
    if sr.shape != wf.shape:
        sr = _resize_like(sr, wf.shape)

    valid = np.isfinite(wf) & np.isfinite(sr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if ignore_zero:
        valid &= (wf != 0) & (sr != 0)
    if np.sum(valid) < 2:
        raise ValueError("At least two valid pixels are required for SQUIRREL.")

    scale, offset = _fit_linear(sr[valid], wf[valid], fit_intercept)
    sr_fit = sr * scale + offset
    residual = wf - sr_fit
    error_map = np.abs(residual).astype(np.float32)

    rse = float(np.sqrt(np.mean(residual[valid] ** 2)))
    rsp = _pearson(sr_fit[valid], wf[valid])

    info = {
        "scale": float(scale),
        "offset": float(offset),
        "rse": rse,
        "rsp": float(rsp),
        "n_pixels": int(np.sum(valid)),
    }
    if render_info is not None:
        info["render"] = render_info
    return error_map, info


def _as_2d_float(image, name):
    """Return a 2D float64 image."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2D image.")
    return image


def _resize_like(image, shape):
    """Resize an image to a target shape using OpenCV interpolation."""
    try:
        import cv2
    except ImportError as exc:
        raise ValueError(
            "OpenCV is required to compare images with different shapes."
        ) from exc

    interpolation = cv2.INTER_AREA if image.size > np.prod(shape) else cv2.INTER_LINEAR
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)


def _fit_linear(source, target, fit_intercept):
    """Fit ``target ~= scale * source + offset``."""
    source = np.asarray(source, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()

    if fit_intercept:
        design = np.column_stack((source, np.ones_like(source)))
        scale, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        return scale, offset

    denom = float(np.sum(source * source))
    if denom == 0.0:
        raise ValueError("Cannot fit a scale from a zero super-resolution image.")
    return float(np.sum(source * target) / denom), 0.0


def _pearson(a, b):
    """Return Pearson correlation, or NaN for flat inputs."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da = a - np.mean(a)
    db = b - np.mean(b)
    denom = np.sqrt(np.sum(da * da) * np.sum(db * db))
    if denom == 0.0:
        return np.nan
    return np.sum(da * db) / denom
