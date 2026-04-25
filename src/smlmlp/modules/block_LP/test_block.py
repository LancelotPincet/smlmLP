#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : block

"""
This file allows to test block

block : This function is a decorator to be used on block function, which allow to use config for default values.
"""

import importlib

import numpy as np
import pytest
import tifffile as tiff

import smlmlp.modules.block_LP._functions.blink.blink_temporal_on as blink_temporal_on_module
from smlmlp.modules.block_LP.block import block
from smlmlp.modules.block_LP._functions.detection.detect_snr import detect_snr
from smlmlp.modules.block_LP._functions.globdetection.globdet_channel import (
    globdet_channels,
)
from smlmlp.modules.block_LP._functions.loading.load_data import load_data
from smlmlp.modules.block_LP._functions.registration.registrate_optimize_images import (
    registrate_optimize_images,
)
from smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant import (
    registrate_solve_redundant,
)


class _Config:
    """Minimal config object used by decorator tests."""

    scale = 3
    offset = 2


def test_block_injects_config_values_and_returns_info():
    """Check config injection and the standardized block return shape."""

    @block(timeit=False)
    def scaled(value, /, scale=1, *, offset=0, cuda=False, parallel=False):
        """Scale a value and return an ``info`` dictionary.

        Parameters
        ----------
        value : int
            Input value.
        scale : int, optional
            Multiplicative scale.
        offset : int, optional
            Additive offset.
        cuda : bool, optional
            Whether CUDA execution was requested.
        parallel : bool, optional
            Whether parallel execution was requested.

        Returns
        -------
        tuple
            A tuple ``(result, info)``.
        """
        info = {"cuda": cuda, "parallel": parallel, "scale": scale}
        return value * scale + offset, info

    result, info = scaled(4, config=_Config())

    assert result == 14
    assert info == {"cuda": False, "parallel": False, "scale": 3}


def test_block_wraps_generators_and_tracks_time():
    """Check generator blocks preserve yielded tuples and timing."""

    @block(timeit=True)
    def generate(*, cuda=False, parallel=False):
        """Yield values with an ``info`` dictionary.

        Parameters
        ----------
        cuda : bool, optional
            Whether CUDA execution was requested.
        parallel : bool, optional
            Whether parallel execution was requested.

        Yields
        ------
        tuple
            A tuple ``(value, info)``.
        """
        for value in (1, 2):
            info = {"cuda": cuda, "parallel": parallel}
            yield value, info

    block.times.pop("generate", None)
    values = list(generate(cuda=False, parallel=False))

    assert values == [
        (1, {"cuda": False, "parallel": False}),
        (2, {"cuda": False, "parallel": False}),
    ]
    assert block.times["generate"] >= 0


def test_registrate_solve_redundant_returns_info_last():
    """Check an implemented block returns an info dict as the final item."""

    shiftx = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    shifty = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    abs_shiftx, abs_shifty, info = registrate_solve_redundant(shiftx, shifty)

    assert abs_shiftx.shape == (3,)
    assert abs_shifty.shape == (3,)
    assert isinstance(info, dict)
    assert info["pairs"] == [(0, 1), (0, 2), (1, 2)]


def test_load_data_yields_chunk_with_public_bbox_parameter(tmp_path):
    """Check TIFF loading uses the public bounding-box parameter."""
    path = tmp_path / "movie.tif"
    stack = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    tiff.imwrite(path, stack, photometric="minisblack")

    chunks = list(
        load_data(
            str(path),
            chunk=2,
            pad=0,
            cameras_bboxes=[[(1, 1, 4, 3)]],
            memmap=True,
        )
    )

    assert len(chunks) == 2
    channels, info = chunks[0]
    assert info["chunk0"] == 0
    assert info["chunk1"] == 1
    np.testing.assert_array_equal(channels[0], stack[:2, 1:3, 1:4])


def test_detect_snr_accepts_scalar_gain():
    """Check scalar gains are broadcast per channel."""
    signals = [np.array([[10.0, 20.0]], dtype=np.float32)]
    bkgds = [np.array([[4.0, 4.0]], dtype=np.float32)]

    snrs, info = detect_snr(signals, bkgds, channels_gains=1.0, parallel=False)

    np.testing.assert_allclose(snrs[0], [[5.0, 10.0]])
    assert info["channels_gains"] == [1.0]


def test_globdet_channels_merges_transformed_channels():
    """Check global detection returns one mean/std merged channel."""
    channels = [
        np.ones((2, 6, 8), dtype=np.float32),
        np.ones((2, 4, 10), dtype=np.float32) * 3,
    ]

    global_channels, info = globdet_channels(channels, mode="mean")
    std_channels, std_info = globdet_channels(channels, mode="std")

    assert len(global_channels) == 1
    assert len(std_channels) == 1
    assert global_channels[0].shape == (2, 4, 8)
    assert std_channels[0].shape == (2, 4, 8)
    np.testing.assert_allclose(global_channels[0], 2.0)
    np.testing.assert_allclose(std_channels[0], 1.0)
    assert len(info["transform_matrices"]) == 2
    assert len(std_info["transform_matrices"]) == 2
    assert info["crop_shape"] == (4, 8)

    out = [np.empty_like(global_channels[0])]
    reused_channels, _ = globdet_channels(channels, mode="mean", global_channels=out)
    assert reused_channels[0] is out[0]

    shifted_channels, shifted_info = globdet_channels(
        channels,
        channels_x_shifts_nm=[1.0, 0.0],
    )
    assert shifted_info["crop_shape"][1] < 8
    np.testing.assert_allclose(shifted_channels[0], 2.0)


def test_registrate_optimize_images_crops_transformed_projections():
    """Check registration projections are cropped to a shared center shape."""
    channels = [
        np.arange(3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8),
        np.arange(3 * 4 * 10, dtype=np.float32).reshape(3, 4, 10),
    ]

    optimized, info = registrate_optimize_images(channels)

    assert [image.shape for image in optimized] == [(4, 8), (4, 8)]
    assert info["crop_shape"] == (4, 8)

    out = [np.empty_like(image) for image in optimized]
    reused_optimized, _ = registrate_optimize_images(channels, optimized=out)
    assert all(image is buffer for image, buffer in zip(reused_optimized, out))

    shifted_optimized, shifted_info = registrate_optimize_images(
        channels,
        channels_x_shifts_nm=[1.0, 0.0],
    )
    assert shifted_info["crop_shape"][1] < 8
    assert len(shifted_optimized) == 2


def test_blink_temporal_on_normalizes_scalar_psf_and_default_crop(monkeypatch):
    """Check default crop length is derived from frames, not channel count."""
    channel = np.ones((10, 2, 2), dtype=np.float32)
    autocorr = np.tile(np.arange(10, 0, -1, dtype=np.float32)[:, None, None], (1, 2, 2))

    monkeypatch.setattr(
        blink_temporal_on_module,
        "img_gaussianfilter",
        lambda channel, **kwargs: np.zeros_like(channel),
    )
    monkeypatch.setattr(
        blink_temporal_on_module,
        "stack_autocorr",
        lambda channel, cuda=False, parallel=False: autocorr,
    )
    monkeypatch.setattr(
        blink_temporal_on_module,
        "curve_fit",
        lambda func, T, y, p0, bounds: (np.array([2.0, 0.0]), None),
    )

    on_time, info = blink_temporal_on_module.blink_temporal_on(
        [channel],
        psf_sigma_nm=120.0,
        exposure_ms=25.0,
    )

    assert on_time == 50.0
    assert info["time"].tolist() == [0.0, 25.0, 50.0, 75.0]


@pytest.mark.parametrize(
    ("module_name", "function_name", "args"),
    [
        (
            "smlmlp.modules.block_LP._functions._block_template",
            "block_template",
            ([],),
        ),
        (
            "smlmlp.modules.block_LP._functions.globlocalization.globloc_fit",
            "globloc_fit",
            ([], [], []),
        ),
        (
            "smlmlp.modules.block_LP._functions.registration.registrate_ecc_affine",
            "registrate_ecc_shift",
            ([],),
        ),
    ],
)
def test_unimplemented_blocks_raise_standard_error(module_name, function_name, args):
    """Check placeholder blocks fail explicitly."""

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        function(*args)


if __name__ == "__main__":
    from corelp import test

    test(__file__)
