#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : block

"""
Test all @block-decorated functions in block_LP/_functions/.
"""

import numpy as np
import pytest
import tifffile as tiff

from smlmlp.modules.block_LP.block import block


class _Config:
    scale = 3
    offset = 2


# %% test block decorator


def test_block_injects_config_values_and_returns_info():
    @block(timeit=False)
    def scaled(value, /, scale=1, *, offset=0, cuda=False, parallel=False):
        info = {"cuda": cuda, "parallel": parallel, "scale": scale}
        return value * scale + offset, info

    result, info = scaled(4, config=_Config())

    assert result == 14
    assert info == {"cuda": False, "parallel": False, "scale": 3}


def test_block_wraps_generators_and_tracks_time():
    @block(timeit=True)
    def generate(*, cuda=False, parallel=False):
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


# %% test registrate_solve_redundant_shift


def test_registrate_solve_redundant_shift():
    from smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant_shift import (
        registrate_solve_redundant_shift,
    )

    shiftx = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    shifty = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    abs_shiftx, abs_shifty, info = registrate_solve_redundant_shift(
        shiftx, shifty
    )

    assert abs_shiftx.shape == (3,)
    assert abs_shifty.shape == (3,)
    assert isinstance(info, dict)
    assert info["pairs"] == [(0, 1), (0, 2), (1, 2)]


def test_registrate_solve_redundant_shift_with_outliers():
    from smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant_shift import (
        registrate_solve_redundant_shift,
    )

    shiftx = np.array([1.0, 2.0, 10.0], dtype=np.float32)
    shifty = np.array([0.0, 1.0, 8.0], dtype=np.float32)

    _, _, info = registrate_solve_redundant_shift(
        shiftx, shifty, sigma_thresh=2.0, max_outliers=1
    )

    assert info["outlier_idx"].ndim == 1


# %% test registrate_solve_redundant_affine


def test_registrate_solve_redundant_affine():
    from smlmlp.modules.block_LP._functions.registration.registrate_solve_redundant_affine import (
        registrate_solve_redundant_affine,
    )

    shiftx = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    shifty = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    angle = np.zeros(3, dtype=np.float32)
    shearx = np.zeros(3, dtype=np.float32)
    sheary = np.zeros(3, dtype=np.float32)
    scalex = np.ones(3, dtype=np.float32)
    scaley = np.ones(3, dtype=np.float32)

    (
        abs_shiftx,
        abs_shifty,
        abs_angle,
        abs_shearx,
        abs_sheary,
        abs_scalex,
        abs_scaley,
        info,
    ) = registrate_solve_redundant_affine(
        shiftx,
        shifty,
        angle,
        shearx,
        sheary,
        scalex,
        scaley,
        {"shape": (16, 16), "ref_pix": (1.0, 1.0)},
    )

    np.testing.assert_allclose(abs_shiftx, [-1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(abs_shifty, [-1 / 3, -1 / 3, 2 / 3], atol=1e-6)
    np.testing.assert_allclose(abs_angle, 0.0, atol=1e-6)
    np.testing.assert_allclose(abs_scalex, 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.mean(info["abs_matrices_second_pass"], axis=0),
        np.eye(3),
        atol=1e-6,
    )


# %% test registrate_pcc_shift


def test_registrate_pcc_shift():
    from smlmlp.modules.block_LP._functions.registration.registrate_pcc_shift import (
        registrate_pcc_shift,
    )

    ch1 = np.random.rand(16, 16).astype(np.float32)
    ch2 = np.random.rand(16, 16).astype(np.float32)

    shiftx, shifty, info = registrate_pcc_shift([ch1, ch2])

    assert len(shiftx) == 1
    assert len(shifty) == 1
    assert info["pairs"] == [(0, 1)]


def test_registrate_pcc_shift_multiple_channels():
    from smlmlp.modules.block_LP._functions.registration.registrate_pcc_shift import (
        registrate_pcc_shift,
    )

    channels = [np.random.rand(16, 16).astype(np.float32) for _ in range(3)]

    _, _, info = registrate_pcc_shift(channels)

    assert len(info["pairs"]) == 3


# %% test registrate_optimize_images


def test_registrate_optimize_images():
    from smlmlp.modules.block_LP._functions.registration.registrate_optimize_images import (
        registrate_optimize_images,
    )

    channels = [
        np.arange(3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8),
        np.arange(3 * 4 * 10, dtype=np.float32).reshape(3, 4, 10),
    ]

    optimized, info = registrate_optimize_images(channels)

    assert [image.shape for image in optimized] == [(4, 8), (4, 8)]
    assert info["crop_shape"] == (4, 8)


# %% test registrate_ecc_affine


def test_registrate_ecc_affine():
    pytest.skip("ECC registration requires similar images to converge")


# %% test load_data


def test_load_data(tmp_path):
    from smlmlp.modules.block_LP._functions.loading.load_data import load_data

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


def test_load_data_reserves_resource_tokens_and_limits_chunk(tmp_path, monkeypatch):
    import importlib

    load_data_module = importlib.import_module(
        "smlmlp.modules.block_LP._functions.loading.load_data"
    )

    class _FakeResource:
        def __init__(self, granted_tokens, reservation_id):
            self.granted_tokens = granted_tokens
            self.reservation_id = reservation_id
            self.takes = []
            self.releases = []

        def take(self, value, owner, minimum=None, **kwargs):
            token = {
                "ok": True,
                "granted_tokens": self.granted_tokens,
                "reservation_id": self.reservation_id,
            }
            self.takes.append({"value": value, "owner": owner, "minimum": minimum})
            return token

        def release(self, token):
            self.releases.append(token)
            return {"ok": True}

    class _FakeGPU:
        def cuda(self):
            return True

    path = tmp_path / "movie.tif"
    stack = np.arange(8 * 4 * 4, dtype=np.uint16).reshape(8, 4, 4)
    tiff.imwrite(path, stack, photometric="minisblack")

    requested_chunk = 6
    granted_chunk = 3
    pad = 1
    frame_size_gb = 4 * 4 * np.dtype(np.float32).itemsize / 1024**3
    ram_factor = load_data_module._MEMORY_COPY_FACTOR + 1
    vram_factor = load_data_module._MEMORY_COPY_FACTOR
    ram_requested = (requested_chunk + 2 * pad) * frame_size_gb * ram_factor
    vram_requested = (requested_chunk + 2 * pad) * frame_size_gb * vram_factor
    vram_granted = (granted_chunk + 2 * pad) * frame_size_gb * vram_factor

    fake_ram = _FakeResource(ram_requested, "ram-token")
    fake_vram = _FakeResource(vram_granted, "vram-token")
    monkeypatch.setattr(load_data_module.computer, "ram", fake_ram)
    monkeypatch.setattr(load_data_module.computer, "vram", fake_vram)
    monkeypatch.setattr(load_data_module.computer, "gpu", _FakeGPU())

    chunks = list(
        load_data_module.load_data(
            str(path),
            chunk=requested_chunk,
            pad=pad,
            memmap=True,
            cuda=True,
        )
    )

    info = chunks[0][1]
    assert len(chunks) == 3
    assert info["chunk_requested"] == requested_chunk
    assert info["chunk"] == granted_chunk
    np.testing.assert_allclose(info["ram_tokens_requested"], ram_requested)
    np.testing.assert_allclose(info["vram_tokens_requested"], vram_requested)
    np.testing.assert_allclose(
        info["ram_tokens_minimum"],
        (1 + 2 * pad) * frame_size_gb * ram_factor,
    )
    np.testing.assert_allclose(
        info["vram_tokens_minimum"],
        (1 + 2 * pad) * frame_size_gb * vram_factor,
    )
    assert fake_ram.takes[0]["owner"] == "load_data:ram"
    assert fake_vram.takes[0]["owner"] == "load_data:vram"
    assert fake_ram.releases[0]["reservation_id"] == "ram-token"
    assert fake_vram.releases[0]["reservation_id"] == "vram-token"


# %% test load_chunking


def test_load_chunking(tmp_path):
    from smlmlp.modules.block_LP._functions.loading.load_chunking import load_chunking

    path = tmp_path / "ROI.tif"
    tiff.imwrite(path, np.zeros((20, 32, 32), dtype=np.uint16))

    loaded_max, info = load_chunking(str(path), cuda=False, parallel=False)

    assert loaded_max >= 0
    assert "loaded_max_cpu" in info


# %% test detect_snr


def test_detect_snr():
    from smlmlp.modules.block_LP._functions.detection.detect_snr import detect_snr

    signals = [np.array([[10.0, 20.0]], dtype=np.float32)]
    bkgds = [np.array([[4.0, 4.0]], dtype=np.float32)]

    snrs, info = detect_snr(signals, bkgds, channels_gains=1.0, parallel=False)

    np.testing.assert_allclose(snrs[0], [[5.0, 10.0]])
    assert info["channels_gains"] == [1.0]


def test_detect_snr_multi_channel():
    from smlmlp.modules.block_LP._functions.detection.detect_snr import detect_snr

    signals = [
        np.array([[10.0, 20.0]], dtype=np.float32),
        np.array([[5.0, 15.0]], dtype=np.float32),
    ]
    bkgds = [
        np.array([[4.0, 4.0]], dtype=np.float32),
        np.array([[9.0, 9.0]], dtype=np.float32),
    ]

    snrs, info = detect_snr(
        signals,
        bkgds,
        noise_corrections=[1.0, 2.0],
        channels_gains=[0.5, 1.0],
    )

    assert len(snrs) == 2


# %% test detect_spatial_maxima


def test_detect_spatial_maxima():
    from smlmlp.modules.block_LP._functions.detection.detect_spatial_maxima import (
        detect_spatial_maxima,
    )

    snr = np.random.rand(5, 16, 16).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)

    fr, x, y, ch, info = detect_spatial_maxima(
        [snr],
        0.9,
        [kernel],
        channels_pixels_nm=[(100.0, 100.0)],
    )

    assert fr.ndim == x.ndim == y.ndim == ch.ndim == 1
    assert "footprints" in info


def test_detect_spatial_maxima_multi_channel():
    from smlmlp.modules.block_LP._functions.detection.detect_spatial_maxima import (
        detect_spatial_maxima,
    )

    snrs = [
        np.random.rand(4, 16, 16).astype(np.float32),
        np.random.rand(4, 16, 16).astype(np.float32),
    ]
    kernels = [
        np.ones((3, 3), dtype=np.float32),
        np.ones((5, 5), dtype=np.float32),
    ]

    _, _, _, _, info = detect_spatial_maxima(
        snrs,
        1.2,
        kernels,
        channels_pixels_nm=[(100.0, 100.0), (110.0, 110.0)],
    )

    assert len(info["footprints"]) == 2


# %% test detect_gain


def test_detect_gain():
    from smlmlp.modules.block_LP._functions.detection.detect_gain import detect_gain

    channel = np.random.rand(20, 32, 32).astype(np.float32)

    gains, info = detect_gain([channel], nbins=50)

    assert len(gains) == 1
    assert "gain" in info
    assert "mean" in info
    assert "var" in info


def test_detect_gain_multi_channel():
    from smlmlp.modules.block_LP._functions.detection.detect_gain import detect_gain

    channels = [
        np.random.rand(20, 32, 32).astype(np.float32),
        np.random.rand(20, 32, 32).astype(np.float32),
    ]

    gains, info = detect_gain(channels, nbins=40)

    assert len(gains) == 2
    assert len(info["gain"]) == 2


# %% test blink_temporal_on


def test_blink_temporal_on(monkeypatch):
    from smlmlp.modules.block_LP._functions.blink import blink_temporal_on as module

    channel = np.ones((10, 2, 2), dtype=np.float32)
    autocorr = np.tile(
        np.arange(10, 0, -1, dtype=np.float32)[:, None, None], (1, 2, 2)
    )

    monkeypatch.setattr(
        module,
        "img_gaussianfilter",
        lambda channel, **kwargs: np.zeros_like(channel),
    )
    monkeypatch.setattr(
        module,
        "stack_autocorr",
        lambda channel, cuda=False, parallel=False: autocorr,
    )
    monkeypatch.setattr(
        module,
        "curve_fit",
        lambda func, T, y, p0, bounds: (np.array([2.0, 0.0]), None),
    )

    on_time, info = module.blink_temporal_on(
        [channel],
        psf_sigma_nm=120.0,
        exposure_ms=25.0,
    )

    assert on_time == 50.0
    assert info["time"].tolist() == [0.0, 25.0, 50.0, 75.0]


# %% test blink_spatial_psf


def test_blink_spatial_psf():
    from smlmlp.modules.block_LP._functions.blink.blink_spatial_psf import blink_spatial_psf

    channel = np.random.rand(20, 64, 64).astype(np.float32)

    psf_sigma, info = blink_spatial_psf([channel], crop_pix=41)

    assert len(psf_sigma) == 1
    assert "ac" in info
    assert "psf" in info


def test_blink_spatial_psf_multi_channel():
    from smlmlp.modules.block_LP._functions.blink.blink_spatial_psf import blink_spatial_psf

    channels = [
        np.random.rand(20, 64, 64).astype(np.float32),
        np.random.rand(20, 64, 64).astype(np.float32),
    ]

    psf_sigma, info = blink_spatial_psf(
        channels,
        crop_pix=31,
        channels_pixels_nm=[(100.0, 100.0), (110.0, 110.0)],
    )

    assert len(psf_sigma) == 2
    assert len(info["psf"]) == 2


# %% test crop_individual_extract


def test_crop_individual_extract():
    from smlmlp.modules.block_LP._functions.crop.crop_individual_extract import (
        crop_individual_extract,
    )

    channel = np.random.rand(10, 32, 32).astype(np.float32)
    fr = np.array([1, 2, 3])
    x = np.array([100.0, 150.0, 200.0])
    y = np.array([120.0, 180.0, 220.0])

    crops, X0, Y0, info = crop_individual_extract([channel], fr, x, y)

    assert len(crops) == 1
    assert len(crops[0]) == 3
    assert np.issubdtype(X0[0].dtype, np.signedinteger)
    assert np.issubdtype(Y0[0].dtype, np.signedinteger)
    assert X0[0][0] < 0
    assert Y0[0][0] < 0


def test_crop_individual_extract_one_based_channels():
    from smlmlp.modules.block_LP._functions.crop.crop_individual_extract import (
        crop_individual_extract,
    )

    channels = [
        np.ones((3, 16, 16), dtype=np.float32),
        np.ones((3, 16, 16), dtype=np.float32) * 2,
    ]
    fr = np.array([1, 1])
    x = np.array([800.0, 800.0])
    y = np.array([800.0, 800.0])
    ch = np.array([1, 2], dtype=np.uint8)

    crops, _, _, _ = crop_individual_extract(
        channels,
        fr,
        x,
        y,
        ch=ch,
        channels_crops_pix=3,
        channels_pixels_nm=100.0,
    )

    assert len(crops) == 2
    np.testing.assert_allclose(crops[0], 1.0)
    np.testing.assert_allclose(crops[1], 2.0)


def test_crop_individual_extract_rejects_zero_based_channels():
    from smlmlp.modules.block_LP._functions.crop.crop_individual_extract import (
        crop_individual_extract,
    )

    channels = [np.ones((1, 8, 8), dtype=np.float32)]

    with pytest.raises(ValueError, match="one-based"):
        crop_individual_extract(
            channels,
            np.array([1]),
            np.array([400.0]),
            np.array([400.0]),
            ch=np.array([0], dtype=np.uint8),
        )


# %% test crop_remove_bkgd


def test_crop_remove_bkgd():
    from smlmlp.modules.block_LP._functions.crop.crop_remove_bkgd import crop_remove_bkgd

    crops = [np.random.rand(4, 7, 7).astype(np.float32)]

    new_crops, info = crop_remove_bkgd(crops)

    assert len(new_crops) == 1
    assert new_crops[0].shape == (4, 7, 7)
    assert "border_medians" in info


def test_crop_remove_bkgd_multi_channel():
    from smlmlp.modules.block_LP._functions.crop.crop_remove_bkgd import crop_remove_bkgd

    crops = [
        np.random.rand(3, 9, 9).astype(np.float32),
        np.random.rand(2, 11, 11).astype(np.float32),
    ]

    new_crops, info = crop_remove_bkgd(crops, cuda=False, parallel=True)

    assert len(new_crops) == 2


# %% test bkgd_spatial_mean


def test_bkgd_spatial_mean():
    from smlmlp.modules.block_LP._functions.background.bkgd_spatial_mean import (
        bkgd_spatial_mean,
    )

    channels = [
        np.random.rand(4, 16, 16).astype(np.float32),
        np.random.rand(4, 16, 16).astype(np.float32),
    ]

    bkgds, noise_corr, info = bkgd_spatial_mean(channels, 7.0)

    assert len(bkgds) == len(channels)
    assert len(noise_corr) == len(channels)
    assert "spatial_mean_sigmas" in info


def test_bkgd_spatial_mean_anisotropic():
    from smlmlp.modules.block_LP._functions.background.bkgd_spatial_mean import (
        bkgd_spatial_mean,
    )

    channels = [
        np.random.rand(4, 16, 16).astype(np.float32),
        np.random.rand(4, 16, 16).astype(np.float32),
    ]

    radii = [(6.0, 6.0), (8.0, 10.0)]
    _, _, info = bkgd_spatial_mean(channels, channels_mean_radii_pix=radii)

    assert len(info["channels_mean_radii_pix"]) == 2


# %% test bkgd_combination


def test_bkgd_combination():
    from smlmlp.modules.block_LP._functions.background.bkgd_combination import (
        bkgd_combination,
    )

    channels = [
        np.random.rand(10, 16, 16).astype(np.float32),
        np.random.rand(10, 16, 16).astype(np.float32),
    ]

    bkgds, noise_corr, info = bkgd_combination(channels)

    assert len(bkgds) == 2
    assert len(noise_corr) == 2
    assert isinstance(info, dict)


def test_bkgd_combination_spatial_opening_only():
    from smlmlp.modules.block_LP._functions.background.bkgd_combination import (
        bkgd_combination,
    )

    channels = [
        np.random.rand(10, 16, 16).astype(np.float32),
        np.random.rand(10, 16, 16).astype(np.float32),
    ]

    _, _, info = bkgd_combination(
        channels,
        do_spatial_opening=True,
        do_temporal_median=False,
        do_spatial_mean=False,
        channels_opening_radii_pix=4.0,
    )

    assert "footprints" in info


# %% test bkgd_spatial_opening


def test_bkgd_spatial_opening():
    from smlmlp.modules.block_LP._functions.background.bkgd_spatial_opening import (
        bkgd_spatial_opening,
    )

    channels = [
        np.random.rand(3, 16, 16).astype(np.float32),
        np.random.rand(3, 16, 16).astype(np.float32),
    ]

    bkgds, noise_corr, info = bkgd_spatial_opening(channels, 3.0)

    assert len(bkgds) == 2
    assert len(noise_corr) == 2
    assert len(info["footprints"]) == 2


def test_bkgd_spatial_opening_per_channel():
    from smlmlp.modules.block_LP._functions.background.bkgd_spatial_opening import (
        bkgd_spatial_opening,
    )

    channels = [
        np.random.rand(3, 16, 16).astype(np.float32),
        np.random.rand(3, 16, 16).astype(np.float32),
    ]

    radii = [(3.0, 3.0), (4.0, 6.0)]
    _, _, info = bkgd_spatial_opening(channels, channels_opening_radii_pix=radii)

    assert len(info["channels_opening_radii_pix"]) == 2


# %% test bkgd_temporal_median


def test_bkgd_temporal_median():
    from smlmlp.modules.block_LP._functions.background.bkgd_temporal_median import (
        bkgd_temporal_median,
    )

    channels = [
        np.random.rand(10, 16, 16).astype(np.float32),
        np.random.rand(10, 16, 16).astype(np.float32),
    ]

    bkgds, noise_corr, info = bkgd_temporal_median(channels, median_window_fr=25)

    assert len(bkgds) == 2
    assert info["median_window_fr"] == 25


def test_bkgd_temporal_median_preallocated():
    from smlmlp.modules.block_LP._functions.background.bkgd_temporal_median import (
        bkgd_temporal_median,
    )

    channels = [
        np.random.rand(10, 16, 16).astype(np.float32),
        np.random.rand(10, 16, 16).astype(np.float32),
    ]

    out = [np.empty_like(ch) for ch in channels]
    bkgds, _, _ = bkgd_temporal_median(
        channels,
        median_window_fr=11,
        bkgds=out,
    )

    assert len(bkgds) == len(channels)


# %% test signal_temporal_filter


def test_signal_temporal_filter():
    from smlmlp.modules.block_LP._functions.signal.signal_temporal_filter import (
        signal_temporal_filter,
    )

    channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    kernel = np.array([1.0, -1.0], dtype=np.float32)

    signals, noise_corr, info = signal_temporal_filter(channels, kernel)

    assert len(signals) == 1
    assert "kernel_factor" in info


def test_signal_temporal_filter_with_background():
    from smlmlp.modules.block_LP._functions.signal.signal_temporal_filter import (
        signal_temporal_filter,
    )

    channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    kernel = np.array([1.0, -1.0], dtype=np.float32)
    bkgds = [np.zeros_like(channels[0])]

    signals, noise_corr, info = signal_temporal_filter(
        channels,
        kernel,
        bkgds=bkgds,
        noise_corrections=[np.float32(2.0)],
    )

    assert len(noise_corr) == 1


# %% test signal_spatial_filter


def test_signal_spatial_filter():
    from smlmlp.modules.block_LP._functions.signal.signal_spatial_filter import (
        signal_spatial_filter,
    )

    channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    kernels = [np.ones((3, 3), dtype=np.float32)]

    signals, noise_corr, info = signal_spatial_filter(channels, kernels)

    assert len(signals) == 1
    assert len(info["kernel_factors"]) == 1


def test_signal_spatial_filter_multi_channel():
    from smlmlp.modules.block_LP._functions.signal.signal_spatial_filter import (
        signal_spatial_filter,
    )

    channels = [
        np.random.rand(10, 8, 8).astype(np.float32),
        np.random.rand(10, 8, 8).astype(np.float32),
    ]
    kernels = [
        np.ones((3, 3), dtype=np.float32),
        np.ones((5, 5), dtype=np.float32),
    ]

    signals, noise_corr, info = signal_spatial_filter(channels, kernels)

    assert len(info["kernel_factors"]) == 2


# %% test signal_combination


def test_signal_combination():
    from smlmlp.modules.block_LP._functions.signal.signal_combination import (
        signal_combination,
    )

    channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    spatial_kernels = [np.ones((3, 3), dtype=np.float32)]

    signals, noise_corr, info = signal_combination(
        channels,
        channels_spatial_kernels=spatial_kernels,
        do_spatial_filter=True,
        do_temporal_filter=False,
    )

    assert len(signals) == 1
    assert "channels_spatial_kernels" in info


def test_signal_combination_with_temporal():
    from smlmlp.modules.block_LP._functions.signal.signal_combination import (
        signal_combination,
    )

    channels = [np.random.rand(10, 8, 8).astype(np.float32)]
    spatial_kernels = [np.ones((3, 3), dtype=np.float32)]
    temporal_kernel = np.array([1.0, -1.0], dtype=np.float32)

    signals, noise_corr, info = signal_combination(
        channels,
        channels_spatial_kernels=spatial_kernels,
        temporal_kernel=temporal_kernel,
        do_spatial_filter=True,
        do_temporal_filter=True,
    )

    assert "temporal_kernel" in info


# %% test globloc_fit


def test_globloc_fit_gauss():
    from smlmlp.modules.block_LP._functions.globlocalization.globloc_fit import (
        globloc_fit,
    )

    crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    x0 = [np.array([10, 20], dtype=np.float32)]
    y0 = [np.array([30, 40], dtype=np.float32)]
    models = ["gauss"]
    inits = [{"sigx": 90.0, "sigy": 90.0, "theta": 0.0, "theta_fit": False}]

    mux, muy, info = globloc_fit(
        crops, x0, y0,
        channels_models=models,
        channels_fit_inits=inits,
        channels_pixels_nm=[(100.0, 100.0)],
    )

    assert mux.shape == muy.shape


def test_globloc_fit_isogauss():
    from smlmlp.modules.block_LP._functions.globlocalization.globloc_fit import (
        globloc_fit,
    )

    crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    x0 = [np.array([10, 20], dtype=np.float32)]
    y0 = [np.array([30, 40], dtype=np.float32)]
    models = ["isogauss"]
    inits = [{"sig": 90.0}]

    mux, muy, info = globloc_fit(
        crops, x0, y0,
        channels_models=models,
        channels_fit_inits=inits,
        channels_pixels_nm=[(100.0, 100.0)],
    )

    assert info["sigma"].ndim == 1


# %% test locs_individual_fit


def test_locs_individual_fit_dispatches_channel_models(monkeypatch):
    import importlib

    module = importlib.import_module(
        "smlmlp.modules.block_LP._functions.localization.locs_individual_fit"
    )
    calls = []

    class _FakeFit:
        """No-op optimizer recording fitted model names."""

        def __init__(self, function, estimator):
            """Store the model function."""
            self.function = function

        def __call__(self, *args):
            """Record the fit call without changing model parameters."""
            calls.append((self.function.model, len(args)))

    class _FakeModel:
        """Minimal model preserving initialized parameters."""

        model = None

        def __init__(self, mux, muy, amp, offset, cuda=False, **kwargs):
            """Store fit parameters and initialization keywords."""
            self.muz = kwargs.pop("muz", None)
            self.mux = mux
            self.muy = muy
            self.amp = amp
            self.offset = offset
            self.kwargs = kwargs

    class _FakeIsoGaussian(_FakeModel):
        """Fake isotropic Gaussian model."""

        model = "isogauss"

        def __init__(self, *args, **kwargs):
            """Store isotropic sigma."""
            super().__init__(*args, **kwargs)
            self.sig = np.full_like(self.mux, self.kwargs["sig"], dtype=np.float32)

    class _FakeGaussian2D(_FakeModel):
        """Fake anisotropic Gaussian model."""

        model = "gauss"

        def __init__(self, *args, **kwargs):
            """Store anisotropic sigmas."""
            super().__init__(*args, **kwargs)
            self.sigx = np.full_like(self.mux, self.kwargs["sigx"], dtype=np.float32)
            self.sigy = np.full_like(self.muy, self.kwargs["sigy"], dtype=np.float32)

    class _FakeSpline3D(_FakeModel):
        """Fake 3D spline model."""

        model = "spline"

    def _fake_mle(distribution):
        """Return a lightweight fake estimator."""
        return ("mle", distribution)

    def _fake_poisson():
        """Return a lightweight fake distribution."""
        return "poisson"

    monkeypatch.setattr(module, "LM", _FakeFit)
    monkeypatch.setattr(module, "MLE", _fake_mle)
    monkeypatch.setattr(module, "Poisson", _fake_poisson)
    monkeypatch.setattr(module, "IsoGaussian", _FakeIsoGaussian)
    monkeypatch.setattr(module, "Gaussian2D", _FakeGaussian2D)
    monkeypatch.setattr(module, "Spline3D", _FakeSpline3D)

    crops = [np.full((1, 3, 3), i + 1, dtype=np.float32) for i in range(3)]
    x0 = [
        np.array([10.0], dtype=np.float32),
        np.array([20.0], dtype=np.float32),
        np.array([30.0], dtype=np.float32),
    ]
    y0 = [
        np.array([100.0], dtype=np.float32),
        np.array([200.0], dtype=np.float32),
        np.array([300.0], dtype=np.float32),
    ]
    tx = [np.linspace(-1.0, 1.0, 8, dtype=np.float32) for _ in range(3)]
    ty = [np.linspace(-1.0, 1.0, 8, dtype=np.float32) for _ in range(3)]
    tz = [np.linspace(-1.0, 1.0, 8, dtype=np.float32) for _ in range(3)]
    coeffs = [np.ones((4, 4, 4), dtype=np.float32) for _ in range(3)]

    mux, muy, muz, info = module.locs_individual_fit(
        crops,
        x0,
        y0,
        channels_fit_models=["isogauss", "gauss", "spline"],
        channels_pixels_nm=[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
        channels_psf_sigmas_nm=[2.0, 2.0, 2.0],
        channels_psf_xsigmas_nm=[3.0, 3.0, 3.0],
        channels_psf_ysigmas_nm=[4.0, 4.0, 4.0],
        channels_psf_3d_xtangents=tx,
        channels_psf_3d_ytangents=ty,
        channels_psf_3d_ztangents=tz,
        channels_psf_3d_spline_coeffs=coeffs,
    )

    np.testing.assert_allclose(mux, [11.0, 21.0, 31.0])
    np.testing.assert_allclose(muy, [101.0, 201.0, 301.0])
    np.testing.assert_allclose(muz, [np.nan, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(info["amp"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(info["offset"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(info["sigma"], [2.0, np.sqrt(12.0), np.nan], equal_nan=True)
    np.testing.assert_allclose(info["sigmax"], [np.nan, 3.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(info["sigmay"], [np.nan, 4.0, np.nan], equal_nan=True)
    assert info["models"] == ["isogauss", "gauss", "spline"]
    assert calls == [("isogauss", 3), ("gauss", 3), ("spline", 4)]


def test_locs_individual_fit_rejects_unknown_model():
    from smlmlp.modules.block_LP._functions.localization.locs_individual_fit import (
        locs_individual_fit,
    )

    crops = [np.zeros((1, 3, 3), dtype=np.float32)]
    x0 = [np.zeros(1, dtype=np.float32)]
    y0 = [np.zeros(1, dtype=np.float32)]

    with pytest.raises(ValueError, match="channels_fit_models"):
        locs_individual_fit(crops, x0, y0, channels_fit_models=["bad"])


# %% test locs_individual_gaussfit


@pytest.mark.skip(reason="source file has variable ordering bug")
def test_locs_individual_gaussfit():
    from smlmlp.modules.block_LP._functions.localization.locs_individual_gaussfit import (
        locs_individual_gaussfit,
    )

    crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    x0 = [np.array([10, 20], dtype=np.float32)]
    y0 = [np.array([30, 40], dtype=np.float32)]

    mux, muy, info = locs_individual_gaussfit(
        crops,
        x0,
        y0,
        channels_pixels_nm=[(100.0, 100.0)],
        channels_psf_xsigmas_nm=[90.0],
        channels_psf_ysigmas_nm=[90.0],
    )

    assert mux.shape == muy.shape
    assert sorted(info) == ["amp", "offset", "sigmax", "sigmay"]


# %% test locs_individual_isogaussfit


def test_locs_individual_isogaussfit():
    from smlmlp.modules.block_LP._functions.localization.locs_individual_isogaussfit import (
        locs_individual_isogaussfit,
    )

    crops = [np.random.rand(2, 7, 7).astype(np.float32)]
    x0 = [np.array([10, 20], dtype=np.float32)]
    y0 = [np.array([30, 40], dtype=np.float32)]

    mux, muy, info = locs_individual_isogaussfit(
        crops,
        x0,
        y0,
        channels_pixels_nm=[(100.0, 100.0)],
        channels_psf_sigmas_nm=[90.0],
    )

    assert mux.shape == muy.shape
    assert sorted(info) == ["amp", "offset", "sigma"]


# %% test locs_individual_splinefit


def test_locs_individual_splinefit(monkeypatch):
    import importlib

    module = importlib.import_module(
        "smlmlp.modules.block_LP._functions.localization.locs_individual_splinefit"
    )
    calls = []

    class _FakeFit:
        """No-op optimizer recording spline fit calls."""

        def __init__(self, function, estimator):
            """Store the model function."""
            self.function = function

        def __call__(self, *args):
            """Record the fit call without changing model parameters."""
            calls.append(len(args))

    class _FakeSpline3D:
        """Minimal 3D spline model preserving initialized parameters."""

        def __init__(self, mux, muy, muz, amp, offset, cuda=False, **kwargs):
            """Store fit parameters and spline metadata."""
            self.mux = mux
            self.muy = muy
            self.muz = muz
            self.amp = amp
            self.offset = offset
            self.kwargs = kwargs

    def _fake_mle(distribution):
        """Return a lightweight fake estimator."""
        return ("mle", distribution)

    def _fake_poisson():
        """Return a lightweight fake distribution."""
        return "poisson"

    monkeypatch.setattr(module, "LM", _FakeFit)
    monkeypatch.setattr(module, "MLE", _fake_mle)
    monkeypatch.setattr(module, "Poisson", _fake_poisson)
    monkeypatch.setattr(module, "Spline3D", _FakeSpline3D)

    crops = [np.full((2, 7, 7), 5.0, dtype=np.float32)]
    x0 = [np.array([10, 20], dtype=np.float32)]
    y0 = [np.array([30, 40], dtype=np.float32)]
    tx = [np.linspace(-1.0, 1.0, 8, dtype=np.float32)]
    ty = [np.linspace(-1.0, 1.0, 8, dtype=np.float32)]
    tz = [np.linspace(-0.5, 0.5, 8, dtype=np.float32)]
    coeffs = [np.ones((4, 4, 4), dtype=np.float32)]

    mux, muy, muz, info = module.locs_individual_splinefit(
        crops,
        x0,
        y0,
        channels_pixels_nm=[(1.0, 1.0)],
        channels_psf_3d_xtangents=tx,
        channels_psf_3d_ytangents=ty,
        channels_psf_3d_ztangents=tz,
        channels_psf_3d_spline_coeffs=coeffs,
    )

    np.testing.assert_allclose(mux, [13.0, 23.0])
    np.testing.assert_allclose(muy, [33.0, 43.0])
    np.testing.assert_allclose(muz, [0.0, 0.0])
    np.testing.assert_allclose(info["amp"], [5.0, 5.0])
    np.testing.assert_allclose(info["offset"], [5.0, 5.0])
    assert sorted(info) == ["amp", "offset"]
    assert calls == [4]


# %% test locs_individual_barycenter


def test_locs_individual_barycenter():
    from smlmlp.modules.block_LP._functions.localization.locs_individual_barycenter import (
        locs_individual_barycenter,
    )

    crops = [np.random.rand(3, 5, 5).astype(np.float32)]
    x0 = [np.array([10, 20, 30], dtype=np.float32)]
    y0 = [np.array([5, 15, 25], dtype=np.float32)]

    mux, muy, info = locs_individual_barycenter(crops, x0, y0)

    assert mux.shape == muy.shape


def test_locs_individual_barycenter_with_pixels():
    from smlmlp.modules.block_LP._functions.localization.locs_individual_barycenter import (
        locs_individual_barycenter,
    )

    crops = [np.random.rand(3, 5, 5).astype(np.float32)]
    x0 = [np.array([10, 20, 30], dtype=np.float32)]
    y0 = [np.array([5, 15, 25], dtype=np.float32)]
    pix = [(100.0, 120.0)]

    mux, muy, info = locs_individual_barycenter(crops, x0, y0, channels_pixels_nm=pix)

    assert mux.ndim == 1


def test_locs_individual_barycenter_cuda_matches_cpu():
    from numba import cuda
    from smlmlp.modules.block_LP._functions.localization.locs_individual_barycenter import (
        locs_individual_barycenter,
    )

    if not cuda.is_available():
        pytest.skip("CUDA is not available")

    rng = np.random.default_rng(0)
    crops = [rng.random((8, 15, 15), dtype=np.float32)]
    x0 = [np.arange(-4, 4, dtype=np.int32)]
    y0 = [np.arange(3, 11, dtype=np.int32)]
    pix = [(108.0, 108.0)]

    cpu_x, cpu_y, _ = locs_individual_barycenter(
        crops,
        x0,
        y0,
        channels_pixels_nm=pix,
        cuda=False,
    )
    gpu_x, gpu_y, _ = locs_individual_barycenter(
        crops,
        x0,
        y0,
        channels_pixels_nm=pix,
        cuda=True,
    )

    np.testing.assert_allclose(gpu_x, cpu_x, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(gpu_y, cpu_y, rtol=1e-5, atol=1e-3)


# %% test globdet_channel


def test_globdet_channel():
    from smlmlp.modules.block_LP._functions.globdetection.globdet_channel import (
        globdet_channel,
    )

    channels = [
        np.ones((2, 6, 8), dtype=np.float32),
        np.ones((2, 4, 10), dtype=np.float32) * 3,
    ]

    global_channels, info = globdet_channel(channels, mode="mean")

    assert len(global_channels) == 1
    assert global_channels[0].shape == (2, 4, 8)
    assert len(info["transform_matrices"]) == 2
    assert info["crop_shape"] == (4, 8)


def test_globdet_channel_std_mode():
    from smlmlp.modules.block_LP._functions.globdetection.globdet_channel import (
        globdet_channel,
    )

    channels = [
        np.ones((2, 6, 8), dtype=np.float32),
        np.ones((2, 4, 10), dtype=np.float32) * 3,
    ]

    global_channels, info = globdet_channel(channels, mode="std")

    assert len(global_channels) == 1
    np.testing.assert_allclose(global_channels[0], 1.0)


def test_globdet_channel_with_shifts():
    from smlmlp.modules.block_LP._functions.globdetection.globdet_channel import (
        globdet_channel,
    )

    channels = [
        np.ones((2, 6, 8), dtype=np.float32),
        np.ones((2, 4, 10), dtype=np.float32) * 3,
    ]

    global_channels, info = globdet_channel(
        channels,
        channels_x_shifts_nm=[1.0, 0.0],
    )

    assert info["crop_shape"][1] < 8


if __name__ == "__main__":
    from corelp import test
    test(__file__)
