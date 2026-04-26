#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : analysis

"""
Test all @analysis-decorated functions in analysis_LP/_functions/.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from smlmlp import analysis


class _Config:
    scale = 3.0
    cuda = True
    parallel = False


# %% test analysis decorator


def test_analysis_decorator_reads_locs_defaults():
    """Analysis functions read missing arguments from Locs data and config."""

    @analysis(df_name="points")
    def scaled(x, *, scale=1.0, cuda=False, parallel=False):
        """Return scaled."""
        info = {}
        return x * scale, cuda, parallel, info

    locs = SimpleNamespace(
        points=SimpleNamespace(x=np.array([1.0, 2.0], dtype=np.float32)),
        config=SimpleNamespace(scale=3.0, cuda=True, parallel=False),
        printer=None,
        times={},
    )

    values, cuda, parallel, info = scaled(locs=locs)

    np.testing.assert_allclose(values, [3.0, 6.0])
    assert cuda is True
    assert parallel is False
    assert info == {}
    assert "scaled" in locs.times


def test_analysis_decorator_without_locs():
    """Analysis functions work without a Locs object."""

    @analysis(df_name="points")
    def scaled(x, *, scale=1.0, cuda=False, parallel=False):
        """Return scaled."""
        info = {}
        return x * scale, cuda, parallel, info

    values, cuda, parallel, info = scaled(np.array([1.0, 2.0], dtype=np.float32))

    np.testing.assert_allclose(values, [1.0, 2.0])
    assert info == {}


def test_analysis_decorator_generator():
    """Analysis decorator wraps generator functions."""

    @analysis(df_name="points")
    def gen(x, *, cuda=False, parallel=False):
        """Yield values."""
        for val in x:
            yield val * 2, cuda, parallel, {}

    locs = SimpleNamespace(
        points=SimpleNamespace(x=np.array([1.0, 2.0], dtype=np.float32)),
        config=SimpleNamespace(cuda=False, parallel=False),
        printer=None,
        times={},
    )

    results = list(gen(locs=locs))

    assert len(results) == 2
    np.testing.assert_allclose(results[0][0], 2.0)
    np.testing.assert_allclose(results[1][0], 4.0)


def test_analysis_explicit_kwargs_do_not_probe_dataframe():
    """Explicit arguments are used before lazy dataframe lookup."""

    @analysis(df_name="detections")
    def scaled(x, *, scale=1.0, cuda=False, parallel=False):
        """Return scaled."""
        return x * scale, cuda, parallel, {}

    class ExplodingDetections:
        @property
        def x(self):
            raise AssertionError("x should not be read")

    locs = SimpleNamespace(
        detections=ExplodingDetections(),
        config=SimpleNamespace(scale=3.0, cuda=False, parallel=False),
        printer=None,
        times={},
    )

    values, _, _, _ = scaled(locs=locs, x=np.array([1.0, 2.0], dtype=np.float32))

    np.testing.assert_allclose(values, [3.0, 6.0])


# %% test aggregate_flux


def test_aggregate_flux():
    from smlmlp.modules.analysis_LP._functions.aggregate.aggregate_flux import (
        aggregate_flux,
    )

    flux, switching, info = aggregate_flux(
        np.array([10.0, 20.0, 30.0, 5.0, 7.0], dtype=np.float32),
        np.array([1, 1, 1, 2, 2], dtype=np.uint64),
        np.array([1, 2, 3, 1, 2], dtype=np.uint32),
        parallel=False,
    )

    np.testing.assert_allclose(flux, [20.0, np.nan], equal_nan=True)
    np.testing.assert_array_equal(switching, [True, False, True, True, True])
    assert info == {}


def test_aggregate_flux_short_blink():
    from smlmlp.modules.analysis_LP._functions.aggregate.aggregate_flux import (
        aggregate_flux,
    )

    flux, switching, info = aggregate_flux(
        np.array([10.0, 20.0], dtype=np.float32),
        np.array([1, 1], dtype=np.uint64),
        np.array([1, 2], dtype=np.uint32),
        parallel=False,
    )

    np.testing.assert_allclose(flux, [np.nan], equal_nan=True)


# %% test aggregate_ratio


def test_aggregate_ratio():
    from smlmlp.modules.analysis_LP._functions.aggregate.aggregate_ratio import (
        aggregate_ratio,
    )

    ratio_x, ratio_y, info = aggregate_ratio(
        np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32),
        np.array([1, 1, 2, 2], dtype=np.uint64),
        np.array([1, 2, 1, 2], dtype=np.uint32),
        x_channels=[1],
        y_channels=[2],
        parallel=False,
    )

    np.testing.assert_allclose(ratio_x, [2.0, 5.0])
    np.testing.assert_allclose(ratio_y, [3.0, 7.0])
    assert info == {}


def test_aggregate_ratio_auto_channels():
    from smlmlp.modules.analysis_LP._functions.aggregate.aggregate_ratio import (
        aggregate_ratio,
    )

    col = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    pnt = np.array([1, 1, 2, 2], dtype=np.uint64)
    ch = np.array([1, 2, 3, 4], dtype=np.uint32)

    ratio_x, ratio_y, info = aggregate_ratio(col, pnt, ch, parallel=False)

    assert ratio_x.shape == ratio_y.shape == (2,)


# %% test lost_frames


def test_lost_frames():
    from smlmlp.modules.analysis_LP._functions.lost.lost_frames import lost_frames

    frame, info = lost_frames(np.array([1, 2, 0, 1, 0], dtype=np.uint16))

    np.testing.assert_array_equal(frame, [1, 1, 2, 2, 3])
    assert info == {}


def test_lost_frames_no_reset():
    from smlmlp.modules.analysis_LP._functions.lost.lost_frames import lost_frames

    frame, info = lost_frames(np.array([1, 2, 3, 4, 5], dtype=np.uint16))

    np.testing.assert_array_equal(frame, [1, 1, 1, 1, 1])


# %% test lost_channels


def test_lost_channels():
    from smlmlp.modules.analysis_LP._functions.lost.lost_channels import lost_channels

    channel, info = lost_channels(np.array([1, 2, 1, 2, 0], dtype=np.uint16))

    np.testing.assert_array_equal(channel, [1, 1, 2, 2, 3])
    assert info == {}


def test_lost_channels_no_reset():
    from smlmlp.modules.analysis_LP._functions.lost.lost_channels import lost_channels

    channel, info = lost_channels(np.array([1, 2, 3, 4, 5], dtype=np.uint16))

    np.testing.assert_array_equal(channel, [1, 1, 1, 1, 1])


# %% test transform_locs


def test_transform_locs():
    from smlmlp.modules.analysis_LP._functions.transform.transform_locs import (
        transform_locs,
    )

    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    ch = np.array([1, 2], dtype=np.int64)
    matrices = np.array(
        [
            [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, -5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    x_t, y_t, info = transform_locs(x, y, ch, matrices, parallel=False)

    np.testing.assert_allclose(x_t, [21.0, 9.0])
    np.testing.assert_allclose(y_t, [13.0, -1.0])
    assert info == {}


def test_transform_locs_single_matrix():
    from smlmlp.modules.analysis_LP._functions.transform.transform_locs import (
        transform_locs,
    )

    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    ch = np.array([1, 1], dtype=np.int64)
    matrix = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    x_t, y_t, info = transform_locs(x, y, ch, matrix[None, :, :], parallel=False)

    assert x_t.shape == y_t.shape == (2,)


# %% test inv_transform_locs


def test_inv_transform_locs():
    from smlmlp.modules.analysis_LP._functions.transform.inv_transform_locs import (
        inv_transform_locs,
    )

    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    ch = np.array([1, 2], dtype=np.int64)
    matrices = np.array(
        [
            [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, -5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    x_t, y_t, info = inv_transform_locs(x, y, ch, matrices, parallel=False)

    assert x_t.shape == y_t.shape == (2,)


# %% test associate_consecutive_frames


def test_associate_consecutive_frames():
    from smlmlp.modules.analysis_LP._functions.associate.associate_consecutive_frames import (
        associate_consecutive_frames,
    )

    xx = np.array([1.0, 2.0, 1.1, 2.1], dtype=np.float32)
    yy = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    fr = np.array([1, 1, 2, 2], dtype=np.uint32)

    tracks, info = associate_consecutive_frames(xx, yy, fr, parallel=False)

    assert tracks.ndim == 1
    assert "n_localizations" in info
    assert "n_tracks" in info


def test_associate_consecutive_frames_empty():
    from smlmlp.modules.analysis_LP._functions.associate.associate_consecutive_frames import (
        associate_consecutive_frames,
    )

    tracks, info = associate_consecutive_frames(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint32),
        parallel=False,
    )

    assert len(tracks) == 0
    assert info["n_localizations"] == 0


def test_associate_consecutive_frames_with_radius():
    from smlmlp.modules.analysis_LP._functions.associate.associate_consecutive_frames import (
        associate_consecutive_frames,
    )

    xx = np.array([1.0, 2.0, 1.1, 2.1], dtype=np.float32)
    yy = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    fr = np.array([1, 1, 2, 2], dtype=np.uint32)

    tracks, info = associate_consecutive_frames(
        xx, yy, fr, association_radius_nm=50.0, parallel=False
    )

    assert tracks.ndim == 1


# %% test associate_consecutive_frames_radius


def test_associate_consecutive_frames_radius():
    from smlmlp.modules.analysis_LP._functions.associate.associate_consecutive_frames_radius import (
        associate_consecutive_frames_radius,
    )

    xx = np.array([1.0, 2.0, 1.1, 2.1, 1.2, 2.2], dtype=np.float32)
    yy = np.array([1.0, 1.0, 2.0, 2.0, 1.05, 2.05], dtype=np.float32)
    fr = np.array([1, 1, 2, 2, 3, 3], dtype=np.uint32)

    radius, info = associate_consecutive_frames_radius(xx, yy, fr, parallel=False)

    assert isinstance(radius, float)
    assert "distances" in info
    assert "peaks" in info


# %% test associate_different_channels


def test_associate_different_channels():
    from smlmlp.modules.analysis_LP._functions.associate.associate_different_channels import (
        associate_different_channels,
    )

    x = np.array([1.0, 2.0, 1.1, 2.1], dtype=np.float32)
    y = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    fr = np.array([1, 1, 2, 2], dtype=np.uint32)
    ch = np.array([1, 2, 1, 2], dtype=np.uint8)

    point, info = associate_different_channels(
        x, y, fr, ch, association_radius_nm=50.0, parallel=False
    )

    assert point.ndim == 1
    assert "n_groups" in info
    assert "max_component_size" in info


def test_associate_different_channels_parallel_zero_is_serial():
    from smlmlp.modules.analysis_LP._functions.associate.associate_different_channels import (
        associate_different_channels,
    )

    point, _ = associate_different_channels(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1, 1], dtype=np.uint32),
        np.array([1, 2], dtype=np.uint8),
        association_radius_nm=50.0,
        parallel=0,
    )

    np.testing.assert_array_equal(point, [1, 1])


@pytest.mark.skip(reason="source file has bug with empty arrays")
def test_associate_different_channels_empty():
    from smlmlp.modules.analysis_LP._functions.associate.associate_different_channels import (
        associate_different_channels,
    )

    point, info = associate_different_channels(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.uint8),
        parallel=False,
    )

    assert len(point) == 0


# %% test placeholders that raise SyntaxError


def test_analysis_template_placeholder():
    from smlmlp.modules.analysis_LP._functions._analysis_template import (
        analysis_template,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        analysis_template(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_associate_density_placeholder():
    from smlmlp.modules.analysis_LP._functions.associate.associate_density import (
        associate_density,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        associate_density(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1, 2], dtype=np.uint32),
        )


def test_associate_molecules_placeholder():
    from smlmlp.modules.analysis_LP._functions.associate.associate_molecules import (
        associate_molecules,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        associate_molecules(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_calibration_convert_placeholder():
    from smlmlp.modules.analysis_LP._functions.calibration.calibration_convert import (
        calibration_convert,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        calibration_convert(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        )


def test_calibration_flim_placeholder():
    from smlmlp.modules.analysis_LP._functions.calibration.calibration_flim import (
        calibration_flim,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        calibration_flim(np.array([1.0], dtype=np.float32))


def test_calibration_fuse_placeholder():
    from smlmlp.modules.analysis_LP._functions.calibration.calibration_fuse import (
        calibration_fuse,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        calibration_fuse(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1, 2], dtype=np.uint64),
        )


def test_calibration_spheres_placeholder():
    from smlmlp.modules.analysis_LP._functions.calibration.calibration_spheres import (
        calibration_spheres,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        calibration_spheres(
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([1], dtype=np.uint64),
        )


def test_calibration_zstacks_placeholder():
    from smlmlp.modules.analysis_LP._functions.calibration.calibration_zstacks import (
        calibration_zstacks,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        calibration_zstacks(
            np.array([1.0], dtype=np.float32),
            np.array([2], dtype=np.uint32),
        )


def test_clustering_dbscan_placeholder():
    from smlmlp.modules.analysis_LP._functions.clustering.clustering_dbscan import (
        clustering_dbscan,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        clustering_dbscan(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_demix_histogram_placeholder():
    from smlmlp.modules.analysis_LP._functions.demix.demix_histogram import (
        demix_histogram,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        demix_histogram(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([2], dtype=np.uint8),
        )


def test_drift_aim_placeholder():
    from smlmlp.modules.analysis_LP._functions.drift.drift_aim import drift_aim

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        drift_aim(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_drift_comet_placeholder():
    from smlmlp.modules.analysis_LP._functions.drift.drift_comet import drift_comet

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        drift_comet(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_drift_crosscorr_placeholder():
    from smlmlp.modules.analysis_LP._functions.drift.drift_crosscorr import (
        drift_crosscorr,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        drift_crosscorr(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_drift_meanshift_placeholder():
    from smlmlp.modules.analysis_LP._functions.drift.drift_meanshift import (
        drift_meanshift,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        drift_meanshift(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_image_colmap_mean():
    from smlmlp.modules.analysis_LP._functions.image.image_colmap import image_colmap

    image, info = image_colmap(
        np.array([1.0, 3.0, 5.0], dtype=np.float32),
        np.array([0, 1, 1], dtype=np.uint32),
        shape=(1, 2),
    )

    np.testing.assert_allclose(image, [[1.0, 4.0]])
    assert info["n_used"] == 3


def test_image_picker_placeholder():
    from smlmlp.modules.analysis_LP._functions.image.image_picker import image_picker

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        image_picker(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_image_pixel_placeholder():
    from smlmlp.modules.analysis_LP._functions.image.image_pixel import image_pixel

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        image_pixel(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32),
            10.0,
        )


def test_image_smlm_gaussian():
    from smlmlp.modules.analysis_LP._functions.image.image_smlm import image_smlm

    image, info = image_smlm(
        np.array([20.0], dtype=np.float32),
        np.array([20.0], dtype=np.float32),
        image_sigma=5.0,
        shape=(5, 5),
        pixel_sr_nm=10.0,
        crop_sigma=5.0,
    )

    assert image.shape == (5, 5)
    assert image[2, 2] == np.max(image)
    assert np.isclose(np.sum(image), 1.0, atol=1e-4)
    assert info["pixel_sr_nm"] == 10.0


def test_image_smlm3d_planes():
    from smlmlp.modules.analysis_LP._functions.image.image_smlm3d import image_smlm3d

    volume, info = image_smlm3d(
        np.array([0.0, 10.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        crlb=0.0,
        shape=(1, 1),
        pixel_sr_nm=10.0,
        z_pixel=10.0,
    )

    assert volume.shape == (2, 1, 1)
    np.testing.assert_allclose(volume[:, 0, 0], [1.0, 1.0])
    assert info["z_shape"] == 2


def test_image_stackmap_placeholder():
    from smlmlp.modules.analysis_LP._functions.image.image_stackmap import image_stackmap

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        image_stackmap(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )


def test_image_vectors_rgb_orientation():
    from smlmlp.modules.analysis_LP._functions.image.image_vectors import image_vectors

    image, info = image_vectors(
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([10.0, 10.0], dtype=np.float32),
        np.array([10.0, 10.0], dtype=np.float32),
        np.array([0.0, 90.0], dtype=np.float32),
        shape=(3, 3),
        pixel_sr_nm=10.0,
        line_length_nm=20.0,
        line_width_nm=10.0,
        color_limits=(0.0, 1.0),
    )

    assert image.shape == (3, 3, 3)
    assert image[1, 0, 2] > image[1, 0, 0]
    assert image[0, 1, 0] > image[0, 1, 2]
    assert info["color_limits"] == (0.0, 1.0)


def test_metric_frc_placeholder():
    from smlmlp.modules.analysis_LP._functions.metric.metric_frc import metric_frc

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        metric_frc(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1, 2], dtype=np.uint32),
        )


def test_metric_nena_placeholder():
    from smlmlp.modules.analysis_LP._functions.metric.metric_nena import metric_nena

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        metric_nena(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1, 2], dtype=np.uint64),
        )


def test_metric_overloc_placeholder():
    from smlmlp.modules.analysis_LP._functions.metric.metric_overloc import metric_overloc

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        metric_overloc(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1, 2], dtype=np.uint64),
        )


def test_metric_photophysics_placeholder():
    from smlmlp.modules.analysis_LP._functions.metric.metric_photophysics import (
        metric_photophysics,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        metric_photophysics(np.array([1.0, 2.0], dtype=np.float32))


def test_metric_squirrel_linear_fit():
    from smlmlp.modules.analysis_LP._functions.metric.metric_squirrel import metric_squirrel

    sr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    widefield = 2.0 * sr + 1.0

    error_map, info = metric_squirrel(widefield, sr, ignore_zero=False)

    np.testing.assert_allclose(error_map, 0.0, atol=1e-6)
    assert np.isclose(info["scale"], 2.0)
    assert np.isclose(info["offset"], 1.0)
    assert np.isclose(info["rse"], 0.0, atol=1e-6)
    assert np.isclose(info["rsp"], 1.0)


def test_modloc_axial_placeholder():
    from smlmlp.modules.analysis_LP._functions.modloc.modloc_axial import modloc_axial

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        modloc_axial(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )


def test_modloc_demodulated_placeholder():
    from smlmlp.modules.analysis_LP._functions.modloc.modloc_demodulated import (
        modloc_demodulated,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        modloc_demodulated(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1, 1], dtype=np.uint64),
            np.array([1, 2], dtype=np.uint32),
        )


def test_modloc_sequential_placeholder():
    from smlmlp.modules.analysis_LP._functions.modloc.modloc_sequential import (
        modloc_sequential,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        modloc_sequential(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1, 2], dtype=np.uint64),
            np.array([1, 2], dtype=np.uint32),
        )


def test_modloc_transverse_placeholder():
    from smlmlp.modules.analysis_LP._functions.modloc.modloc_transverse import (
        modloc_transverse,
    )

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        modloc_transverse(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )


def test_orient_polar2d_placeholder():
    from smlmlp.modules.analysis_LP._functions.orient.orient_polar2d import orient_polar2d

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        orient_polar2d(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([100.0, 200.0], dtype=np.float32),
        )


def test_orient_polar3d_placeholder():
    from smlmlp.modules.analysis_LP._functions.orient.orient_polar3d import orient_polar3d

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        orient_polar3d(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0], dtype=np.float32),
        )


if __name__ == "__main__":
    from corelp import test
    test(__file__)
