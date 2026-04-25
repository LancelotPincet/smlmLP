#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-21
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : analysis

"""
This file allows to test analysis

analysis : This decorator defines a function as an analysis on a Locs object.
"""

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from smlmlp import analysis
from smlmlp.modules.analysis_LP._functions.aggregate.aggregate_flux import aggregate_flux
from smlmlp.modules.analysis_LP._functions.aggregate.aggregate_ratio import aggregate_ratio
from smlmlp.modules.analysis_LP._functions.lost.lost_channels import lost_channels
from smlmlp.modules.analysis_LP._functions.lost.lost_frames import lost_frames
from smlmlp.modules.analysis_LP._functions.transform.inv_transform_loc import inv_transform_locs
from smlmlp.modules.analysis_LP._functions.transform.transform_locs import transform_locs


PLACEHOLDERS = [
    ("smlmlp.modules.analysis_LP._functions._analysis_template", "analysis_template"),
    ("smlmlp.modules.analysis_LP._functions.associate.associate_density", "associate_density"),
    ("smlmlp.modules.analysis_LP._functions.associate.associate_molecules", "associate_molecules"),
    ("smlmlp.modules.analysis_LP._functions.calibration.calibration_convert", "calibration_convert"),
    ("smlmlp.modules.analysis_LP._functions.calibration.calibration_flim", "calibration_flim"),
    ("smlmlp.modules.analysis_LP._functions.calibration.calibration_fuse", "calibration_fuse"),
    ("smlmlp.modules.analysis_LP._functions.calibration.calibration_spheres", "calibration_spheres"),
    ("smlmlp.modules.analysis_LP._functions.calibration.calibration_zstacks", "calibration_zstack"),
    ("smlmlp.modules.analysis_LP._functions.clustering.clustering_dbscan", "clustering_dbscan"),
    ("smlmlp.modules.analysis_LP._functions.demix.demix_histogram", "demix_histogram"),
    ("smlmlp.modules.analysis_LP._functions.drift.drift_aim", "drift_aim"),
    ("smlmlp.modules.analysis_LP._functions.drift.drift_comet", "drift_comet"),
    ("smlmlp.modules.analysis_LP._functions.drift.drift_crosscorr", "drift_crosscorr"),
    ("smlmlp.modules.analysis_LP._functions.drift.drift_meanshift", "drift_meanshift"),
    ("smlmlp.modules.analysis_LP._functions.image.image_colmap", "image_colmap"),
    ("smlmlp.modules.analysis_LP._functions.image.image_picker", "image_picker"),
    ("smlmlp.modules.analysis_LP._functions.image.image_pixel", "image_pixel"),
    ("smlmlp.modules.analysis_LP._functions.image.image_smlm", "image_smlm"),
    ("smlmlp.modules.analysis_LP._functions.image.image_smlm3d", "image_smlm3d"),
    ("smlmlp.modules.analysis_LP._functions.image.image_stackmap", "image_stackmap"),
    ("smlmlp.modules.analysis_LP._functions.image.image_vectors", "image_vectors"),
    ("smlmlp.modules.analysis_LP._functions.metric.metric_frc", "metric_frc"),
    ("smlmlp.modules.analysis_LP._functions.metric.metric_nena", "metric_nena"),
    ("smlmlp.modules.analysis_LP._functions.metric.metric_overloc", "metric_overloc"),
    ("smlmlp.modules.analysis_LP._functions.metric.metric_photophysics", "metric_photophysics"),
    ("smlmlp.modules.analysis_LP._functions.metric.metric_squirrel", "metric_squirrel"),
    ("smlmlp.modules.analysis_LP._functions.modloc.modloc_axial", "modloc_axial"),
    ("smlmlp.modules.analysis_LP._functions.modloc.modloc_demodulated", "modloc_demodulated"),
    ("smlmlp.modules.analysis_LP._functions.modloc.modloc_sequential", "modloc_sequential"),
    ("smlmlp.modules.analysis_LP._functions.modloc.modloc_transverse", "modloc_transverse"),
    ("smlmlp.modules.analysis_LP._functions.orient.orient_polar2d", "orient_polar2d"),
    ("smlmlp.modules.analysis_LP._functions.orient.orient_polar3d", "orient_polar3d"),
]


def _placeholder_call(function):
    """Build minimal call arguments for a placeholder analysis."""
    args = []
    kwargs = {}
    value = np.array([1.0, 2.0], dtype=np.float32)

    for name, parameter in inspect.signature(function).parameters.items():
        if name in {"cuda", "parallel"} or parameter.default is not inspect._empty:
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value

    return args, kwargs


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


def test_aggregate_flux_returns_flux_switching_and_info():
    """Aggregate flux keeps middle frames and marks switching frames."""
    flux, switching, info = aggregate_flux(
        np.array([10.0, 20.0, 30.0, 5.0, 7.0]),
        np.array([1, 1, 1, 2, 2]),
        np.array([1, 2, 3, 1, 2]),
        parallel=False,
    )

    np.testing.assert_allclose(flux, [20.0, np.nan], equal_nan=True)
    np.testing.assert_array_equal(switching, [True, False, True, True, True])
    assert info == {}


def test_aggregate_ratio_returns_ratios_and_info():
    """Aggregate ratio sums configured channel groups per point."""
    ratio_x, ratio_y, info = aggregate_ratio(
        np.array([2.0, 3.0, 5.0, 7.0]),
        np.array([1, 1, 2, 2]),
        np.array([1, 2, 1, 2]),
        x_channels=[1],
        y_channels=[2],
        parallel=False,
    )

    np.testing.assert_allclose(ratio_x, [2.0, 5.0])
    np.testing.assert_allclose(ratio_y, [3.0, 7.0])
    assert info == {}


def test_lost_frames_returns_frame_and_info():
    """Lost frame inference returns the standard analysis tuple."""
    frame, info = lost_frames(np.array([1, 2, 0, 1, 0], dtype=np.uint16))

    np.testing.assert_array_equal(frame, [1, 1, 2, 2, 3])
    assert info == {}


def test_lost_channels_returns_channel_and_info():
    """Lost channel inference increments when frame coordinates reset."""
    channel, info = lost_channels(np.array([1, 2, 1, 2, 0], dtype=np.uint16))

    np.testing.assert_array_equal(channel, [1, 1, 2, 2, 3])
    assert info == {}


def test_transform_locs_round_trip_uses_channel_matrices():
    """Forward and inverse transforms round-trip per-channel coordinates."""
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    ch = np.array([1, 2], dtype=np.uint8)
    matrices = np.array(
        [
            [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, -5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    x_t, y_t, info = transform_locs(x, y, ch, matrices, parallel=False)
    x_back, y_back, inv_info = inv_transform_locs(x_t, y_t, ch, matrices, parallel=False)

    np.testing.assert_allclose(x_t, [21.0, 9.0])
    np.testing.assert_allclose(y_t, [13.0, -1.0])
    np.testing.assert_allclose(x_back, x)
    np.testing.assert_allclose(y_back, y)
    assert info == inv_info == {}


@pytest.mark.parametrize(("module_name", "function_name"), PLACEHOLDERS)
def test_unimplemented_placeholders_raise_syntax_error(module_name, function_name):
    """Unimplemented analyses raise the standardized placeholder error."""
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    args, kwargs = _placeholder_call(function)

    with pytest.raises(SyntaxError, match="Not implemented yet"):
        function(*args, **kwargs)
