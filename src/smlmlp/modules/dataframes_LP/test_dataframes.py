#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-23
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : dataframes

"""
This file allows to test dataframes

dataframes : This is a dict of all the dataframes that are created.
"""
from types import SimpleNamespace

import numpy as np

from smlmlp import dataframes


def test_dataframes_registry_contains_core_dataframes() :
    """The dataframe registry imports core dataframe classes."""
    assert {"detections", "frames", "points"}.issubset(dataframes)
    assert dataframes["detections"].index_header == "detection"
    assert dataframes["frames"].index_header == "frame"


def test_dataframe_columns_are_registered_on_classes() :
    """Dataframe classes expose descriptor metadata after import."""
    detections = dataframes["detections"]

    assert detections.columns_dict["det"].header == "detection"
    assert "detection" in detections.head2save


def test_detection_intensity_alias_targets_existing_column() :
    """Intensity falls back to the registered Gaussian signal column."""
    detections = dataframes["detections"]

    assert detections.columns_dict["intensity"].func(object()) == "gaussian_signal"
    assert "gaussian_signal" in detections.columns_dict


def test_point_tilt_alias_targets_tilt_column() :
    """Configured polar3d tilt resolves to the tilt column, not azimuth."""
    points = dataframes["points"]
    fake = SimpleNamespace(locs=SimpleNamespace(config=SimpleNamespace(tilt_method="polar3d")))

    assert points.columns_dict["tilt"].func(fake) == "tilt_polar3d"


def test_channel_crop_shape_uses_y_and_x_indices() :
    """Channel crop shape descriptors keep y/x dimensions distinct."""
    channels = dataframes["channels"]
    fake = SimpleNamespace(
        ch=[1, 2],
        locs=SimpleNamespace(config=SimpleNamespace(channels_crops_pix=[(7, 9), (11, 13)])),
    )

    assert channels.columns_dict["x_cropshape"].func(fake).tolist() == [9.0, 13.0]
    assert channels.columns_dict["y_cropshape"].func(fake).tolist() == [7.0, 11.0]


def test_blink_off_time_uses_frame_column() :
    """Blink off-time derives gaps from the registered frame column."""
    blinks = dataframes["blinks"]
    fake = SimpleNamespace(
        fr=np.array([1.0, 4.0, 2.0], dtype=np.float32),
        mol=np.array([1, 1, 2], dtype=np.uint64),
        on_time=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        locs=SimpleNamespace(config=SimpleNamespace(exposure_ms=10.0)),
    )

    off_time = blinks.columns_dict["off_time"].func(fake)

    np.testing.assert_allclose(off_time, [29.0, np.nan, np.nan], equal_nan=True)


if __name__ == "__main__":
    from corelp import test

    test(__file__)
