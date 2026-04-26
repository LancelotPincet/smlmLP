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
import pytest

from smlmlp import Locs, dataframes


# %% test dataframe registry


def test_dataframes_registry_contains_core_dataframes():
    """The dataframe registry imports core dataframe classes."""
    assert {"detections", "frames", "points"}.issubset(dataframes)
    assert dataframes["detections"].index_header == "detection"
    assert dataframes["frames"].index_header == "frame"


def test_dataframe_columns_are_registered_on_classes():
    """Dataframe classes expose descriptor metadata after import."""
    detections = dataframes["detections"]

    assert detections.columns_dict["det"].header == "detection"
    assert "detection" in detections.head2save
    assert "filter" not in detections.head2save
    assert "x [nm]" not in dataframes["points"].head2save


def test_column_exists_properties_are_installed_on_all_dataframe_classes():
    """All dataframe classes expose all dynamic column-exists properties."""
    assert hasattr(dataframes["detections"], "x_exists")
    assert hasattr(dataframes["points"], "x_det_exists")
    assert hasattr(dataframes["frames"], "x_exists")


def test_column_exists_properties_check_corresponding_dataframes():
    """Column-exists properties inspect the physical owning dataframe."""
    locs = Locs()

    assert locs.detections.det_exists is False
    assert locs.detections.x_det_exists is False
    assert locs.detections.x_exists is False
    assert "points" not in locs.df_dict

    locs.detections.x_det = np.array([10.0, 20.0], dtype=np.float32)
    locs.detections.pnt = np.array([1, 2], dtype=np.uint64)
    points = locs.points
    points.x = np.array([10.0, 20.0], dtype=np.float32)

    assert locs.detections.x_det_exists is True
    assert points.x_det_exists is True
    assert locs.detections.pnt_exists is True
    assert points.pnt_exists is True
    assert "point" not in points.columns
    assert locs.detections.x_exists is True
    assert points.x_exists is True


# %% test column aliases


def test_detection_intensity_alias_targets_existing_column():
    """Intensity falls back to the registered Gaussian intensity column."""
    detections = dataframes["detections"]

    assert detections.columns_dict["intensity"].func(object()) == "gaussian_intensity"
    assert "gaussian_intensity" in detections.columns_dict


def test_sigma_aliases_do_not_recurse_when_missing():
    """Missing sigma aliases resolve to None instead of looping."""
    locs = Locs()

    assert locs.detections.sigma is None


def test_sigma_aliases_use_existing_fit_columns():
    """Sigma aliases use physical fit columns when available."""
    locs = Locs()
    locs.detections.sigma_fit = np.array([100.0, 121.0], dtype=np.float32)

    np.testing.assert_allclose(locs.detections.sigma, [100.0, 121.0])
    np.testing.assert_allclose(locs.detections.sigma_x, [100.0, 121.0])
    np.testing.assert_allclose(locs.detections.sigma_y, [100.0, 121.0])


def test_point_tilt_alias_targets_tilt_column():
    """Configured polar3d tilt resolves to the tilt column, not azimuth."""
    points = dataframes["points"]
    fake = SimpleNamespace(
        locs=SimpleNamespace(config=SimpleNamespace(tilt_method="polar3d"))
    )

    assert points.columns_dict["tilt"].func(fake) == "tilt_polar3d"


# %% test channel columns


def test_channel_crop_shape_uses_y_and_x_indices():
    """Channel crop shape descriptors keep y/x dimensions distinct."""
    channels = dataframes["channels"]
    fake = SimpleNamespace(
        ch=[1, 2],
        locs=SimpleNamespace(
            config=SimpleNamespace(channels_crops_pix=[(7, 9), (11, 13)])
        ),
    )

    assert channels.columns_dict["x_cropshape"].func(fake).tolist() == [
        9.0,
        13.0,
    ]
    assert channels.columns_dict["y_cropshape"].func(fake).tolist() == [
        7.0,
        11.0,
    ]


def test_empty_lazy_dataframes_are_empty():
    """Lazy dataframe creation handles empty detection index columns."""
    locs = Locs()

    assert len(locs.points) == 0
    assert len(locs.channels) == 0


def test_points_missing_xy_does_not_recurse():
    """Point association stops when aligned detection coordinates are missing."""
    locs = Locs()
    locs.config.cameras[0].nchannels = 2
    locs.detections.fr = np.array([1, 1], dtype=np.uint32)
    locs.detections.ch = np.array([1, 2], dtype=np.uint8)

    assert locs.points is None
    assert "points" not in locs.df_dict


def test_points_associate_from_detection_coordinates():
    """Point IDs are built from detection-level aligned coordinates."""
    locs = Locs()
    locs.config.cameras[0].nchannels = 2
    locs.detections.fr = np.array([1, 1, 2, 2], dtype=np.uint32)
    locs.detections.ch = np.array([1, 2, 1, 2], dtype=np.uint8)
    locs.detections.x_globfit = np.array([1.0, 2.0, 1.1, 2.1], dtype=np.float32)
    locs.detections.y_globfit = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(locs.detections.pnt, [1, 1, 2, 2])
    assert locs.points.index.tolist() == [1, 2]


def test_parent_aggregation_aligns_on_child_index():
    """Merged columns are reindexed to child rows instead of raw group order."""
    locs = Locs()
    locs.detections.pnt = np.array([1, 3], dtype=np.uint64)
    locs.detections.x_det = np.array([10.0, 30.0], dtype=np.float32)
    points = locs.points
    points.loc[2, "temporary"] = 0

    assert points.index.tolist() == [1, 3, 2]
    np.testing.assert_allclose(points.x_det, [10.0, 30.0, np.nan], equal_nan=True)


def test_index_columns_can_merge_from_parent_index():
    """Child dataframes can aggregate their parent's index values."""
    locs = Locs()
    locs.detections.pnt = np.array([1, 2], dtype=np.uint64)
    locs.detections.blk = np.array([1, 1], dtype=np.uint64)

    np.testing.assert_array_equal(locs.blinks.pnt, [1])


def test_spread_stops_on_missing_child_ids():
    """Spreading validates non-zero child ids before assigning detections."""
    locs = Locs()
    locs.detections.pnt = np.array([1, 2], dtype=np.uint64)
    points = locs.points
    points.drop(index=2, inplace=True)
    points.x = np.array([10.0], dtype=np.float32)

    assert locs.detections.x is None
    assert "x [nm]" not in locs.detections.columns


def test_dynamic_zernike_columns_are_registered_once():
    """Dynamic Zernike descriptors have unique names and set_name metadata."""
    locs = Locs()

    locs.detections.nzernike = 2
    locs.detections.nzernike = 2

    assert "zernike_01" in locs.detections.columns_dict
    assert "zernike 01" in locs.detections.head2save
    np.testing.assert_allclose(locs.detections.zernike_01, np.zeros(locs.ndetections))


# %% test blink columns


def test_blink_off_time_uses_frame_column():
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


# %% test sequence columns

# Note: sequence-specific column tests can be added if needed


if __name__ == "__main__":
    from corelp import test

    test(__file__)
