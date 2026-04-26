#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Locs

"""
This file allows to test Locs

Locs : This class define objects corresponding to localizations sets for one experiment.
"""



import numpy as np
import pandas as pd
import pytest

from smlmlp import Locs


def test_locs_reuses_existing_instance() :
    """Passing a Locs instance returns the same object."""
    locs = Locs()

    assert Locs(locs) is locs


def test_locs_times_aliases_time_cache() :
    """The analysis decorator timing alias points to the Locs time cache."""
    locs = Locs()

    locs.times["step"] = 1.5

    assert locs.time["step"] == 1.5


def test_locs_opens_dataframe_and_assigns_columns() :
    """A dataframe source populates the detections dataframe."""
    source = pd.DataFrame(
        {
            "frame": [1, 1, 2],
            "x detection [nm]": [10.0, 20.0, 30.0],
            "y detection [nm]": [15.0, 25.0, 35.0],
        }
    )

    locs = Locs(source)

    np.testing.assert_array_equal(locs.detections.fr, np.array([1, 1, 2], dtype=np.uint32))
    np.testing.assert_allclose(locs.detections.x_det, [10.0, 20.0, 30.0])
    assert locs.ndetections == 3


def test_locs_open_skips_unknown_columns_with_warning() :
    """Unknown dataframe headers are warned about and ignored."""
    source = pd.DataFrame({"frame": [1], "unknown": [99]})

    with pytest.warns(UserWarning, match="Skipping opening unknown column") :
        locs = Locs(source)

    assert locs.ndetections == 1
    assert "unknown" not in locs.detections.columns


def test_locs_filter_returns_filtered_locs() :
    """Filtering keeps matching detections and removes the temporary filter column."""
    locs = Locs()
    locs.detections.fr = np.array([1, 1, 2], dtype=np.uint32)
    locs.detections.x_det = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    filtered = locs.filter(mask=locs.detections.x_det > 15)

    np.testing.assert_allclose(filtered.detections.x_det, [20.0, 30.0])
    assert "filter" not in filtered.detections.columns


def test_locs_filter_uses_physical_index_columns_for_children() :
    """Filtering child dataframes uses physical ids without lazy columns."""
    detections = pd.DataFrame(
        {"frame": [1, 2], "channel": [1, 2], "filter": [True, True]}
    )
    frames = pd.DataFrame(index=pd.Index([1, 2], name="frame"))
    locs = Locs([detections, frames])
    locs.config.cameras[0].nchannels = 2

    filtered = locs.filter(mask=np.array([True, False]))

    assert len(filtered.detections) == 1
    assert filtered.frames.index.tolist() == [1]
    assert "filter" not in filtered.detections.columns


def test_locs_combine_adds_channel_labels() :
    """Combining localization sets concatenates detections and labels sources."""
    first = Locs()
    first.detections.fr = np.array([1], dtype=np.uint32)
    first.detections.x_det = np.array([10.0], dtype=np.float32)

    second = Locs()
    second.detections.fr = np.array([2, 3], dtype=np.uint32)
    second.detections.x_det = np.array([20.0, 30.0], dtype=np.float32)

    combined = first.combine(second)

    assert combined.ndetections == 3
    np.testing.assert_array_equal(combined.detections.ch, [1, 2, 2])


if __name__ == "__main__":
    from corelp import test

    test(__file__)
