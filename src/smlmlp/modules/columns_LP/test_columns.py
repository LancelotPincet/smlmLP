#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-23
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : columns

"""
This file allows to test columns

columns : This is a dict of all the columns that are created.
"""

from smlmlp import columns
from smlmlp.modules.columns_LP.columns import Columns


# %% test Columns class


def test_columns_registry_defaults():
    """Columns registries expose column and header mappings."""
    registry = Columns(example="value")

    assert registry["example"] == "value"
    assert registry.headers == {}
    assert registry.head2save == []


def test_global_columns_contains_registered_descriptors():
    """Import side effects register dataframe column descriptors."""
    from smlmlp import dataframes

    assert "det" in columns
    assert columns.headers["detection"] is columns["det"]
    assert "detections" in dataframes


def test_columns_HEAD2save_tracks_ordered_headers():
    """The head2save list maintains column save order."""
    registry = Columns()
    registry.headers["col1"] = "desc1"
    registry.headers["col2"] = "desc2"
    registry.head2save = ["col1", "col2"]

    assert registry.head2save == ["col1", "col2"]


if __name__ == "__main__":
    from corelp import test

    test(__file__)