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


def test_columns_registry_defaults() :
    """Columns registries expose column and header mappings."""
    registry = Columns(example="value")

    assert registry["example"] == "value"
    assert registry.headers == {}
    assert registry.head2save == []


def test_global_columns_contains_registered_descriptors() :
    """Import side effects register dataframe column descriptors."""
    from smlmlp import dataframes

    assert "det" in columns
    assert columns.headers["detection"] is columns["det"]
    assert "detections" in dataframes


if __name__ == "__main__":
    from corelp import test

    test(__file__)
