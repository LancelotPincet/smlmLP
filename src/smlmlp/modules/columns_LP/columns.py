#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-23
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : columns
"""
This is a dict of all the columns that are created.

The Columns class extends dict to provide a registry for dataframe column
descriptors with additional metadata for headers and save order.
"""

# %% Class


class Columns(dict):
    """
    Dictionary subclass for managing dataframe column descriptors.

    Attributes
    ----------
    headers : dict
        Mapping of column header names to column descriptors.
    head2save : list
        Ordered list of column headers to save.

    Examples
    --------
    >>> cols = Columns()
    >>> cols["mycol"] = "value"
    >>> cols.headers["myheader"] = cols["mycol"]
    """

    def __init__(self, *args, **kwargs):
        """Initialize the columns dictionary with headers and save list."""
        super().__init__(*args, **kwargs)
        self.headers = {}
        self.head2save = []


columns = Columns()


if __name__ == "__main__":
    from corelp import test

    test(__file__)