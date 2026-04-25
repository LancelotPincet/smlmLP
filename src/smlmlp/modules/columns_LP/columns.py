#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-23
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : columns

"""
This is a dict of all the columns that are created.
"""

# %% Class
class Columns(dict) :
    """This is a dict of all the columns that are created."""

    def __init__(self, *args, **kwargs) :
        """Initialize the object."""
        super().__init__(*args, **kwargs)
        self.headers = {}
        self.head2save = []


columns = Columns()


if __name__ == "__main__":
    from corelp import test

    test(__file__)
