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



class Columns(dict) :
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.headers = {} # header: column object
        self.head2save = [] # headers to save
columns = Columns()


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)