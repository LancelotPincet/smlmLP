#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

import pandas as pd



class BaseDataFrame(pd.DataFrame) :
    """
    Base class for localization dataframes.

    Parameters
    ----------
    locs : Locs
        Parent localization container.
    """

    def __init__(self, locs) :
        """Initialize the object."""
        if self.index_header is None : raise SyntaxError(f'DataFrame {self.__class__} should have an index name defined via @column decorator')
        self.locs = locs

    def __setitem__(self, key, value) :
        """Implement __setitem__."""
        super().__setitem__(key, value)
        self._rebase_default_index_to_one()

    def _rebase_default_index_to_one(self) :
        """Rebase a default pandas RangeIndex from zero-based to one-based."""
        idx = self.index
        n = len(self)
        if n == 0 or not isinstance(idx, pd.RangeIndex) or idx.step != 1 :
            return
        if idx.start == 0 and idx.stop == n :
            self.index = pd.RangeIndex(start=1, stop=n + 1, step=1)
            self.index.name = self.index_header
    
    index_header = None
    locs = None

    def __setattr__(self, name, value):
        """Route descriptor assignment before falling back to pandas."""
        cls = type(self)
        attr = getattr(cls, name, None)

        if hasattr(attr, "__set__"):
            return attr.__set__(self, value)

        return super().__setattr__(name, value)
