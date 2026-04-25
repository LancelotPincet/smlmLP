#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd



# %% Function
class BaseDataFrame(pd.DataFrame) :
    '''
    Localization base dataframe
    '''

    def __init__(self, locs) :
        if self.index_header is None : raise SyntaxError(f'DataFrame {self.__class__} should have an index name defined via @column decorator')
        self.locs = locs

    def __setitem__(self, key, value) :
        super().__setitem__(key, value)
        self._rebase_default_index_to_one()

    def _rebase_default_index_to_one(self) :
        '''If pandas used a default 0..n-1 RangeIndex, replace it with 1..n.'''
        idx = self.index
        n = len(self)
        if n == 0 or not isinstance(idx, pd.RangeIndex) or idx.step != 1 :
            return
        if idx.start == 0 and idx.stop == n :
            self.index = pd.RangeIndex(start=1, stop=n + 1, step=1)
            self.index.name = self.index_header
    
    # Attributes
    index_header = None # raise error if stays None
    locs = None # overriden in __init__

    def __setattr__(self, name, value):
        cls = type(self)
        attr = getattr(cls, name, None)

        # If class attribute is a descriptor with __set__
        if hasattr(attr, "__set__"):
            return attr.__set__(self, value)

        # Otherwise fallback to pandas
        return super().__setattr__(name, value)
