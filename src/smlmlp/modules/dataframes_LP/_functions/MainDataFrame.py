#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd



# %% Function
class MainDataFrame(pd.DataFrame) : # This class is here to host dynamic main dataframe columns functions
    '''
    Localization main dataframe
    '''

    def __init__(self, locs) :
        if self.index_name is None : raise SyntaxError(f'DataFrame {self.__class__} should have an index name defined via @column decorator')
        self.locs = locs
        # Empty frame with a 1-based RangeIndex (length 0). Pandas resets to 0..n-1 on the first
        # column assignment; __setitem__ then rebases to 1..n when appropriate.
        pd.DataFrame.__init__(self, index=pd.RangeIndex(start=1, stop=1, step=1))
        self.index.name = self.index_name

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
            self.index.name = self.index_name
    
    # Attributes
    index_name = None # raise error if stays None
    locs = None # overriden in __init__
    head2save = [] # detections dataframe head2save

    def __setattr__(self, name, value):
        cls = type(self)
        attr = getattr(cls, name, None)

        # If class attribute is a descriptor with __set__
        if hasattr(attr, "__set__"):
            return attr.__set__(self, value)

        # Otherwise fallback to pandas
        return super().__setattr__(name, value)