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
        if self.index_name is None : raise SyntaxError(f'DataFrame {self.__class__} should have an index name defined via @column decorator')
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
            self.index.name = self.index_name
    
    # Attributes
    index_name = None # raise error if stays None
    locs = None # overriden in __init__

    def __setattr__(self, name, value):
        cls = type(self)
        attr = getattr(cls, name, None)

        # If class attribute is a descriptor with __set__
        if hasattr(attr, "__set__"):
            return attr.__set__(self, value)

        # Otherwise fallback to pandas
        return super().__setattr__(name, value)

    # Column names

    col_name = None # string defining dynamic column name
    @property
    def col(self) :
        if self.col_name is None :
            raise SyntaxError('Cannot get dynamic col without defining "col_name" in function call')
        return getattr(self, self.col_name)
    @col.setter
    def col(self, value) :
        if self.col_name is None :
            raise SyntaxError('Cannot set dynamic col without defining "col_name" in function call')
        setattr(self, self.col_name, value)
    
    vecx_name = None # string defining dynamic column name
    @property
    def vecx(self) :
        if self.vecx_name is None :
            raise SyntaxError('Cannot get dynamic vecx without defining "vecx_name" in function call')
        return getattr(self, self.vecx_name)
    @vecx.setter
    def vecx(self, value) :
        if self.vecx_name is None :
            raise SyntaxError('Cannot set dynamic vecx without defining "vecx_name" in function call')
        setattr(self, self.vecx_name, value)

    vecy_name = None # string defining dynamic column name
    @property
    def vecy(self) :
        if self.vecy_name is None :
            raise SyntaxError('Cannot get dynamic vecy without defining "vecy_name" in function call')
        return getattr(self, self.vecy_name)
    @vecy.setter
    def vecy(self, value) :
        if self.vecy_name is None :
            raise SyntaxError('Cannot set dynamic vecy without defining "vecy_name" in function call')
        setattr(self, self.vecy_name, value)
