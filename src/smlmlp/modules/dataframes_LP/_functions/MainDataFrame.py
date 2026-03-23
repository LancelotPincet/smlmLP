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
        pd.DataFrame.__init__(self)
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