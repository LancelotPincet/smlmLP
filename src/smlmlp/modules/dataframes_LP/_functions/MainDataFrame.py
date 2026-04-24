#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import pandas as pd
from smlmlp import BaseDataFrame



# %% Function
class MainDataFrame(BaseDataFrame) : # This class is here to host dynamic main dataframe columns functions
    '''
    Localization main dataframe
    '''

    def __init__(self, locs) :
        super().__init__(locs)
        pd.DataFrame.__init__(self, index=pd.RangeIndex(start=1, stop=1, step=1))
        self.index.name = self.index_header

    # Attributes
    head2save = [] # detections dataframe head2save
