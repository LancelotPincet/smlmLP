#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class points(DataFrame) :
    '''
    Points dataframe
    '''

    @column(headers=['point'], save=True, index=True, agg='min')
    def pnt(self:np.uint64) :
        if self.locs.detections.ch is None :
            return np.arange(1, ).
        return None