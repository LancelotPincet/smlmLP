#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class elements(DataFrame) :
    '''
    Elements dataframe
    '''

    @column(headers=['element'], save=True, agg='min', index="blinks")
    def elm(self:np.uint16) :
        from smlmlp import index_elements
        return index_elements(locs=self.locs)[0]