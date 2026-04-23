#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class dyes(DataFrame) :
    '''
    Dyes dataframe
    '''

    @column(headers=['dye'], save=True, agg='min', index="blinks")
    def dye(self:np.uint8) :
        if self.locs.config.ndyes == 1 :
            return np.ones(self.locs.ndetections, dtype=np.uint8)
        else :
            from smlmlp import index_dyes
            return index_dyes(locs=self.locs)[0]