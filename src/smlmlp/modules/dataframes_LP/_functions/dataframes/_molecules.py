#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class molecules(DataFrame) :
    '''
    Molecules dataframe
    '''

    @column(headers=['molecule'], save=True, agg='min', index="blinks")
    def mol(self:np.uint64) :
        from smlmlp import index_molecules
        return index_molecules(locs=self.locs)[0]



    # --- Photophysics ---

    @column(headers=['off time [ms]'], save=True, agg='mean')
    def off_time(self:np.float32) :
        unique, counts = np.unique(self.locs.blinks.mol, return_counts=True)
        if unique[0] == 0 : unique, counts = unique[1:], counts[1:]
        return counts * self.locs.config.exposure_ms
