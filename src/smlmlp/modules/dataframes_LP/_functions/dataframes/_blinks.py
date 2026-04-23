#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class blinks(DataFrame) :
    '''
    Blinks dataframe
    '''

    @column(headers=['blink'], save=True, agg='min', index="points")
    def blk(self:np.uint64) :
        from smlmlp import index_blinks
        return index_blinks(locs=self.locs)[0]



    # --- Photophysics ---

    @column(headers=['on time [ms]'], save=True, agg='mean')
    def on_time(self:np.float32) :
        unique, counts = np.unique(self.locs.points.blk, return_counts=True)
        if unique[0] == 0 : unique, counts = unique[1:], counts[1:]
        return counts * self.locs.config.exposure_ms

    @column(headers=['flux [photons]'], save=True, agg='mean')
    def flux(self:np.float32) :
        from smlmlp import demix_flux
        return demix_flux(locs=self.locs)[0]


