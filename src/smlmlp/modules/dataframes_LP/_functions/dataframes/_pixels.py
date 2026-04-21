#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class pixels(DataFrame) :
    '''
    Pixels dataframe
    '''

    @column(headers=['pixel'], save=True, index=True, agg='min')
    def pix(self:np.uint32) :
        return np.round(self.y / self.y_pixel) * self.x_shape + np.round(self.x / self.x_pixel)