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

    @column(headers=['element'], dtype=np.uint16, save=True, agg='min', index="blinks")
    def elm(self) :
        from smlmlp import index_elements
        return index_elements(locs=self.locs)[0]



    # Fusing

    @column(headers=['element x centroid [nm]'], dtype=np.float32, save=False, agg='mean')
    def elm_x0(self) :
        return "x"

    @column(headers=['element y centroid [nm]'], dtype=np.float32, save=False, agg='mean')
    def elm_y0(self) :
        return "y"
