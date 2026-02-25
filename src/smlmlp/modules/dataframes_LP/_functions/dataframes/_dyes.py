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

    @column(headers=['dye'], save=True, index=True, agg='min')
    def dye(self:np.uint8) :
        return None