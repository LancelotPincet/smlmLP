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

    @column(headers=['molecule'], save=True, index=True, agg='min')
    def mol(self:np.uint64) :
        return None