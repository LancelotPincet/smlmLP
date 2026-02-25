#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class channels(DataFrame) :
    '''
    Channels dataframe
    '''

    @column(headers=['channel'], save=True, index=True, agg='min')
    def ch(self:np.uint8) :
        return None