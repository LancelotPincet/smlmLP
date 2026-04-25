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

    @column(headers=['molecule'], dtype=np.uint64, save=True, agg='min', index="blinks")
    def mol(self) :
        from smlmlp import associate_molecules
        return associate_molecules(locs=self.locs)[0]

