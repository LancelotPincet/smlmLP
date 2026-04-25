#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class sequences(DataFrame) :
    '''
    Sequences dataframe
    '''

    @column(headers=['sequence'], dtype=np.uint32, save=True, agg='min', index="points")
    def seq(self) :
        array = self.fr // self.locs.config.frames_per_sequence
        array[self.fr != 0] += 1
        return array

