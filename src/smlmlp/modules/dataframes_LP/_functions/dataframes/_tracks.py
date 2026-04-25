#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class tracks(DataFrame) :
    '''
    Tracks dataframe
    '''

    @column(headers=['track'], dtype=np.uint64, save=True, agg='min', index="points")
    def trk(self) :
        from smlmlp import associate_consecutive_frames
        return associate_consecutive_frames(association_radius_nm=self.config.track_association_radius_nm, z=None, locs=self.locs)[0]

