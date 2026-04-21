#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column, index_frames
import numpy as np



# %% Function
class frames(DataFrame) :
    '''
    Frames dataframe
    '''

    @column(headers=['frame', 'frames'], save=True, index=True, agg='min')
    def fr(self:np.uint32) :
        return index_frames(locs=self.locs)



    # --- Drifts ---

    @column(headers=['nodrift [nm]'], save=False, index=False, agg='mean')
    def nodrift(self:np.float32) :
        return np.zeros(self.nframes, dtype=np.float32)

    @column(headers=['drift x [nm]'], save=True, index=False, agg='mean')
    def dx(self:np.float32) :
        return "nodrift"

    @column(headers=['drift y [nm]'], save=True, index=False, agg='mean')
    def dy(self:np.float32) :
        return "nodrift"

    @column(headers=['drift z [nm]'], save=True, index=False, agg='mean')
    def dz(self:np.float32) :
        return "nodrift"

    @column(headers=['drift phi [rad]'], save=True, index=False, agg='mean')
    def dphi(self:np.float32) :
        return "nodrift"

    @column(headers=['drift freq [Hz]'], save=True, index=False, agg='mean')
    def dfreq(self:np.float32) :
        return "nodrift"
