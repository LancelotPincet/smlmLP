#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class frames(DataFrame) :
    '''
    Frames dataframe
    '''

    @column(headers=['frame', 'frames'], save=True, agg='min', index="points")
    def fr(self:np.uint32) :
        from smlmlp import index_frames
        return index_frames(locs=self.locs)[0]



    # --- Drifts ---

    @column(headers=['nodrift [nm]'], save=False, agg='mean')
    def nodrift(self:np.float32) :
        return np.zeros(self.nframes, dtype=np.float32)

    @column(headers=['drift x [nm]'], save=False, agg='mean')
    def dx(self:np.float32) :
        match self.locs.config.drift_method :
            case "none" : return "nodrift"
            case "crosscorr" : return "dx_crosscorr"
            case "comet" : return "dx_comet"
            case "aim" : return "dx_aim"
            case "meanshift" : return "dx_meanshift"
            case _ : raise ValueError('dx-method not recognized')

    @column(headers=['drift y [nm]'], save=False, agg='mean')
    def dy(self:np.float32) :
        match self.locs.config.drift_method :
            case "none" : return "nodrift"
            case "crosscorr" : return "dy_crosscorr"
            case "comet" : return "dy_comet"
            case "aim" : return "dy_aim"
            case "meanshift" : return "dy_meanshift"
            case _ : raise ValueError('dy-method not recognized')

    @column(headers=['drift z [nm]'], save=True, agg='mean')
    def dz(self:np.float32) :
        match self.locs.config.drift_method :
            case "none" : return "nodrift"
            case "crosscorr" : return "dz_crosscorr"
            case "comet" : return "dz_comet"
            case "aim" : return "dz_aim"
            case "meanshift" : return "dz_meanshift"
            case _ : raise ValueError('dz-method not recognized')

    @column(headers=['drift x crosscorr [nm]'], save=True, agg='mean')
    def dx_crosscorr(self:np.float32) :
        from smlmlp import drift_crosscorr
        self.dx_crosscorr, self.dy_crosscorr, self.dz_crosscorr, _ = drift_crosscorr(locs=self.locs)
        return "dx_crosscorr"

    @column(headers=['drift y crosscorr [nm]'], save=True, agg='mean')
    def dy_crosscorr(self:np.float32) :
        from smlmlp import drift_crosscorr
        self.dx_crosscorr, self.dy_crosscorr, self.dz_crosscorr, _ = drift_crosscorr(locs=self.locs)
        return "dy_crosscorr"

    @column(headers=['drift z crosscorr [nm]'], save=True, agg='mean')
    def dz_crosscorr(self:np.float32) :
        from smlmlp import drift_crosscorr
        self.dx_crosscorr, self.dy_crosscorr, self.dz_crosscorr, _ = drift_crosscorr(locs=self.locs)
        return "dz_crosscorr"

    @column(headers=['drift x comet [nm]'], save=True, agg='mean')
    def dx_comet(self:np.float32) :
        from smlmlp import drift_comet
        self.dx_comet, self.dy_comet, self.dz_comet, _ = drift_comet(locs=self.locs)
        return "dx_comet"

    @column(headers=['drift y comet [nm]'], save=True, agg='mean')
    def dy_comet(self:np.float32) :
        from smlmlp import drift_comet
        self.dx_comet, self.dy_comet, self.dz_comet, _ = drift_comet(locs=self.locs)
        return "dy_comet"

    @column(headers=['drift z comet [nm]'], save=True, agg='mean')
    def dz_comet(self:np.float32) :
        from smlmlp import drift_comet
        self.dx_comet, self.dy_comet, self.dz_comet, _ = drift_comet(locs=self.locs)
        return "dz_comet"

    @column(headers=['drift x aim [nm]'], save=True, agg='mean')
    def dx_aim(self:np.float32) :
        from smlmlp import drift_aim
        self.dx_aim, self.dy_aim, self.dz_aim, _ = drift_aim(locs=self.locs)
        return "dx_aim"

    @column(headers=['drift y aim [nm]'], save=True, agg='mean')
    def dy_aim(self:np.float32) :
        from smlmlp import drift_aim
        self.dx_aim, self.dy_aim, self.dz_aim, _ = drift_aim(locs=self.locs)
        return "dy_aim"

    @column(headers=['drift z aim [nm]'], save=True, agg='mean')
    def dz_aim(self:np.float32) :
        from smlmlp import drift_aim
        self.dx_aim, self.dy_aim, self.dz_aim, _ = drift_aim(locs=self.locs)
        return "dz_aim"

    @column(headers=['drift x meanshift [nm]'], save=True, agg='mean')
    def dx_meanshift(self:np.float32) :
        from smlmlp import drift_meanshift
        self.dx_meanshift, self.dy_meanshift, self.dz_meanshift, _ = drift_meanshift(locs=self.locs)
        return "dx_meanshift"

    @column(headers=['drift y meanshift [nm]'], save=True, agg='mean')
    def dy_meanshift(self:np.float32) :
        from smlmlp import drift_meanshift
        self.dx_meanshift, self.dy_meanshift, self.dz_meanshift, _ = drift_meanshift(locs=self.locs)
        return "dy_meanshift"

    @column(headers=['drift z meanshift [nm]'], save=True, agg='mean')
    def dz_meanshift(self:np.float32) :
        from smlmlp import drift_meanshift
        self.dx_meanshift, self.dy_meanshift, self.dz_meanshift, _ = drift_meanshift(locs=self.locs)
        return "dz_meanshift"



    # Time

    @column(headers=['time [ms]'], save=False, agg='mean')
    def time(self:np.float32) :
        return self.fr * self.locs.config.exposure_ms

