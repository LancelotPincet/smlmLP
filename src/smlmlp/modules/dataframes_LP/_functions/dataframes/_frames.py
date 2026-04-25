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

    @column(headers=['frame', 'frames'], dtype=np.uint32, save=True, agg='min', index="points")
    def fr(self) :
        from smlmlp import lost_frames
        return lost_frames(locs=self.locs)[0]



    # --- Drifts ---

    @column(headers=['nodrift [nm]'], dtype=np.float32, save=False, agg='mean')
    def nodrift(self) :
        return np.zeros(self.nframes, dtype=np.float32)

    @column(headers=['drift x [nm]'], dtype=np.float32, save=False, agg='mean')
    def dx(self) :
        match self.locs.config.drift_method :
            case "none" : return "nodrift"
            case "crosscorr" : return "dx_crosscorr"
            case "comet" : return "dx_comet"
            case "aim" : return "dx_aim"
            case "meanshift" : return "dx_meanshift"
            case _ : raise ValueError('dx-method not recognized')

    @column(headers=['drift y [nm]'], dtype=np.float32, save=False, agg='mean')
    def dy(self) :
        match self.locs.config.drift_method :
            case "none" : return "nodrift"
            case "crosscorr" : return "dy_crosscorr"
            case "comet" : return "dy_comet"
            case "aim" : return "dy_aim"
            case "meanshift" : return "dy_meanshift"
            case _ : raise ValueError('dy-method not recognized')

    @column(headers=['drift z [nm]'], dtype=np.float32, save=True, agg='mean')
    def dz(self) :
        match self.locs.config.drift_method :
            case "none" : return "nodrift"
            case "crosscorr" : return "dz_crosscorr"
            case "comet" : return "dz_comet"
            case "aim" : return "dz_aim"
            case "meanshift" : return "dz_meanshift"
            case _ : raise ValueError('dz-method not recognized')

    @column(headers=['drift x crosscorr [nm]'], dtype=np.float32, save=True, agg='mean')
    def dx_crosscorr(self) :
        from smlmlp import drift_crosscorr
        self.dx_crosscorr, self.dy_crosscorr, self.dz_crosscorr, _ = drift_crosscorr(locs=self.locs)
        return "dx_crosscorr"

    @column(headers=['drift y crosscorr [nm]'], dtype=np.float32, save=True, agg='mean')
    def dy_crosscorr(self) :
        from smlmlp import drift_crosscorr
        self.dx_crosscorr, self.dy_crosscorr, self.dz_crosscorr, _ = drift_crosscorr(locs=self.locs)
        return "dy_crosscorr"

    @column(headers=['drift z crosscorr [nm]'], dtype=np.float32, save=True, agg='mean')
    def dz_crosscorr(self) :
        from smlmlp import drift_crosscorr
        self.dx_crosscorr, self.dy_crosscorr, self.dz_crosscorr, _ = drift_crosscorr(locs=self.locs)
        return "dz_crosscorr"

    @column(headers=['drift x comet [nm]'], dtype=np.float32, save=True, agg='mean')
    def dx_comet(self) :
        from smlmlp import drift_comet
        self.dx_comet, self.dy_comet, self.dz_comet, _ = drift_comet(locs=self.locs)
        return "dx_comet"

    @column(headers=['drift y comet [nm]'], dtype=np.float32, save=True, agg='mean')
    def dy_comet(self) :
        from smlmlp import drift_comet
        self.dx_comet, self.dy_comet, self.dz_comet, _ = drift_comet(locs=self.locs)
        return "dy_comet"

    @column(headers=['drift z comet [nm]'], dtype=np.float32, save=True, agg='mean')
    def dz_comet(self) :
        from smlmlp import drift_comet
        self.dx_comet, self.dy_comet, self.dz_comet, _ = drift_comet(locs=self.locs)
        return "dz_comet"

    @column(headers=['drift x aim [nm]'], dtype=np.float32, save=True, agg='mean')
    def dx_aim(self) :
        from smlmlp import drift_aim
        self.dx_aim, self.dy_aim, self.dz_aim, _ = drift_aim(locs=self.locs)
        return "dx_aim"

    @column(headers=['drift y aim [nm]'], dtype=np.float32, save=True, agg='mean')
    def dy_aim(self) :
        from smlmlp import drift_aim
        self.dx_aim, self.dy_aim, self.dz_aim, _ = drift_aim(locs=self.locs)
        return "dy_aim"

    @column(headers=['drift z aim [nm]'], dtype=np.float32, save=True, agg='mean')
    def dz_aim(self) :
        from smlmlp import drift_aim
        self.dx_aim, self.dy_aim, self.dz_aim, _ = drift_aim(locs=self.locs)
        return "dz_aim"

    @column(headers=['drift x meanshift [nm]'], dtype=np.float32, save=True, agg='mean')
    def dx_meanshift(self) :
        from smlmlp import drift_meanshift
        self.dx_meanshift, self.dy_meanshift, self.dz_meanshift, _ = drift_meanshift(locs=self.locs)
        return "dx_meanshift"

    @column(headers=['drift y meanshift [nm]'], dtype=np.float32, save=True, agg='mean')
    def dy_meanshift(self) :
        from smlmlp import drift_meanshift
        self.dx_meanshift, self.dy_meanshift, self.dz_meanshift, _ = drift_meanshift(locs=self.locs)
        return "dy_meanshift"

    @column(headers=['drift z meanshift [nm]'], dtype=np.float32, save=True, agg='mean')
    def dz_meanshift(self) :
        from smlmlp import drift_meanshift
        self.dx_meanshift, self.dy_meanshift, self.dz_meanshift, _ = drift_meanshift(locs=self.locs)
        return "dz_meanshift"



    # Time

    @column(headers=['time [ms]'], dtype=np.float32, save=False, agg='mean')
    def time(self) :
        return self.fr * self.locs.config.exposure_ms

