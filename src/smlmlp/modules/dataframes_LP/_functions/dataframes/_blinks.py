#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class blinks(DataFrame) :
    '''
    Blinks dataframe
    '''

    @column(headers=['blink'], dtype=np.uint64, save=True, agg='min', index="points")
    def blk(self) :
        from smlmlp import associate_consecutive_frames
        return associate_consecutive_frames(association_radius_nm=self.config.blink_association_radius_nm, z_association_radius_nm=self.config.blink_z_association_radius_nm, locs=self.locs)[0]



    # --- Photophysics ---

    @column(headers=['on time [ms]'], dtype=np.float32, save=True, agg='mean')
    def on_time(self) :
        unique, counts = np.unique(self.locs.points.blk, return_counts=True)
        if unique[0] == 0 : unique, counts = unique[1:], counts[1:]
        return counts * self.locs.config.exposure_ms

    @column(headers=['off time [ms]'], dtype=np.float32, save=True, agg='mean')
    def off_time(self) :
        order = np.lexsort((self.fr, self.mol))
        mol_s, fr_s = self.mol[order], self.frame[order]
        diff_s = np.empty_like(fr_s)
        diff_s[:-1] = fr_s[1:] - fr_s[:-1]
        diff_s[:-1][mol_s[1:] != mol_s[:-1]] = np.nan
        diff_s[-1] = np.nan
        diff = np.empty_like(diff_s)
        diff[order] = diff_s
        return diff * self.locs.config.exposure_ms - self.on_time

    @column(headers=['flux [photon]'], dtype=np.float32, save=True, agg='mean')
    def flux(self) :
        from smlmlp import aggregate_flux
        self.flux, self.switching, _ = aggregate_flux(locs=self.locs)
        return "flux"

    @column(headers=['switching'], dtype=np.bool_, save=True, agg='max')
    def switching(self) :
        from smlmlp import aggregate_flux
        self.flux, self.switching, _ = aggregate_flux(locs=self.locs)
        return "switching"



    # --- Demixing ---

    @column(headers=['demix'], dtype=np.uint8, save=False, agg='min')
    def demix(self) :
        match self.locs.config.demix_method :
            case "spectral" : return "demix_spectral"
            case "flux" : return "demix_flux"
            case "lifetime" : return "demix_lifetime"
            case _ : raise ValueError('demix-method not recognized')

    @column(headers=['demix spectral'], dtype=np.uint8, save=True, agg='min')
    def demix_spectral(self) :
        return "spectral_ratio"

    @column(headers=['demix flux'], dtype=np.uint8, save=True, agg='min')
    def demix_flux(self) :
        return self.flux / self.irradiance

    @column(headers=['demix lifetime'], dtype=np.uint8, save=True, agg='min')
    def demix_lifetime(self) :
        return "lifetime"



    # --- Demixing 2D ---

    @column(headers=['demix x'], dtype=np.uint8, save=False, agg='min')
    def demix_x(self) :
        match self.locs.config.demix2d_method :
            case "spectral" : return "demix_x_spectral"
            case _ : raise ValueError('demix2d-method not recognized')

    @column(headers=['demix y'], dtype=np.uint8, save=False, agg='min')
    def demix_y(self) :
        match self.locs.config.demix2d_method :
            case "spectral" : return "demix_y_spectral"
            case _ : raise ValueError('demix2d-method not recognized')

    @column(headers=['demix spectral x [photon]'], dtype=np.uint8, save=True, agg='min')
    def demix_x_spectral(self) :
        return "spectral_x"

    @column(headers=['demix spectral y [photon]'], dtype=np.uint8, save=True, agg='min')
    def demix_y_spectral(self) :
        return "spectral_y"



    # --- Fuse ---

    @column(headers=['element centroid x distance [nm]'], dtype=np.float32, save=True, agg='mean')
    def elm_x(self) :
        return self.xx - self.elm_x0

    @column(headers=['element centroid y distance [nm]'], dtype=np.float32, save=True, agg='mean')
    def elm_y(self) :
        return self.yy - self.elm_y0
