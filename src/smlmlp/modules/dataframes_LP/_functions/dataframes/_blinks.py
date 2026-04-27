#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class blinks(DataFrame) :
    """Blink-level dataframe aggregated from points."""

    @column(headers=['blink'], dtype=np.uint64, fill=0, save=True, agg='min', index="points")
    def blk(self) :
        """Associate points across consecutive frames into blinks."""
        from smlmlp import associate_consecutive_frames

        return associate_consecutive_frames(association_radius_nm=self.locs.config.blink_association_radius_nm, z_association_radius_nm=self.locs.config.blink_z_association_radius_nm, locs=self.locs)[0]


    # Photophysics

    @column(headers=['on time [ms]'], dtype=np.float32, fill=np.nan, save=True, agg='mean')
    def on_time(self) :
        """Compute blink on-time from point counts."""
        unique, counts = np.unique(self.locs.points.blk, return_counts=True)
        if unique[0] == 0 : unique, counts = unique[1:], counts[1:]
        return counts * self.locs.config.exposure_ms

    @column(headers=['off time [ms]'], dtype=np.float32, fill=np.nan, save=True, agg='mean')
    def off_time(self) :
        """Compute blink off-time from frame gaps within molecules."""
        order = np.lexsort((self.fr, self.mol))
        mol_s, fr_s = self.mol[order], self.fr[order]
        diff_s = np.empty_like(fr_s)
        diff_s[:-1] = fr_s[1:] - fr_s[:-1]
        diff_s[:-1][mol_s[1:] != mol_s[:-1]] = np.nan
        diff_s[-1] = np.nan
        diff = np.empty_like(diff_s)
        diff[order] = diff_s
        return diff * self.locs.config.exposure_ms - self.on_time

    @column(headers=['flux [photon]'], dtype=np.float32, fill=0, save=True, agg='mean')
    def flux(self) :
        """Aggregate blink flux."""
        from smlmlp import aggregate_flux

        self.flux, self.switching, _ = aggregate_flux(locs=self.locs)
        return "flux"

    @column(headers=['switching'], dtype=np.bool_, fill=0, save=True, agg='max')
    def switching(self) :
        """Aggregate blink switching state."""
        from smlmlp import aggregate_flux

        self.flux, self.switching, _ = aggregate_flux(locs=self.locs)
        return "switching"


    # Demixing

    @column(headers=['demix'], dtype=np.uint8, fill=0, save=False, agg='min')
    def demix(self) :
        """Select the configured one-dimensional demixing value."""
        match self.locs.config.demix_method :
            case "spectral" : return "demix_spectral"
            case "flux" : return "demix_flux"
            case "lifetime" : return "demix_lifetime"
            case _ : raise ValueError('demix-method not recognized')

    @column(headers=['demix spectral'], dtype=np.uint8, fill=0, save=True, agg='min')
    def demix_spectral(self) :
        """Alias spectral ratio as spectral demixing value."""
        return "spectral_ratio"

    @column(headers=['demix flux'], dtype=np.uint8, fill=0, save=True, agg='min')
    def demix_flux(self) :
        """Compute flux-normalized demixing value."""
        return self.flux / self.irradiance

    @column(headers=['demix lifetime'], dtype=np.uint8, fill=0, save=True, agg='min')
    def demix_lifetime(self) :
        """Alias lifetime as lifetime demixing value."""
        return "lifetime"


    # Demixing 2D

    @column(headers=['demix x'], dtype=np.uint8, fill=0, save=False, agg='min')
    def demix_x(self) :
        """Select the configured x demixing value."""
        match self.locs.config.demix2d_method :
            case "spectral" : return "demix_x_spectral"
            case _ : raise ValueError('demix2d-method not recognized')

    @column(headers=['demix y'], dtype=np.uint8, fill=0, save=False, agg='min')
    def demix_y(self) :
        """Select the configured y demixing value."""
        match self.locs.config.demix2d_method :
            case "spectral" : return "demix_y_spectral"
            case _ : raise ValueError('demix2d-method not recognized')

    @column(headers=['demix spectral x [photon]'], dtype=np.uint8, fill=0, save=True, agg='min')
    def demix_x_spectral(self) :
        """Alias spectral x intensity as x demixing value."""
        return "spectral_x"

    @column(headers=['demix spectral y [photon]'], dtype=np.uint8, fill=0, save=True, agg='min')
    def demix_y_spectral(self) :
        """Alias spectral y intensity as y demixing value."""
        return "spectral_y"


    # Fuse

    @column(headers=['element centroid x distance [nm]'], dtype=np.float32, fill=np.nan, save=True, agg='mean')
    def elm_x(self) :
        """Return x distance from element centroid."""
        return self.xx - self.elm_x0

    @column(headers=['element centroid y distance [nm]'], dtype=np.float32, fill=np.nan, save=True, agg='mean')
    def elm_y(self) :
        """Return y distance from element centroid."""
        return self.yy - self.elm_y0
