#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class points(DataFrame) :
    '''
    Points dataframe
    '''

    @column(headers=['point'], save=True, agg='min', index="detections")
    def pnt(self:np.uint64) :
        if self.locs.nchannels == 1 :
            return self.locs.detections.det
        else :
            from smlmlp import index_points
            return index_points(locs=self.locs)[0]
    


    # --- Coordinates ---

    @column(headers=['x [nm]', 'xnm'], save=True, agg='mean')
    def x(self:np.float32) :
        return self.x_measured - self.dx

    @column(headers=['y [nm]', 'ynm'], save=True, agg='mean')
    def y(self:np.float32) :
        return self.y_measured - self.dy

    @column(headers=['z [nm]', 'znm'], save=True, agg='mean')
    def z(self:np.float32) :
        return self.z_measured - self.dz



    # --- xy ---

    @column(headers=['x measured [nm]'], save=False, agg='mean')
    def x_measured(self:np.float32) :
        match self.locs.config.x_method :
            case "det" : return "x_det"
            case "fit" : return "x_fit"
            case "modloc" : return "x_modloc"
            case "timeloc" : return "x_timeloc"
            case _ : raise ValueError('x-method not recognized')

    @column(headers=['y measured [nm]'], save=False, agg='mean')
    def y_measured(self:np.float32) :
        match self.locs.config.y_method :
            case "det" : return "y_det"
            case "fit" : return "y_fit"
            case "modloc" : return "y_modloc"
            case "timeloc" : return "y_timeloc"
            case _ : raise ValueError('y-method not recognized')

    @column(headers=['x modloc [nm]'], save=True, agg='mean')
    def x_modloc(self:np.float32) :
        from smlmlp import modloc_transverse
        self.x_modloc, self.y_modloc, _ = modloc_transverse(locs=self.locs)
        return "x_modloc"

    @column(headers=['y modloc [nm]'], save=True, agg='mean')
    def y_modloc(self:np.float32) :
        from smlmlp import modloc_transverse
        self.x_modloc, self.y_modloc, _ = modloc_transverse(locs=self.locs)
        return "y_modloc"

    @column(headers=['x timeloc [nm]'], save=True, agg='mean')
    def x_timeloc(self:np.float32) :
        from smlmlp import timeloc_transverse
        self.x_timeloc, self.y_timeloc, _ = timeloc_transverse(locs=self.locs)
        return "x_timeloc"

    @column(headers=['y timeloc [nm]'], save=True, agg='mean')
    def y_timeloc(self:np.float32) :
        from smlmlp import timeloc_transverse
        self.x_timeloc, self.y_timeloc, _ = timeloc_transverse(locs=self.locs)
        return "y_timeloc"



    # --- z ---

    @column(headers=['z measured [nm]'], save=False, agg='mean')
    def z_measured(self:np.float32) :
        match self.locs.config.z_method :
            case "fit" : return "z_fit"
            case "astig" : return "z_astig"
            case "biplane" : return "z_biplane"
            case "donald" : return "z_donald"
            case "daisy" : return "z_daisy"
            case "modloc" : return "z_modloc"
            case "miet" : return "z_miet"
            case "qtirf" : return "z_qtirf"
            case _ : raise ValueError('z-method not recognized')

    @column(headers=['z modloc [nm]'], save=True, agg='mean')
    def z_modloc(self:np.float32) :
        from smlmlp import modloc_axial
        return modloc_axial(locs=self.locs)[0]

    @column(headers=['z timeloc [nm]'], save=True, agg='mean')
    def z_timeloc(self:np.float32) :
        from smlmlp import timeloc_axial
        return timeloc_axial(locs=self.locs)[0]

    @column(headers=['z astigmatism [nm]'], save=True, agg='mean')
    def z_astig(self:np.float32) :
        from smlmlp import psf_astig
        return psf_astig(locs=self.locs)[0]

    @column(headers=['z biplane [nm]'], save=True, agg='mean')
    def z_biplane(self:np.float32) :
        from smlmlp import psf_biplane
        return psf_biplane(locs=self.locs)[0]

    @column(headers=['z donald [nm]'], save=True, agg='mean')
    def z_donald(self:np.float32) :
        from smlmlp import surface_donald
        return surface_donald(locs=self.locs)[0]

    @column(headers=['z miet [nm]'], save=True, agg='mean')
    def z_miet(self:np.float32) :
        from smlmlp import surface_miet
        return surface_miet(locs=self.locs)[0]

    @column(headers=['z qtirf [nm]'], save=True, agg='mean')
    def z_qtirf(self:np.float32) :
        from smlmlp import surface_qtirf
        return surface_qtirf(locs=self.locs)[0]



    # --- orientation azimuth ---

    @column(headers=['azimuth [deg]'], save=False, agg='mean')
    def azimuth(self:np.float32) :
        match self.locs.config.azimuth_method :
            case "polar2d" : return "azimuth_polar2d"
            case "polar3d" : return "azimuth_polar3d"
            case _ : raise ValueError('azimuth-method not recognized')

    @column(headers=['azimuth polar2d [deg]'], save=True, agg='mean')
    def azimuth_polar2d(self:np.float32) :
        from smlmlp import orient_polar2d
        return orient_polar2d(locs=self.locs)[0]

    @column(headers=['azimuth polar3d [deg]'], save=True, agg='mean')
    def azimuth_polar3d(self:np.float32) :
        from smlmlp import orient_polar3d
        self.azimuth_polar3d, self.tilt_polar3d = orient_polar3d(locs=self.locs)[0]
        return "azimuth_polar3d"



    # --- orientation tilt ---

    @column(headers=['tilt [deg]'], save=False, agg='mean')
    def tilt(self:np.float32) :
        match self.locs.config.tilt_method :
            case "polar3d" : return "azimuth_polar3d"
            case _ : raise ValueError('tilt-method not recognized')

    @column(headers=['tilt polar3d [deg]'], save=True, agg='mean')
    def tilt_polar3d(self:np.float32) :
        from smlmlp import orient_polar3d
        self.azimuth_polar3d, self.tilt_polar3d = orient_polar3d(locs=self.locs)[0]
        return "tilt_polar3d"



    # --- phase ---

    @column(headers=['phase [rad]'], save=False, agg='mean')
    def phase(self:np.float32) :
        match self.locs.config.phase_method :
            case "demodulated" : return "phase_demodulated"
            case "sequential" : return "phase_sequential"
            case _ : raise ValueError('phase-method not recognized')

    @column(headers=['phase demodulated [rad]'], save=True, agg='mean')
    def phase_demodulated(self:np.float32) :
        from smlmlp import modloc_demodulated
        return modloc_demodulated(locs=self.locs)[0]

    @column(headers=['phase sequential [rad]'], save=True, agg='mean')
    def phase_sequential(self:np.float32) :
        from smlmlp import modloc_sequential
        return modloc_sequential(locs=self.locs)[0]



    # --- lifetime ---

    @column(headers=['lifetime [ns]'], save=False, agg='mean')
    def lifetime(self:np.float32) :
        match self.locs.config.lifetime_method :
            case "tcspc" : return "lifetime_tcspc"
            case "iflim" : return "lifetime_iflim"
            case "dpflim" : return "lifetime_dpflim"
            case _ : raise ValueError('lifetime-method not recognized')

    @column(headers=['lifetime tcspc [ns]'], save=True, agg='mean')
    def lifetime_tcspc(self:np.float32) :
        from smlmlp import flim_tcspc
        return flim_tcspc(locs=self.locs)[0]

    @column(headers=['lifetime iflim [ns]'], save=True, agg='mean')
    def lifetime_iflim(self:np.float32) :
        from smlmlp import flim_iflim
        return flim_iflim(locs=self.locs)[0]

    @column(headers=['lifetime dpflim [ns]'], save=True, agg='mean')
    def lifetime_dpflim(self:np.float32) :
        from smlmlp import flim_dpflim
        return flim_dpflim(locs=self.locs)[0]



    # --- frequency ---

    @column(headers=['frequency [hz]'], save=False, agg='mean')
    def frequency(self:np.float32) :
        match self.locs.config.frequency_method :
            case "singlespad" : return "frequency_singlespad"
            case "spadarray" : return "frequency_spadarray"
            case _ : raise ValueError('timeloc-method not recognized')

    @column(headers=['frequency singlespad [hz]'], save=True, agg='mean')
    def frequency_singlespad(self:np.float32) :
        from smlmlp import timeloc_singlespad
        return timeloc_singlespad(locs=self.locs)[0]

    @column(headers=['frequency spadarray [hz]'], save=True, agg='mean')
    def frequency_spadarray(self:np.float32) :
        from smlmlp import timeloc_spadarray
        return timeloc_spadarray(locs=self.locs)[0]



