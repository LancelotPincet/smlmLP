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

    @column(headers=['point'], dtype=np.uint64, save=True, agg='min', index="detections")
    def pnt(self) :
        if self.locs.config.nchannels == 1 :
            return self.locs.detections.det
        else :
            from smlmlp import index_points
            return index_points(locs=self.locs)[0]
    


    # --- Coordinates ---

    @column(headers=['x [nm]', 'xnm'], dtype=np.float32, save=True, agg='mean')
    def x(self) :
        return self.x_measured - self.dx

    @column(headers=['y [nm]', 'ynm'], dtype=np.float32, save=True, agg='mean')
    def y(self) :
        return self.y_measured - self.dy

    @column(headers=['z [nm]', 'znm'], dtype=np.float32, save=True, agg='mean')
    def z(self) :
        return self.z_measured - self.dz



    # --- xy ---

    @column(headers=['x measured [nm]'], dtype=np.float32, save=False, agg='mean')
    def x_measured(self) :
        match self.locs.config.x_method :
            case "det" : return "x_det"
            case "fit" : return "x_fit"
            case "modloc" : return "x_modloc"
            case "timeloc" : return "x_timeloc"
            case _ : raise ValueError('x-method not recognized')

    @column(headers=['y measured [nm]'], dtype=np.float32, save=False, agg='mean')
    def y_measured(self) :
        match self.locs.config.y_method :
            case "det" : return "y_det"
            case "fit" : return "y_fit"
            case "modloc" : return "y_modloc"
            case "timeloc" : return "y_timeloc"
            case _ : raise ValueError('y-method not recognized')

    @column(headers=['x modloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_modloc(self) :
        from smlmlp import modloc_transverse
        self.x_modloc, self.y_modloc, _ = modloc_transverse(locs=self.locs)
        return "x_modloc"

    @column(headers=['y modloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_modloc(self) :
        from smlmlp import modloc_transverse
        self.x_modloc, self.y_modloc, _ = modloc_transverse(locs=self.locs)
        return "y_modloc"

    @column(headers=['x timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_timeloc(self) :
        from smlmlp import timeloc_transverse
        self.x_timeloc, self.y_timeloc, _ = timeloc_transverse(locs=self.locs)
        return "x_timeloc"

    @column(headers=['y timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_timeloc(self) :
        from smlmlp import timeloc_transverse
        self.x_timeloc, self.y_timeloc, _ = timeloc_transverse(locs=self.locs)
        return "y_timeloc"



    # --- z ---

    @column(headers=['z measured [nm]'], dtype=np.float32, save=False, agg='mean')
    def z_measured(self) :
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

    @column(headers=['z modloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_modloc(self) :
        from smlmlp import modloc_axial
        return modloc_axial(locs=self.locs)[0]

    @column(headers=['z timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_timeloc(self) :
        from smlmlp import timeloc_axial
        return timeloc_axial(locs=self.locs)[0]

    @column(headers=['z astigmatism [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_astig(self) :
        from smlmlp import psf_astig
        return psf_astig(locs=self.locs)[0]

    @column(headers=['z biplane [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_biplane(self) :
        from smlmlp import psf_biplane
        return psf_biplane(locs=self.locs)[0]

    @column(headers=['z donald [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_donald(self) :
        from smlmlp import surface_donald
        return surface_donald(locs=self.locs)[0]

    @column(headers=['z miet [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_miet(self) :
        from smlmlp import surface_miet
        return surface_miet(locs=self.locs)[0]

    @column(headers=['z qtirf [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_qtirf(self) :
        from smlmlp import surface_qtirf
        return surface_qtirf(locs=self.locs)[0]



    # --- orientation azimuth ---

    @column(headers=['azimuth [deg]'], dtype=np.float32, save=False, agg='mean')
    def azimuth(self) :
        match self.locs.config.azimuth_method :
            case "polar2d" : return "azimuth_polar2d"
            case "polar3d" : return "azimuth_polar3d"
            case _ : raise ValueError('azimuth-method not recognized')

    @column(headers=['azimuth polar2d [deg]'], dtype=np.float32, save=True, agg='mean')
    def azimuth_polar2d(self) :
        from smlmlp import orient_polar2d
        return orient_polar2d(locs=self.locs)[0]

    @column(headers=['azimuth polar3d [deg]'], dtype=np.float32, save=True, agg='mean')
    def azimuth_polar3d(self) :
        from smlmlp import orient_polar3d
        self.azimuth_polar3d, self.tilt_polar3d = orient_polar3d(locs=self.locs)[0]
        return "azimuth_polar3d"



    # --- orientation tilt ---

    @column(headers=['tilt [deg]'], dtype=np.float32, save=False, agg='mean')
    def tilt(self) :
        match self.locs.config.tilt_method :
            case "polar3d" : return "azimuth_polar3d"
            case _ : raise ValueError('tilt-method not recognized')

    @column(headers=['tilt polar3d [deg]'], dtype=np.float32, save=True, agg='mean')
    def tilt_polar3d(self) :
        from smlmlp import orient_polar3d
        self.azimuth_polar3d, self.tilt_polar3d = orient_polar3d(locs=self.locs)[0]
        return "tilt_polar3d"



    # --- phase ---

    @column(headers=['phase [rad]'], dtype=np.float32, save=False, agg='mean')
    def phase(self) :
        match self.locs.config.phase_method :
            case "demodulated" : return "phase_demodulated"
            case "sequential" : return "phase_sequential"
            case _ : raise ValueError('phase-method not recognized')

    @column(headers=['phase demodulated [rad]'], dtype=np.float32, save=True, agg='mean')
    def phase_demodulated(self) :
        from smlmlp import modloc_demodulated
        return modloc_demodulated(locs=self.locs)[0]

    @column(headers=['phase sequential [rad]'], dtype=np.float32, save=True, agg='mean')
    def phase_sequential(self) :
        from smlmlp import modloc_sequential
        return modloc_sequential(locs=self.locs)[0]



    # --- lifetime ---

    @column(headers=['lifetime [ns]'], dtype=np.float32, save=False, agg='mean')
    def lifetime(self) :
        match self.locs.config.lifetime_method :
            case "tcspc" : return "lifetime_tcspc"
            case "iflim" : return "lifetime_iflim"
            case "dpflim" : return "lifetime_dpflim"
            case _ : raise ValueError('lifetime-method not recognized')

    @column(headers=['lifetime tcspc [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_tcspc(self) :
        from smlmlp import flim_tcspc
        return flim_tcspc(locs=self.locs)[0]

    @column(headers=['lifetime iflim [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_iflim(self) :
        from smlmlp import flim_iflim
        return flim_iflim(locs=self.locs)[0]

    @column(headers=['lifetime dpflim [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_dpflim(self) :
        from smlmlp import flim_dpflim
        return flim_dpflim(locs=self.locs)[0]



    # --- frequency ---

    @column(headers=['frequency [hz]'], dtype=np.float32, save=False, agg='mean')
    def frequency(self) :
        match self.locs.config.frequency_method :
            case "singlespad" : return "frequency_singlespad"
            case "spadarray" : return "frequency_spadarray"
            case _ : raise ValueError('timeloc-method not recognized')

    @column(headers=['frequency singlespad [hz]'], dtype=np.float32, save=True, agg='mean')
    def frequency_singlespad(self) :
        from smlmlp import timeloc_singlespad
        return timeloc_singlespad(locs=self.locs)[0]

    @column(headers=['frequency spadarray [hz]'], dtype=np.float32, save=True, agg='mean')
    def frequency_spadarray(self) :
        from smlmlp import timeloc_spadarray
        return timeloc_spadarray(locs=self.locs)[0]



