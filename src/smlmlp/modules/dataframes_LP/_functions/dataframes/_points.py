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
            from smlmlp import associate_different_channels
            return associate_different_channels(association_radius_nm=self.locs.config.channel_association_radius_nm, locs=self.locs)[0]
    


    # --- Coordinates ---

    @column(headers=['x stable [nm]'], dtype=np.float32, save=True, agg='mean')
    def xx(self) :
        if self.dx is None : return "x"
        return self.x - self.dx

    @column(headers=['y stable [nm]'], dtype=np.float32, save=True, agg='mean')
    def yy(self) :
        if self.dy is None : return "y"
        return self.y - self.dy

    @column(headers=['z stable [nm]'], dtype=np.float32, save=True, agg='mean')
    def zz(self) :
        if self.dz is None : return "z"
        return self.z - self.dz



    # --- xy ---

    @column(headers=['x [nm]', 'xnm'], dtype=np.float32, save=False, agg='mean')
    def x(self) :
        match self.locs.config.x_method :
            case "det" : return "x_globdet"
            case "fit" : return "x_eff"
            case "modloc" : return "x_modloc"
            case "timeloc" : return "x_timeloc"
            case _ : raise ValueError('x-method not recognized')

    @column(headers=['y [nm]', 'ynm'], dtype=np.float32, save=False, agg='mean')
    def y(self) :
        match self.locs.config.y_method :
            case "det" : return "y_globdet"
            case "fit" : return "y_eff"
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
        if self.x_freq is None : return None
        from smlmlp import calibration_convert
        return calibration_convert(self.x_freq, self.locs.config.x_timeloc_calibration)[0]

    @column(headers=['y timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_timeloc(self) :
        if self.y_freq is None : return None
        from smlmlp import calibration_convert
        return calibration_convert(self.y_freq, self.locs.config.y_timeloc_calibration)[0]



    # --- z ---

    @column(headers=['z [nm]', 'znm'], dtype=np.float32, save=False, agg='mean')
    def z(self) :
        match self.locs.config.z_method :
            case "fit" : return "z_eff"
            case "modloc" : return "z_modloc"
            case "timeloc" : return "z_timeloc"
            case "astig" : return "z_astig"
            case "biplane" : return "z_biplane"
            case "donald" : return "z_donald"
            case "miet" : return "z_miet"
            case "qtirf" : return "z_qtirf"
            case _ : raise ValueError('z-method not recognized')

    @column(headers=['z modloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_modloc(self) :
        from smlmlp import modloc_axial
        return modloc_axial(locs=self.locs)[0]

    @column(headers=['z timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_timeloc(self) :
        if self.z_freq is None : return None
        from smlmlp import calibration_convert
        return calibration_convert(self.z_freq, self.locs.config.z_timeloc_calibration)[0]

    @column(headers=['z astigmatism [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_astig(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.sigma_ratio, self.locs.config.z_astig_calibration)[0]

    @column(headers=['z biplane [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_biplane(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.biplane_ratio, self.locs.config.z_biplane_calibration)[0]

    @column(headers=['z donald [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_donald(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.donald_ratio, self.locs.config.z_donald_calibration)[0]

    @column(headers=['z miet [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_miet(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.lifetime, self.locs.config.z_miet_calibration)[0]

    @column(headers=['z qtirf [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_qtirf(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.intensity/self.irradiance, self.locs.config.z_qtirf_calibration)[0]



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
        return None

    @column(headers=['lifetime iflim [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_iflim(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.iflim_ratio, self.locs.config.lifetime_iflim_calibration)[0]

    @column(headers=['lifetime dpflim [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_dpflim(self) :
        from smlmlp import calibration_convert
        return calibration_convert(self.dpflim_ratio, self.locs.config.lifetime_dpflim_calibration)[0]



    # --- frequency ---

    @column(headers=['x frequency [hz]'], dtype=np.float32, save=True, agg='mean')
    def x_freq(self) :
        return None

    @column(headers=['y frequency [hz]'], dtype=np.float32, save=True, agg='mean')
    def y_freq(self) :
        return None

    @column(headers=['z frequency [hz]'], dtype=np.float32, save=True, agg='mean')
    def z_freq(self) :
        return None



    # --- spectral ---

    @column(headers=['spectral ratio'], dtype=np.float32, save=True, agg='mean')
    def spectral_ratio(self) :
        return self.spectral_y / self.spectral_x

    @column(headers=['spectral x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def spectral_x(self) :
        from smlmlp import aggregate_ratio
        self.spectral_x, self.spectral_y, _ = aggregate_ratio(self.intensity, x_channels=self.spectral_x_channels, y_channels=self.spectral_y_channels, locs=self.locs)
        return "spectral_x"

    @column(headers=['spectral y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def spectral_y(self) :
        from smlmlp import aggregate_ratio
        self.spectral_x, self.spectral_y, _ = aggregate_ratio(self.intensity, x_channels=self.spectral_x_channels, y_channels=self.spectral_y_channels, locs=self.locs)
        return "spectral_y"



    # --- biplane ---

    @column(headers=['biplane ratio'], dtype=np.float32, save=True, agg='mean')
    def biplane_ratio(self) :
        return self.biplane_y / self.biplane_x

    @column(headers=['biplane x width [nm]'], dtype=np.float32, save=True, agg='mean')
    def biplane_x(self) :
        from smlmlp import aggregate_ratio
        self.biplane_x, self.biplane_y, _ = aggregate_ratio(self.sigma, x_channels=self.biplane_x_channels, y_channels=self.biplane_y_channels, locs=self.locs)
        return "biplane_x"

    @column(headers=['biplane y width [nm]'], dtype=np.float32, save=True, agg='mean')
    def biplane_y(self) :
        from smlmlp import aggregate_ratio
        self.biplane_x, self.biplane_y, _ = aggregate_ratio(self.sigma, x_channels=self.biplane_x_channels, y_channels=self.biplane_y_channels, locs=self.locs)
        return "biplane_y"



    # --- donald ---

    @column(headers=['donald ratio'], dtype=np.float32, save=True, agg='mean')
    def donald_ratio(self) :
        return self.donald_y / self.donald_x

    @column(headers=['donald x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def donald_x(self) :
        from smlmlp import aggregate_ratio
        self.donald_x, self.donald_y, _ = aggregate_ratio(self.sigma, x_channels=self.donald_x_channels, y_channels=self.donald_y_channels, locs=self.locs)
        return "donald_x"

    @column(headers=['donald y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def donald_y(self) :
        from smlmlp import aggregate_ratio
        self.donald_x, self.donald_y, _ = aggregate_ratio(self.sigma, x_channels=self.donald_x_channels, y_channels=self.donald_y_channels, locs=self.locs)
        return "donald_y"



    # --- iflim ---

    @column(headers=['iflim ratio'], dtype=np.float32, save=True, agg='mean')
    def iflim_ratio(self) :
        return self.iflim_y / self.iflim_x

    @column(headers=['iflim x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def iflim_x(self) :
        from smlmlp import aggregate_ratio
        self.iflim_x, self.iflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.iflim_x_channels, y_channels=self.iflim_y_channels, locs=self.locs)
        return "iflim_x"

    @column(headers=['iflim y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def iflim_y(self) :
        from smlmlp import aggregate_ratio
        self.iflim_x, self.iflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.iflim_x_channels, y_channels=self.iflim_y_channels, locs=self.locs)
        return "iflim_y"



    # --- dpflim ---

    @column(headers=['dpflim ratio'], dtype=np.float32, save=True, agg='mean')
    def dpflim_ratio(self) :
        return self.dpflim_y / self.dpflim_x

    @column(headers=['dpflim x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def dpflim_x(self) :
        from smlmlp import aggregate_ratio
        self.dpflim_x, self.dpflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.dpflim_x_channels, y_channels=self.dpflim_y_channels, locs=self.locs)
        return "dpflim_x"

    @column(headers=['dpflim y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def dpflim_y(self) :
        from smlmlp import aggregate_ratio
        self.dpflim_x, self.dpflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.dpflim_x_channels, y_channels=self.dpflim_y_channels, locs=self.locs)
        return "dpflim_y"


