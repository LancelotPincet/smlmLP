#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import MainDataFrame, column
import numpy as np



# %% Function
class detections(MainDataFrame) :
    '''
    Frames dataframe
    '''

    @column(headers=['detection', 'id'], save=True, index=True, agg='min')
    def det(self:np.uint64) :
        return None



    # --- Filters ---

    @column(headers=['filter'], save=False, index=False, agg='min')
    def keep(self:np.bool_) : # to be used with the self.filter function only
        return None



    # --- Detections ---

    @column(headers=['x detection [nm]'], save=True, index=False, agg='mean')
    def x_det(self:np.float32) :
        return None

    @column(headers=['y detection [nm]'], save=True, index=False, agg='mean')
    def y_det(self:np.float32) :
        return None



    # --- Fits ---

    @column(headers=['x [nm]', 'xnm'], save=True, index=False, agg='mean')
    def x(self:np.float32) :
        return "x_det"

    @column(headers=['y [nm]', 'ynm'], save=True, index=False, agg='mean')
    def y(self:np.float32) :
        return "y_det"

    @column(headers=['z [nm]', 'znm'], save=True, index=False, agg='mean')
    def z(self:np.float32) :
        return "z_daisy"

    @column(headers=['amplitude [photon.pix-2]', 'amplitude [photon]'], save=True, index=False, agg='mean')
    def amp(self:np.float32) :
        return None

    @column(headers=['offset [photon.pix-2]', 'offset [photon]'], save=True, index=False, agg='mean')
    def os(self:np.float32) :
        return None

    @column(headers=['standard error x [nm]'], save=True, index=False, agg='mean')
    def x_se(self:np.float32) :
        return None

    @column(headers=['standard error y [nm]'], save=True, index=False, agg='mean')
    def y_se(self:np.float32) :
        return None

    @column(headers=['standard error z [nm]'], save=True, index=False, agg='mean')
    def z_se(self:np.float32) :
        return None

    @column(headers=['standard error amplitude [photon.pix-2]'], save=True, index=False, agg='mean')
    def amp_se(self:np.float32) :
        return None

    @column(headers=['standard error offset [photon.pix-2]'], save=True, index=False, agg='mean')
    def os_se(self:np.float32) :
        return None

    @column(headers=['chi2'], save=True, index=False, agg='max')
    def chi2(self:np.float32) :
        return None

    @column(headers=['iteration number'], save=True, index=False, agg='max')
    def n_iter(self:np.uint16) :
        return None

    @column(headers=['converged'], save=True, index=False, agg='min')
    def converged(self:np.bool_) :
        return None



    # --- Number of photons ---

    @column(headers=['peak snr'], save=False, index=False, agg='mean')
    def snr_peak(self:np.float32) :
        return self.signal / self.noise

    @column(headers=['snr'], save=False, index=False, agg='mean')
    def sbr(self:np.float32) :
        return self.signal / self.bkgd

    @column(headers=['signal [photon]', 'intensity [photon]', 'phot'], save=True, index=False, agg='mean')
    def signal(self:np.float32) :
        return "gaussian_signal"

    @column(headers=['bkgd [photon]', 'bg'], save=True, index=False, agg='mean')
    def bkgd(self:np.float32) :
        return "gaussian_bkgd"

    @column(headers=['noise [photon]', 'bkgstd [photon]'], save=True, index=False, agg='mean')
    def noise(self:np.float32) :
        return np.sqrt(self.bkgd)

    @column(headers=['gaussian signal [photon]'], save=False, index=False, agg='mean')
    def gaussian_signal(self:np.float32) :
        return self.amp * 2 * np.pi * (self.sigma_x / self.x_pixel) * (self.sigma_y / self.y_pixel)

    @column(headers=['gaussian bkgd [photon]'], save=False, index=False, agg='mean')
    def gaussian_bkgd(self:np.float32) :
        return self.os * self.config.cropsize**2

    @column(headers=['gaussian noise [photon]'], save=False, index=False, agg='mean')
    def gaussian_noise(self:np.float32) :
        return np.sqrt(self.gaussian_bkgd)



    # --- Gaussians ---

    @column(headers=['Gaussian sigma [nm]', 'sigma [nm]'], save=True, index=False, agg='mean')
    def sigma(self:np.float32) :
        return np.sqrt(self.sigma_x * self.sigma_y)

    @column(headers=['Gaussian sigma x [nm]', 'sigma x [nm]', 'sigma_x [nm]', 'PSFxnm'], save=True, index=False, agg='mean')
    def sigma_x(self:np.float32) :
        return "sigma"

    @column(headers=['Gaussian sigma y [nm]', 'sigma y [nm]', 'sigma_y [nm]', 'PSFynm'], save=True, index=False, agg='mean')
    def sigma_y(self:np.float32) :
        return "sigma"

    @column(headers=['sigma ratio', 'ellipticity'], save=False, index=False, agg='mean')
    def sigma_ratio(self:np.float32) :
        return self.sigma_x / self.sigma_y

    @column(headers=['sigma angle [rad]', 'angle [rad]'], save=True, index=False, agg='mean')
    def sigma_angle(self:np.float32) :
        return None



    # --- Precision ---

    @column(headers=['CRLB [nm]'], save=False, index=False, agg='mean')
    def crlb(self:np.float32) :
        return (self.crlb_xy**2 * self.crlb_z)**(1/3)

    @column(headers=['CRLB xy [nm]', 'uncertainty_xy'], save=False, index=False, agg='mean')
    def crlb_xy(self:np.float32) :
        # tau = 2*np.pi * (self.bkgd_std**2 + self.camera_compensation) * (self.sigmax * self.sigmay + self.pixel**2/12) / self.pixel**2 / self.signal
        # if self.fit_estimator == 'MLE' :
        #     return np.sqrt((self.camera_gain * self.sigmax*self.sigmay + (self.pixel**2/12)) / self.signal * (1 + 4*tau + np.sqrt(2*tau/(1+4*tau))))
        # if self.fit_estimator == 'LSE' :
        #     return np.sqrt((self.camera_gain * self.sigmax*self.sigmay + (self.pixel**2/12)) / self.signal * (4*tau + 16/9))
        # return None
        return None

    @column(headers=['CRLB z [nm]', 'uncertainty_z'], save=False, index=False, agg='mean')
    def crlb_z(self:np.float32) :
        return "crlb_xy"



