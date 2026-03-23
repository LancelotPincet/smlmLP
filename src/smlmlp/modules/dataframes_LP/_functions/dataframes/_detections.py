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

    @column(headers=['area [pix2]'], save=True, index=False, agg='mean')
    def area(self:np.uint8) :
        return None

    @column(headers=['x detection [nm]'], save=True, index=False, agg='mean')
    def xdet(self:np.float32) :
        return None

    @column(headers=['y detection [nm]'], save=True, index=False, agg='mean')
    def ydet(self:np.float32) :
        return None



    # --- Fits ---

    @column(headers=['x [nm]', 'xnm'], save=True, index=False, agg='mean')
    def x(self:np.float32) :
        return "xdet"

    @column(headers=['y [nm]', 'ynm'], save=True, index=False, agg='mean')
    def y(self:np.float32) :
        return "ydet"

    @column(headers=['z [nm]', 'znm'], save=True, index=False, agg='mean')
    def z(self:np.float32) :
        return "zdaisy"

    @column(headers=['chi2'], save=True, index=False, agg='max')
    def chi2(self:np.float32) :
        return None

    @column(headers=['iteration number'], save=True, index=False, agg='max')
    def n_iter(self:np.uint16) :
        return None

    @column(headers=['converged'], save=True, index=False, agg='min')
    def converged(self:np.bool_) :
        return None

    @column(headers=['fit amplitude [photon.pix-2]'], save=True, index=False, agg='mean')
    def amp(self:np.float32) :
        return None

    @column(headers=['fit offset [photon.pix-2]', 'offset [photon]'], save=True, index=False, agg='mean')
    def os(self:np.float32) :
        return None



    # --- Number of photons ---

    @column(headers=['signal [photon]', 'intensity [photon]', 'phot'], save=True, index=False, agg='sum')
    def signal(self:np.float32) :
        return "gaussiansignal"

    @column(headers=['gaussian signal [photon]'], save=False, index=False, agg='sum')
    def gaussiansignal(self:np.float32) :
        return self.amp / 2 / np.pi / (self.sigmax / self.config.xpixel) / (self.sigmay / self.config.ypixel)

    @column(headers=['bkgd [photon]', 'bg'], save=True, index=False, agg='sum')
    def bkgd(self:np.float32) :
        return "gaussianbkgd"

    @column(headers=['gaussian bkgd [photon]'], save=False, index=False, agg='sum')
    def gaussianbkgd(self:np.float32) :
        return self.os * self.config.cropsize**2

    @column(headers=['noise [photon]', 'bkgstd [photon]'], save=True, index=False, agg='mean')
    def noise(self:np.float32) :
        return np.sqrt(self.bkgd) #Poisson noise variance = expected



    # --- Gaussians size ---

    @column(headers=['Gaussian sigma [nm]', 'sigma [nm]'], save=True, index=False, agg='mean')
    def sigma(self:np.float32) :
        return np.sqrt(self.sigmax * self.sigmay)

    @column(headers=['Gaussian sigma x [nm]', 'sigma x [nm]', 'sigma_x [nm]', 'PSFxnm'], save=True, index=False, agg='mean')
    def sigmax(self:np.float32) :
        return "sigma"

    @column(headers=['Gaussian sigma y [nm]', 'sigma y [nm]', 'sigma_y [nm]', 'PSFynm'], save=True, index=False, agg='mean')
    def sigmay(self:np.float32) :
        return self.sigmax

    @column(headers=['sigma ratio', 'ellipticity'], save=False, index=False, agg='mean')
    def sigmaratio(self:np.float32) :
        return self.sigmax / self.sigmay

    @column(headers=['sigma angle [rad]', 'angle [rad]'], save=True, index=False, agg='mean')
    def sigmaangle(self:np.float32) :
        return None



    # --- CRLB ---

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



