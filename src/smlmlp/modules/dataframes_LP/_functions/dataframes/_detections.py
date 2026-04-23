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

    @column(headers=['detection', 'id'], save=True, agg='min', index=True)
    def det(self:np.uint64) :
        return None



    # --- Filters ---

    @column(headers=['filter'], save=False, agg='min')
    def keep(self:np.bool_) : # to be used with the self.filter function only
        return None



    # --- Detections ---

    @column(headers=['x detection [nm]'], save=True, agg='mean')
    def x_det(self:np.float32) :
        return None

    @column(headers=['y detection [nm]'], save=True, agg='mean')
    def y_det(self:np.float32) :
        return None



    # --- PSF fits ---

    @column(headers=['x fit [nm]'], save=True, agg='mean')
    def x_fit(self:np.float32) :
        return "x_det"

    @column(headers=['y fit [nm]'], save=True, agg='mean')
    def y_fit(self:np.float32) :
        return "y_det"

    @column(headers=['z fit [nm]'], save=True, agg='mean')
    def z_fit(self:np.float32) :
        return None

    @column(headers=['amplitude fit [photon.pix-2]', 'amplitude [photon]'], save=True, agg='mean')
    def amp_fit(self:np.float32) :
        return None

    @column(headers=['offset fit [photon.pix-2]', 'offset [photon]'], save=True, agg='mean')
    def os_fit(self:np.float32) :
        return None

    @column(headers=['standard error x fit [nm]'], save=True, agg='mean')
    def x_se(self:np.float32) :
        return None

    @column(headers=['standard error y fit [nm]'], save=True, agg='mean')
    def y_se(self:np.float32) :
        return None

    @column(headers=['standard error z fit [nm]'], save=True, agg='mean')
    def z_se(self:np.float32) :
        return None

    @column(headers=['standard error amplitude fit [photon.pix-2]'], save=True, agg='mean')
    def amp_se(self:np.float32) :
        return None

    @column(headers=['standard error offset fit [photon.pix-2]'], save=True, agg='mean')
    def os_se(self:np.float32) :
        return None

    @column(headers=['chi2'], save=True, agg='max')
    def chi2(self:np.float32) :
        return None

    @column(headers=['iteration number'], save=True, agg='max')
    def n_iter(self:np.uint16) :
        return None

    @column(headers=['converged'], save=True, agg='min')
    def converged(self:np.bool_) :
        return None



    # --- Photometry ---

    @column(headers=['amplitude [photon.pix-2]'], save=True, agg='mean')
    def amp(self:np.float32) :
        return "amp_fit"

    @column(headers=['offset [photon.pix-2]'], save=True, agg='mean')
    def os(self:np.float32) :
        return "os_fit"

    @column(headers=['intensity [photon]', 'phot'], save=True, agg='mean')
    def intensity(self:np.float32) :
        return "gaussian_intensity"

    @column(headers=['snr'], save=False, agg='mean')
    def snr(self:np.float32) :
        return self.intensity / np.sqrt(self.os * self.x_cropshape * self.y_cropshape + self.intensity + self.read_noise**2 * self.x_cropshape * self.y_cropshape)

    @column(headers=['snr peak'], save=False, agg='mean')
    def snr_peak(self:np.float32) :
        return self.amp / np.sqrt(self.os + self.amp + self.read_noise**2)

    @column(headers=['sbr'], save=False, agg='mean')
    def sbr(self:np.float32) :
        return self.intensity / (self.os * self.x_cropshape * self.y_cropshape)

    @column(headers=['sbr peak'], save=False, agg='mean')
    def sbr_peak(self:np.float32) :
        return self.amp / self.os



    # --- Gaussians ---

    @column(headers=['gaussian intensity [photon]'], save=False, agg='mean')
    def gaussian_signal(self:np.float32) :
        return self.amp * 2 * np.pi * (self.sigma_x / self.x_pixel) * (self.sigma_y / self.y_pixel)

    @column(headers=['gaussian sigma [nm]'], save=True, agg='mean')
    def sigma(self:np.float32) :
        if self.locs.config.model not in ['isogaussian', 'gaussian'] :
            return np.sqrt(self.nphotons / 2 / np.pi / self.amp)
        return np.sqrt(self.sigma_x * self.sigma_y)

    @column(headers=['gaussian sigma x [nm]'], save=True, agg='mean')
    def sigma_x(self:np.float32) :
        return "sigma_x_fit"

    @column(headers=['gaussian sigma y [nm]'], save=True, agg='mean')
    def sigma_y(self:np.float32) :
        return "sigma_y_fit"

    @column(headers=['gaussian sigma fit [nm]', 'sigma [nm]'], save=True, agg='mean')
    def sigma_fit(self:np.float32) :
        return "sigma"

    @column(headers=['gaussian sigma x fit [nm]', 'sigma x [nm]', 'sigma_x [nm]', 'PSFxnm'], save=True, agg='mean')
    def sigma_x_fit(self:np.float32) :
        return "sigma_fit"

    @column(headers=['gaussian sigma y fit [nm]', 'sigma y [nm]', 'sigma_y [nm]', 'PSFynm'], save=True, agg='mean')
    def sigma_y_fit(self:np.float32) :
        return "sigma_fit"

    @column(headers=['sigma ratio', 'ellipticity'], save=False, agg='mean')
    def sigma_ratio(self:np.float32) :
        return self.sigma_x / self.sigma_y

    @column(headers=['sigma angle [deg]', 'angle [deg]'], save=True, agg='mean')
    def sigma_angle(self:np.float32) :
        return "psf_theta"



    # --- CRLB ---

    @column(headers=['crlb xy [nm]', 'uncertainty_xy', 'uncertainty_xy [nm]'], save=False, agg='mean')
    def crlb(self:np.float32) :
        return np.sqrt((self.crlb_x**2 + self.crlb_y**2) / 2)

    @column(headers=['crlb x [nm]', 'uncertainty_x', 'uncertainty_x [nm]'], save=False, agg='mean')
    def crlb_x(self:np.float32) :
        ang = np.radians(self.sigma_angle)
        sig = np.sqrt(self.sigma_x**2 * np.cos(ang)**2 + self.sigma_y**2 * np.sin(ang)**2)
        pixel2 = self.x_pixel**2
        effective_width = sig**2 + pixel2 / 12
        noise2 = self.os + self.read_noise**2
        tau = 2 * np.pi * noise2 * effective_width / pixel2 / self.signal
        if self.locs.config.estimator == 'mle' and self.locs.config.distribution == 'poisson':
            correction = (1 + 4 * tau + np.sqrt(2 * tau / (1 + 4 * tau)))
        elif self.locs.config.estimator == 'lse' or (self.locs.config.estimator == 'mle' and self.locs.config.distribution == 'normal') :
            correction = (4 * tau + 16 / 9)
        else :
            raise ValueError('Fitting config not recognized')
        return np.sqrt( effective_width / self.signal * correction )

    @column(headers=['crlb y [nm]', 'uncertainty_y', 'uncertainty_y [nm]'], save=False, agg='mean')
    def crlb_y(self:np.float32) :
        ang = np.radians(self.sigma_angle)
        sig = np.sqrt(self.sigma_y**2 * np.cos(ang)**2 + self.sigma_x**2 * np.sin(ang)**2)
        pixel2 = self.y_pixel**2
        effective_width = sig**2 + pixel2 / 12
        noise2 = self.os + self.read_noise**2
        tau = 2 * np.pi * noise2 * effective_width / pixel2 / self.signal
        if self.locs.config.estimator == 'mle' and self.locs.config.distribution == 'poisson':
            correction = (1 + 4 * tau + np.sqrt(2 * tau / (1 + 4 * tau)))
        elif self.locs.config.estimator == 'lse' or (self.locs.config.estimator == 'mle' and self.locs.config.distribution == 'normal') :
            correction = (4 * tau + 16 / 9)
        else :
            raise ValueError('Fitting config not recognized')
        return np.sqrt( effective_width / self.signal * correction )



    # --- Aberations ---

    @column(headers=[f'no zernike'], save=False, agg='mean')
    def nozernike(self:np.float32) :
        return np.zeros(self.ndetections, dtype=np.float32)

    @property
    def nzernike(self) :
        return self._nzernike
    @nzernike.setter
    def nzernike(self, value) :
        self._nzernike = int(value)
        cls = self.__class__
        for i in range(1, int(value) +1) :
            @column(headers=[f'zernike {i:02}'], save=True, agg='mean')
            def zernike(self:np.float32) :
                return "nozernike"
            setattr(cls, f'zernike_{i:02}', zernike)


