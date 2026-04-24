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

    @column(headers=['detection', 'id'], dtype=np.uint64, save=True, agg='min', index=True)
    def det(self) :
        return None



    # --- Filters ---

    @column(headers=['filter'], dtype=np.bool_, save=False, agg='min')
    def keep(self) : # to be used with the self.filter function only
        return None



    # --- Detections ---

    @column(headers=['x detection [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_det(self) :
        return None

    @column(headers=['y detection [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_det(self) :
        return None



    # --- PSF fits ---

    @column(headers=['x fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_fit(self) :
        return "x_det"

    @column(headers=['y fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_fit(self) :
        return "y_det"

    @column(headers=['z fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_fit(self) :
        return None

    @column(headers=['amplitude fit [photon.pix-2]', 'amplitude [photon]'], dtype=np.float32, save=True, agg='mean')
    def amp_fit(self) :
        return None

    @column(headers=['offset fit [photon.pix-2]', 'offset [photon]'], dtype=np.float32, save=True, agg='mean')
    def os_fit(self) :
        return None

    @column(headers=['standard error x fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_se(self) :
        return None

    @column(headers=['standard error y fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_se(self) :
        return None

    @column(headers=['standard error z fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_se(self) :
        return None

    @column(headers=['standard error amplitude fit [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def amp_se(self) :
        return None

    @column(headers=['standard error offset fit [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def os_se(self) :
        return None

    @column(headers=['chi2'], dtype=np.float32, save=True, agg='max')
    def chi2(self) :
        return None

    @column(headers=['iteration number'], dtype=np.float32, save=True, agg='max')
    def n_iter(self) :
        return None

    @column(headers=['converged'], dtype=np.bool_, save=True, agg='min')
    def converged(self) :
        return None



    # --- Photometry ---

    @column(headers=['amplitude [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def amp(self) :
        return "amp_fit"

    @column(headers=['offset [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def os(self) :
        return "os_fit"

    @column(headers=['intensity [photon]', 'phot'], dtype=np.float32, save=True, agg='mean')
    def intensity(self) :
        return "gaussian_intensity"

    @column(headers=['snr'], dtype=np.float32, save=False, agg='mean')
    def snr(self) :
        return self.intensity / np.sqrt(self.os * self.x_cropshape * self.y_cropshape + self.intensity + self.read_noise**2 * self.x_cropshape * self.y_cropshape)

    @column(headers=['snr peak'], dtype=np.float32, save=False, agg='mean')
    def snr_peak(self) :
        return self.amp / np.sqrt(self.os + self.amp + self.read_noise**2)

    @column(headers=['sbr'], dtype=np.float32, save=False, agg='mean')
    def sbr(self) :
        return self.intensity / (self.os * self.x_cropshape * self.y_cropshape)

    @column(headers=['sbr peak'], dtype=np.float32, save=False, agg='mean')
    def sbr_peak(self) :
        return self.amp / self.os



    # --- Gaussians ---

    @column(headers=['gaussian intensity [photon]'], dtype=np.float32, save=False, agg='mean')
    def gaussian_signal(self) :
        return self.amp * 2 * np.pi * (self.sigma_x / self.x_pixel) * (self.sigma_y / self.y_pixel)

    @column(headers=['gaussian sigma [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma(self) :
        if self.locs.config.model not in ['isogaussian', 'gaussian'] :
            return np.sqrt(self.nphotons / 2 / np.pi / self.amp)
        return np.sqrt(self.sigma_x * self.sigma_y)

    @column(headers=['gaussian sigma x [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma_x(self) :
        return "sigma_x_fit"

    @column(headers=['gaussian sigma y [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma_y(self) :
        return "sigma_y_fit"

    @column(headers=['gaussian sigma fit [nm]', 'sigma [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma_fit(self) :
        return "sigma"

    @column(headers=['gaussian sigma x fit [nm]', 'sigma x [nm]', 'sigma_x [nm]', 'PSFxnm'], dtype=np.float32, save=True, agg='mean')
    def sigma_x_fit(self) :
        return "sigma_fit"

    @column(headers=['gaussian sigma y fit [nm]', 'sigma y [nm]', 'sigma_y [nm]', 'PSFynm'], dtype=np.float32, save=True, agg='mean')
    def sigma_y_fit(self) :
        return "sigma_fit"

    @column(headers=['sigma ratio', 'ellipticity'], dtype=np.float32, save=False, agg='mean')
    def sigma_ratio(self) :
        return self.sigma_x / self.sigma_y

    @column(headers=['sigma angle [deg]', 'angle [deg]'], dtype=np.float32, save=True, agg='mean')
    def sigma_angle(self) :
        return "psf_theta"



    # --- CRLB ---

    @column(headers=['crlb xy [nm]', 'uncertainty_xy', 'uncertainty_xy [nm]'], dtype=np.float32, save=False, agg='mean')
    def crlb(self) :
        return np.sqrt((self.crlb_x**2 + self.crlb_y**2) / 2)

    @column(headers=['crlb x [nm]', 'uncertainty_x', 'uncertainty_x [nm]'], dtype=np.float32, save=False, agg='mean')
    def crlb_x(self) :
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

    @column(headers=['crlb y [nm]', 'uncertainty_y', 'uncertainty_y [nm]'], dtype=np.float32, save=False, agg='mean')
    def crlb_y(self) :
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

    @column(headers=[f'no zernike'], dtype=np.float32, save=False, agg='mean')
    def nozernike(self) :
        return np.zeros(self.ndetections, dtype=np.float32)

    @property
    def nzernike(self) :
        return self._nzernike
    @nzernike.setter
    def nzernike(self, value) :
        self._nzernike = int(value)
        cls = self.__class__
        for i in range(1, int(value) +1) :
            @column(headers=[f'zernike {i:02}'], dtype=np.float32, save=True, agg='mean')
            def zernike(self) :
                return "nozernike"
            setattr(cls, f'zernike_{i:02}', zernike)


