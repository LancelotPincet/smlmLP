#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import MainDataFrame, column


class detections(MainDataFrame) :
    """Detection-level main dataframe."""

    @column(headers=['detection', 'id'], dtype=np.uint64, save=True, agg='min', index=True)
    def det(self) :
        """Return explicit detection identifiers when present."""
        return None



    # Filters

    @column(headers=['filter'], dtype=np.bool_, save=False, agg='min')
    def keep(self) :
        """Temporary filter mask used by Locs.filter."""
        return None



    # Detections

    @column(headers=['x global detection [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_globdet(self) :
        """Return explicit global x detection coordinates."""
        return None

    @column(headers=['y global detection [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_globdet(self) :
        """Return explicit global y detection coordinates."""
        return None

    @column(headers=['x detection [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_det(self) :
        """Return channel-local x detection coordinates."""
        if self.x_globdet is None or self.y_globdet is None : return None
        if self.config.nchannels == 1 : return "x_globdet"
        from smlmlp import inv_transform_locs

        self.x_det, self.y_det, _ = inv_transform_locs(locs=self.locs)
        return "x_det"

    @column(headers=['y detection [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_det(self) :
        """Return channel-local y detection coordinates."""
        if self.x_globdet is None or self.y_globdet is None : return None
        if self.config.nchannels == 1 : return "y_globdet"
        from smlmlp import inv_transform_locs

        self.x_det, self.y_det, _ = inv_transform_locs(locs=self.locs)
        return "y_det"



    # PSF fits

    @column(headers=['x fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_fit(self) :
        """Alias fitted x to detection x by default."""
        return "x_det"

    @column(headers=['y fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_fit(self) :
        """Alias fitted y to detection y by default."""
        return "y_det"

    @column(headers=['x global fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_globfit(self) :
        """Return global x fit coordinates."""
        if self.x_fit is None or self.y_fit is None : return None
        if self.config.nchannels == 1 : return "x_fit"
        from smlmlp import transform_locs

        self.x_globfit, self.y_globfit, _ = transform_locs(locs=self.locs)
        return "x_globfit"

    @column(headers=['y global fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_globfit(self) :
        """Return global y fit coordinates."""
        if self.x_fit is None or self.y_fit is None : return None
        if self.config.nchannels == 1 : return "y_fit"
        from smlmlp import transform_locs

        self.x_globfit, self.y_globfit, _ = transform_locs(locs=self.locs)
        return "y_globfit"

    @column(headers=['z fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_fit(self) :
        """Return explicit fitted z coordinates."""
        return None

    @column(headers=['amplitude fit [photon.pix-2]', 'amplitude [photon]'], dtype=np.float32, save=True, agg='mean')
    def amp_fit(self) :
        """Return explicit fitted amplitude."""
        return None

    @column(headers=['offset fit [photon.pix-2]', 'offset [photon]'], dtype=np.float32, save=True, agg='mean')
    def os_fit(self) :
        """Return explicit fitted offset."""
        return None

    @column(headers=['standard error x fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_se(self) :
        """Return explicit x standard error."""
        return None

    @column(headers=['standard error y fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_se(self) :
        """Return explicit y standard error."""
        return None

    @column(headers=['standard error z fit [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_se(self) :
        """Return explicit z standard error."""
        return None

    @column(headers=['standard error amplitude fit [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def amp_se(self) :
        """Return explicit amplitude standard error."""
        return None

    @column(headers=['standard error offset fit [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def os_se(self) :
        """Return explicit offset standard error."""
        return None

    @column(headers=['chi2'], dtype=np.float32, save=True, agg='max')
    def chi2(self) :
        """Return explicit chi-squared fit value."""
        return None

    @column(headers=['iteration number'], dtype=np.float32, save=True, agg='max')
    def n_iter(self) :
        """Return explicit fit iteration count."""
        return None

    @column(headers=['converged'], dtype=np.bool_, save=True, agg='min')
    def converged(self) :
        """Return explicit convergence flags."""
        return None



    # Effective

    @column(headers=['x effective [photon]'], dtype=np.float32, save=True, agg='mean')
    def x_eff(self) :
        """Return effective x values after channel masking."""
        if all([self.locs.config.x_channels[i]==i+1 for i in range(self.locs.config.nchannels)]) :
            return "x_globfit"
        array = np.copy(self.x_globfit)
        array[~np.isin(self.ch, self.locs.config.x_channels)] = np.nan
        return array

    @column(headers=['y effective [photon]'], dtype=np.float32, save=True, agg='mean')
    def y_eff(self) :
        """Return effective y values after channel masking."""
        if all([self.locs.config.y_channels[i]==i+1 for i in range(self.locs.config.nchannels)]) :
            return "y_globfit"
        array = np.copy(self.y_globfit)
        array[~np.isin(self.ch, self.locs.config.y_channels)] = np.nan
        return array

    @column(headers=['z effective [photon]'], dtype=np.float32, save=True, agg='mean')
    def z_eff(self) :
        """Return effective z values after channel masking."""
        if all([self.locs.config.z_channels[i]==i+1 for i in range(self.locs.config.nchannels)]) :
            return "z_fit"
        array = np.copy(self.z_fit)
        array[~np.isin(self.ch, self.locs.config.z_channels)] = np.nan
        return array

    @column(headers=['intensity effective [photon]'], dtype=np.float32, save=True, agg='mean')
    def intensity_eff(self) :
        """Return effective intensity values after channel masking."""
        if all([self.locs.config.intensity_channels[i]==i+1 for i in range(self.locs.config.nchannels)]) :
            return "intensity"
        array = np.copy(self.intensity)
        array[~np.isin(self.ch, self.locs.config.intensity_channels)] = np.nan
        return array



    # Photometry

    @column(headers=['amplitude [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def amp(self) :
        """Alias amplitude to fitted amplitude."""
        return "amp_fit"

    @column(headers=['offset [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def os(self) :
        """Alias offset to fitted offset."""
        return "os_fit"

    @column(headers=['intensity [photon]', 'phot'], dtype=np.float32, save=True, agg='mean')
    def intensity(self) :
        """Alias intensity to Gaussian signal."""
        return "gaussian_signal"

    @column(headers=['snr'], dtype=np.float32, save=False, agg='mean')
    def snr(self) :
        """Compute integrated signal-to-noise ratio."""
        return self.intensity / np.sqrt(self.os * self.x_cropshape * self.y_cropshape + self.intensity + self.read_noise**2 * self.x_cropshape * self.y_cropshape)

    @column(headers=['snr peak'], dtype=np.float32, save=False, agg='mean')
    def snr_peak(self) :
        """Compute peak signal-to-noise ratio."""
        return self.amp / np.sqrt(self.os + self.amp + self.read_noise**2)

    @column(headers=['sbr'], dtype=np.float32, save=False, agg='mean')
    def sbr(self) :
        """Compute integrated signal-to-background ratio."""
        return self.intensity / (self.os * self.x_cropshape * self.y_cropshape)

    @column(headers=['sbr peak'], dtype=np.float32, save=False, agg='mean')
    def sbr_peak(self) :
        """Compute peak signal-to-background ratio."""
        return self.amp / self.os



    # Gaussians

    @column(headers=['gaussian intensity [photon]'], dtype=np.float32, save=False, agg='mean')
    def gaussian_signal(self) :
        """Compute Gaussian integrated signal."""
        return self.amp * 2 * np.pi * (self.sigma_x / self.x_pixel) * (self.sigma_y / self.y_pixel)

    @column(headers=['gaussian sigma [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma(self) :
        """Compute effective Gaussian sigma."""
        if self.locs.config.model not in ['isogaussian', 'gaussian'] :
            return np.sqrt(self.nphotons / 2 / np.pi / self.amp)
        return np.sqrt(self.sigma_x * self.sigma_y)

    @column(headers=['gaussian sigma x [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma_x(self) :
        """Alias Gaussian x sigma to fitted x sigma."""
        return "sigma_x_fit"

    @column(headers=['gaussian sigma y [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma_y(self) :
        """Alias Gaussian y sigma to fitted y sigma."""
        return "sigma_y_fit"

    @column(headers=['gaussian sigma fit [nm]', 'sigma [nm]'], dtype=np.float32, save=True, agg='mean')
    def sigma_fit(self) :
        """Alias fitted Gaussian sigma to sigma."""
        return "sigma"

    @column(headers=['gaussian sigma x fit [nm]', 'sigma x [nm]', 'sigma_x [nm]', 'PSFxnm'], dtype=np.float32, save=True, agg='mean')
    def sigma_x_fit(self) :
        """Alias fitted x sigma to fitted sigma."""
        return "sigma_fit"

    @column(headers=['gaussian sigma y fit [nm]', 'sigma y [nm]', 'sigma_y [nm]', 'PSFynm'], dtype=np.float32, save=True, agg='mean')
    def sigma_y_fit(self) :
        """Alias fitted y sigma to fitted sigma."""
        return "sigma_fit"

    @column(headers=['sigma ratio', 'ellipticity'], dtype=np.float32, save=False, agg='mean')
    def sigma_ratio(self) :
        """Compute the x/y Gaussian sigma ratio."""
        return self.sigma_x / self.sigma_y

    @column(headers=['sigma angle [deg]', 'angle [deg]'], dtype=np.float32, save=True, agg='mean')
    def sigma_angle(self) :
        """Alias Gaussian sigma angle to PSF theta."""
        return "psf_theta"



    # CRLB

    @column(headers=['crlb xy [nm]', 'uncertainty_xy', 'uncertainty_xy [nm]'], dtype=np.float32, save=False, agg='mean')
    def crlb(self) :
        """Combine x and y CRLB values."""
        return np.sqrt((self.crlb_x**2 + self.crlb_y**2) / 2)

    @column(headers=['crlb x [nm]', 'uncertainty_x', 'uncertainty_x [nm]'], dtype=np.float32, save=False, agg='mean')
    def crlb_x(self) :
        """Compute x CRLB from Gaussian fit parameters."""
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
        """Compute y CRLB from Gaussian fit parameters."""
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



    # Aberrations

    @column(headers=[f'no zernike'], dtype=np.float32, save=False, agg='mean')
    def nozernike(self) :
        """Return a zero Zernike aberration column."""
        return np.zeros(self.ndetections, dtype=np.float32)

    @property
    def nzernike(self) :
        """Return the number of dynamic Zernike columns."""
        return self._nzernike

    @nzernike.setter
    def nzernike(self, value) :
        """Install dynamic Zernike columns."""
        self._nzernike = int(value)
        cls = self.__class__
        for i in range(1, int(value) +1) :
            @column(headers=[f'zernike {i:02}'], dtype=np.float32, save=True, agg='mean')
            def zernike(self) :
                """Alias a dynamic Zernike column to zero aberration."""
                return "nozernike"
            setattr(cls, f'zernike_{i:02}', zernike)



    # Simulations

    @column(headers=['x groundtruth [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_groundtruth(self) :
        """Return explicit x ground-truth coordinates."""
        return None

    @column(headers=['y groundtruth [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_groundtruth(self) :
        """Return explicit y ground-truth coordinates."""
        return None
