#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class points(DataFrame) :
    """Point-level dataframe aggregated from detections."""

    @column(headers=['point'], dtype=np.uint64, save=True, agg='min', index="detections")
    def pnt(self) :
        """Assign point identifiers across channels."""
        if self.locs.config.nchannels == 1 :
            return self.locs.detections.det
        else :
            from smlmlp import associate_different_channels

            dets = self.locs.detections
            if dets.x_globfit is None or dets.y_globfit is None :
                return None
            return associate_different_channels(locs=self.locs)[0]
    


    # Coordinates

    @column(headers=['x stable [nm]'], dtype=np.float32, save=True, agg='mean')
    def xx(self) :
        """Return drift-corrected x coordinates."""
        if not self.dx_exists : return "x"
        return self.x - self.dx

    @column(headers=['y stable [nm]'], dtype=np.float32, save=True, agg='mean')
    def yy(self) :
        """Return drift-corrected y coordinates."""
        if not self.dy_exists : return "y"
        return self.y - self.dy

    @column(headers=['z stable [nm]'], dtype=np.float32, save=True, agg='mean')
    def zz(self) :
        """Return drift-corrected z coordinates."""
        if not self.dz_exists : return "z"
        return self.z - self.dz



    # xy

    @column(headers=['x [nm]', 'xnm'], dtype=np.float32, save=False, agg='mean')
    def x(self) :
        """Select the configured x localization value."""
        match self.locs.config.x_method :
            case "det" : return "x_globdet"
            case "fit" : return "x_eff"
            case "modloc" : return "x_modloc"
            case "timeloc" : return "x_timeloc"
            case _ : raise ValueError('x-method not recognized')

    @column(headers=['y [nm]', 'ynm'], dtype=np.float32, save=False, agg='mean')
    def y(self) :
        """Select the configured y localization value."""
        match self.locs.config.y_method :
            case "det" : return "y_globdet"
            case "fit" : return "y_eff"
            case "modloc" : return "y_modloc"
            case "timeloc" : return "y_timeloc"
            case _ : raise ValueError('y-method not recognized')

    @column(headers=['x modloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_modloc(self) :
        """Compute x position with transverse modulation localization."""
        from smlmlp import modloc_transverse

        self.x_modloc, self.y_modloc, _ = modloc_transverse(locs=self.locs)
        return "x_modloc"

    @column(headers=['y modloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_modloc(self) :
        """Compute y position with transverse modulation localization."""
        from smlmlp import modloc_transverse

        self.x_modloc, self.y_modloc, _ = modloc_transverse(locs=self.locs)
        return "y_modloc"

    @column(headers=['x timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def x_timeloc(self) :
        """Convert x frequency to x position."""
        if self.x_freq is None : return None
        from smlmlp import calibration_convert

        return calibration_convert(self.x_freq, self.locs.config.x_timeloc_calibration)[0]

    @column(headers=['y timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def y_timeloc(self) :
        """Convert y frequency to y position."""
        if self.y_freq is None : return None
        from smlmlp import calibration_convert

        return calibration_convert(self.y_freq, self.locs.config.y_timeloc_calibration)[0]



    # z

    @column(headers=['z [nm]', 'znm'], dtype=np.float32, save=False, agg='mean')
    def z(self) :
        """Select the configured z localization value."""
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
        """Compute z position with axial modulation localization."""
        from smlmlp import modloc_axial

        return modloc_axial(locs=self.locs)[0]

    @column(headers=['z timeloc [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_timeloc(self) :
        """Convert z frequency to z position."""
        if self.z_freq is None : return None
        from smlmlp import calibration_convert

        return calibration_convert(self.z_freq, self.locs.config.z_timeloc_calibration)[0]

    @column(headers=['z astigmatism [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_astig(self) :
        """Convert astigmatism ratio to z position."""
        from smlmlp import calibration_convert

        return calibration_convert(self.sigma_ratio, self.locs.config.z_astig_calibration)[0]

    @column(headers=['z biplane [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_biplane(self) :
        """Convert biplane ratio to z position."""
        from smlmlp import calibration_convert

        return calibration_convert(self.biplane_ratio, self.locs.config.z_biplane_calibration)[0]

    @column(headers=['z donald [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_donald(self) :
        """Convert DONALD ratio to z position."""
        from smlmlp import calibration_convert

        return calibration_convert(self.donald_ratio, self.locs.config.z_donald_calibration)[0]

    @column(headers=['z miet [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_miet(self) :
        """Convert MIET lifetime to z position."""
        from smlmlp import calibration_convert

        return calibration_convert(self.lifetime, self.locs.config.z_miet_calibration)[0]

    @column(headers=['z qtirf [nm]'], dtype=np.float32, save=True, agg='mean')
    def z_qtirf(self) :
        """Convert qTIRF intensity ratio to z position."""
        from smlmlp import calibration_convert

        return calibration_convert(self.intensity/self.irradiance, self.locs.config.z_qtirf_calibration)[0]



    # Orientation azimuth

    @column(headers=['azimuth [deg]'], dtype=np.float32, save=False, agg='mean')
    def azimuth(self) :
        """Select the configured azimuth estimate."""
        match self.locs.config.azimuth_method :
            case "polar2d" : return "azimuth_polar2d"
            case "polar3d" : return "azimuth_polar3d"
            case _ : raise ValueError('azimuth-method not recognized')

    @column(headers=['azimuth polar2d [deg]'], dtype=np.float32, save=True, agg='mean')
    def azimuth_polar2d(self) :
        """Compute azimuth with 2D polarization orientation."""
        from smlmlp import orient_polar2d

        return orient_polar2d(locs=self.locs)[0]

    @column(headers=['azimuth polar3d [deg]'], dtype=np.float32, save=True, agg='mean')
    def azimuth_polar3d(self) :
        """Compute azimuth with 3D polarization orientation."""
        from smlmlp import orient_polar3d

        self.azimuth_polar3d, self.tilt_polar3d = orient_polar3d(locs=self.locs)[0]
        return "azimuth_polar3d"



    # Orientation tilt

    @column(headers=['tilt [deg]'], dtype=np.float32, save=False, agg='mean')
    def tilt(self) :
        """Select the configured tilt estimate."""
        match self.locs.config.tilt_method :
            case "polar3d" : return "tilt_polar3d"
            case _ : raise ValueError('tilt-method not recognized')

    @column(headers=['tilt polar3d [deg]'], dtype=np.float32, save=True, agg='mean')
    def tilt_polar3d(self) :
        """Compute tilt with 3D polarization orientation."""
        from smlmlp import orient_polar3d

        self.azimuth_polar3d, self.tilt_polar3d = orient_polar3d(locs=self.locs)[0]
        return "tilt_polar3d"



    # Phase

    @column(headers=['phase [rad]'], dtype=np.float32, save=False, agg='mean')
    def phase(self) :
        """Select the configured modulation phase estimate."""
        match self.locs.config.phase_method :
            case "demodulated" : return "phase_demodulated"
            case "sequential" : return "phase_sequential"
            case _ : raise ValueError('phase-method not recognized')

    @column(headers=['phase demodulated [rad]'], dtype=np.float32, save=True, agg='mean')
    def phase_demodulated(self) :
        """Compute demodulated modulation phase."""
        from smlmlp import modloc_demodulated

        return modloc_demodulated(locs=self.locs)[0]

    @column(headers=['phase sequential [rad]'], dtype=np.float32, save=True, agg='mean')
    def phase_sequential(self) :
        """Compute sequential modulation phase."""
        from smlmlp import modloc_sequential

        return modloc_sequential(locs=self.locs)[0]



    # Lifetime

    @column(headers=['lifetime [ns]'], dtype=np.float32, save=False, agg='mean')
    def lifetime(self) :
        """Select the configured lifetime estimate."""
        match self.locs.config.lifetime_method :
            case "tcspc" : return "lifetime_tcspc"
            case "iflim" : return "lifetime_iflim"
            case "dpflim" : return "lifetime_dpflim"
            case _ : raise ValueError('lifetime-method not recognized')

    @column(headers=['lifetime tcspc [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_tcspc(self) :
        """Return explicit TCSPC lifetime values."""
        return None

    @column(headers=['lifetime iflim [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_iflim(self) :
        """Convert IFLIM ratio to lifetime."""
        from smlmlp import calibration_convert

        return calibration_convert(self.iflim_ratio, self.locs.config.lifetime_iflim_calibration)[0]

    @column(headers=['lifetime dpflim [ns]'], dtype=np.float32, save=True, agg='mean')
    def lifetime_dpflim(self) :
        """Convert DPFLIM ratio to lifetime."""
        from smlmlp import calibration_convert

        return calibration_convert(self.dpflim_ratio, self.locs.config.lifetime_dpflim_calibration)[0]



    # Frequency

    @column(headers=['x frequency [hz]'], dtype=np.float32, save=True, agg='mean')
    def x_freq(self) :
        """Return explicit x frequency values."""
        return None

    @column(headers=['y frequency [hz]'], dtype=np.float32, save=True, agg='mean')
    def y_freq(self) :
        """Return explicit y frequency values."""
        return None

    @column(headers=['z frequency [hz]'], dtype=np.float32, save=True, agg='mean')
    def z_freq(self) :
        """Return explicit z frequency values."""
        return None



    # Spectral

    @column(headers=['spectral ratio'], dtype=np.float32, save=True, agg='mean')
    def spectral_ratio(self) :
        """Compute spectral y/x ratio."""
        return self.spectral_y / self.spectral_x

    @column(headers=['spectral x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def spectral_x(self) :
        """Aggregate spectral x-channel intensity."""
        from smlmlp import aggregate_ratio

        self.spectral_x, self.spectral_y, _ = aggregate_ratio(self.intensity, x_channels=self.locs.config.spectral_x_channels, y_channels=self.locs.config.spectral_y_channels, locs=self.locs)
        return "spectral_x"

    @column(headers=['spectral y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def spectral_y(self) :
        """Aggregate spectral y-channel intensity."""
        from smlmlp import aggregate_ratio

        self.spectral_x, self.spectral_y, _ = aggregate_ratio(self.intensity, x_channels=self.locs.config.spectral_x_channels, y_channels=self.locs.config.spectral_y_channels, locs=self.locs)
        return "spectral_y"



    # Biplane

    @column(headers=['biplane ratio'], dtype=np.float32, save=True, agg='mean')
    def biplane_ratio(self) :
        """Compute biplane y/x ratio."""
        return self.biplane_y / self.biplane_x

    @column(headers=['biplane x width [nm]'], dtype=np.float32, save=True, agg='mean')
    def biplane_x(self) :
        """Aggregate biplane x-channel width."""
        from smlmlp import aggregate_ratio

        self.biplane_x, self.biplane_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.biplane_x_channels, y_channels=self.locs.config.biplane_y_channels, locs=self.locs)
        return "biplane_x"

    @column(headers=['biplane y width [nm]'], dtype=np.float32, save=True, agg='mean')
    def biplane_y(self) :
        """Aggregate biplane y-channel width."""
        from smlmlp import aggregate_ratio

        self.biplane_x, self.biplane_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.biplane_x_channels, y_channels=self.locs.config.biplane_y_channels, locs=self.locs)
        return "biplane_y"



    # DONALD

    @column(headers=['donald ratio'], dtype=np.float32, save=True, agg='mean')
    def donald_ratio(self) :
        """Compute DONALD y/x ratio."""
        return self.donald_y / self.donald_x

    @column(headers=['donald x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def donald_x(self) :
        """Aggregate DONALD x-channel value."""
        from smlmlp import aggregate_ratio

        self.donald_x, self.donald_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.donald_x_channels, y_channels=self.locs.config.donald_y_channels, locs=self.locs)
        return "donald_x"

    @column(headers=['donald y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def donald_y(self) :
        """Aggregate DONALD y-channel value."""
        from smlmlp import aggregate_ratio

        self.donald_x, self.donald_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.donald_x_channels, y_channels=self.locs.config.donald_y_channels, locs=self.locs)
        return "donald_y"



    # IFLIM

    @column(headers=['iflim ratio'], dtype=np.float32, save=True, agg='mean')
    def iflim_ratio(self) :
        """Compute IFLIM y/x ratio."""
        return self.iflim_y / self.iflim_x

    @column(headers=['iflim x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def iflim_x(self) :
        """Aggregate IFLIM x-channel value."""
        from smlmlp import aggregate_ratio

        self.iflim_x, self.iflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.iflim_x_channels, y_channels=self.locs.config.iflim_y_channels, locs=self.locs)
        return "iflim_x"

    @column(headers=['iflim y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def iflim_y(self) :
        """Aggregate IFLIM y-channel value."""
        from smlmlp import aggregate_ratio

        self.iflim_x, self.iflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.iflim_x_channels, y_channels=self.locs.config.iflim_y_channels, locs=self.locs)
        return "iflim_y"



    # DPFLIM

    @column(headers=['dpflim ratio'], dtype=np.float32, save=True, agg='mean')
    def dpflim_ratio(self) :
        """Compute DPFLIM y/x ratio."""
        return self.dpflim_y / self.dpflim_x

    @column(headers=['dpflim x intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def dpflim_x(self) :
        """Aggregate DPFLIM x-channel value."""
        from smlmlp import aggregate_ratio

        self.dpflim_x, self.dpflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.dpflim_x_channels, y_channels=self.locs.config.dpflim_y_channels, locs=self.locs)
        return "dpflim_x"

    @column(headers=['dpflim y intensity [photon]'], dtype=np.float32, save=True, agg='mean')
    def dpflim_y(self) :
        """Aggregate DPFLIM y-channel value."""
        from smlmlp import aggregate_ratio

        self.dpflim_x, self.dpflim_y, _ = aggregate_ratio(self.sigma, x_channels=self.locs.config.dpflim_x_channels, y_channels=self.locs.config.dpflim_y_channels, locs=self.locs)
        return "dpflim_y"
