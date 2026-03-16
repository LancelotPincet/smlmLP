#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from corelp import prop
from smlmlp import Camera
import numpy as np




# %% Function
class Channel :
    '''
    Defines channel instance
    '''

    metadata = [ # (metadata, group)
        ("flip", "Cameras"),
        ("psf_sigx", "Blinks"),
        ("psf_sigy", "Blinks"),
        ("psf_theta", "Blinks"),
        ("psf_tx", "Blinks"),
        ("psf_ty", "Blinks"),
        ("psf_coeffs", "Blinks"),
        ]
    properties = ["psf_sigma", "psf_radius", "psf_diameter", "psf_fwhm"]



    def __init__(self, camera) :
        self.camera = camera



    # Bounding box
    @prop(iterable=2, dtype=bool)
    def flip(self) :
        return False

    # PSF
    @prop()
    def psf_tx(self) : # spline
        return None
    @prop()
    def psf_ty(self) : # spline
        return None
    @prop()
    def psf_coeffs(self) : # spline
        return None
    @prop()
    def psf_sigx(self) : # [nm]
        return 0.21 * 670 / 1.5
    @prop()
    def psf_sigy(self) : # [nm]
        return 0.21 * 670 / 1.5
    @prop()
    def psf_theta(self) : # [°]
        return 0.
    @property
    def psf_sigma(self) : # [nm]
        return np.sqrt(self.psf_sigx * self.psf_sigy)
    @psf_sigma.setter
    def psf_sigma(self, value) :
        self.psf_sigx, self.psf_sigy = value, value
    @property
    def psf_wl_na(self) : # [nm]
        return self.sigma / 0.21
    @psf_wl_na.setter
    def psf_wl_na(self, value) :
        self.psf_sigma = 0.21 * value
    @property
    def psf_radius(self) : # nm
        return 0.61 * self.psf_wl_na
    @psf_radius.setter
    def psf_radius(self, value) :
        self.psf_wl_na = value / 0.61
    @property
    def psf_diameter(self) : # nm
        return 1.22 * self.psf_wl_na
    @psf_diameter.setter
    def psf_diameter(self, value) :
        self.psf_wl_na = value / 1.22
    @property
    def psf_fwhm(self) : # nm
        return 0.51 * self.psf_wl_na
    @psf_fwhm.setter
    def psf_fwhm(self, value) :
        self.psf_wl_na = value / 0.51



# Adding Camera metadata
for data, _ in Camera.metadata :
    @property
    def camera_property(self, data=data) :
        return getattr(self.camera, data)
    setattr(Channel, data, camera_property)



