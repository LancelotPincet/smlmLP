#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from corelp import prop
from smlmlp import Camera




# %% Function
class Channel :
    '''
    Defines channel instance
    '''

    metadata = []
    properties = []



    def __init__(self, camera) :
        self.camera = camera

    # PSF
    @prop()
    def psf_wl_na(self) : # [nm]
        return 670 / 1.5 # default value
    @prop()
    def psf_eccentricity(self) :
        return 0.
    @prop()
    def psf_theta(self) : # [°]
        return 0.
    @property
    def psf_sigma(self) : # nm
        return 0.21 * self.psf_wl_na
    @psf_sigma.setter
    def psf_sigma(self, value) :
        self.psf_wl_na = value / 0.21
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
for data in Camera.metadata :
    @property
    def camera_property(self, data=data) :
        return getattr(self.camera, data)
    setattr(Channel, data, camera_property)



