#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from corelp import prop
from smlmlp import Camera
import numpy as np
from arrlp import coordinates




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
    properties = ["psf_sigma", "psf_radius", "psf_diameter", "psf_fwhm", "spatial_kernel"]



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




    # Spatial kernel

    @property
    def spatial_subtract_factor(self) :
        return self.camera.config.spatial_subtract_factor

    @property
    def spatial_kernel_shape(self) :
        sigma_pix = max(self.psf_sigx, self.psf_sigy) / min(self.pixel)
        if self.spatial_subtract_factor is not None : sigma_pix *= self.spatial_subtract_factor
        sigma_pix = int(np.ceil(sigma_pix))
        return (sigma_pix * 6 + 1, sigma_pix * 6 + 1)

    @property
    def psf_kernel(self) :
        Y, X = coordinates(shape=self.spatial_kernel_shape, pixel=self.pixel, grid=False)
        if self.psf_tx is not None and self.psf_ty is not None and self.psf_coeffs is not None :
            from funclp import Spline2D
            tx, ty = np.asarray(self.psf_tx), np.asarray(self.psf_ty)
            coeffs = np.asarray(self.psf_coeffs).reshape(ty.size-4, tx.size-4) # 4 = k + 1 (order + 1)
            spline = Spline2D(tx=tx, ty=ty, coeffs=coeffs)
            k = spline(X, Y)
        else :
            from funclp import Gaussian2D
            gaussian = Gaussian2D(sigx=self.psf_sigx, sigy=self.psf_sigy, theta=self.psf_theta, pixx=self.pixel[1], pixy=self.pixel[0])
            k = gaussian(X, Y)
        k -= k.min()
        k /= k.sum()
        return k

    @property
    def spatial_subtract_kernel(self) :
        if self.spatial_subtract_factor is None : return None
        pixel = self.pixel[0] / self.spatial_subtract_factor, self.pixel[1] / self.spatial_subtract_factor
        Y, X = coordinates(shape=self.spatial_kernel_shape, pixel=pixel, grid=False)
        if self.psf_tx is not None and self.psf_ty is not None and self.psf_coeffs is not None :
            from funclp import Spline2D
            tx, ty = np.asarray(self.psf_tx), np.asarray(self.psf_ty)
            coeffs = np.asarray(self.psf_coeffs).reshape(ty.size-4, tx.size-4) # 4 = k + 1 (order + 1)
            spline = Spline2D(tx=tx, ty=ty, coeffs=coeffs)
            k = spline(X, Y)
        else :
            from funclp import Gaussian2D
            gaussian = Gaussian2D(sigx=self.psf_sigx, sigy=self.psf_sigy, theta=self.psf_theta, pixx=self.pixel[1], pixy=self.pixel[0])
            k = gaussian(X, Y)
        k -= k.min()
        k /= k.sum()
        return k

    @prop(cache=True)
    def spatial_kernel(self) :
        if self.spatial_subtract_factor is None : return self.psf_kernel
        return self.psf_kernel - self.spatial_subtract_kernel



# Adding Camera metadata
for data, _ in Camera.metadata :
    @property
    def camera_property(self, data=data) :
        return getattr(self.camera, data)
    setattr(Channel, data, camera_property)



