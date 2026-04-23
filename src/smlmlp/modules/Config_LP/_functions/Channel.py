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
        ("psf_xsigma_nm", "Blinks"),
        ("psf_ysigma_nm", "Blinks"),
        ("psf_theta_deg", "Blinks"),
        ("psf_xtangents", "Data"),
        ("psf_ytangents", "Data"),
        ("psf_spline_coeffs", "Data"),
        ("x_shift_nm", "Registration"),
        ("y_shift_nm", "Registration"),
        ("rotation_deg", "Registration"),
        ("x_shear", "Registration"),
        ("y_shear", "Registration"),
        ("wf_image", "Data"),
        ]
    properties = ["psf_sigma_nm", "psf_radius_nm", "psf_diameter_nm", "psf_fwhm_nm", "spatial_kernel", "spatial_kernel_shape", "psf_kernel", "default_crop_nm", "crop_pix"]



    def __init__(self, camera) :
        self.camera = camera



    # Bounding box

    @prop(iterable=2, dtype=bool)
    def flip(self) :
        return False



    # PSF

    @prop()
    def psf_xtangents(self) : # spline
        return None
    @prop()
    def psf_ytangents(self) : # spline
        return None
    @prop()
    def psf_spline_coeffs(self) : # spline
        return None

    @prop()
    def psf_xsigma_nm(self) : # [nm]
        return 0.21 * 670 / 1.5
    @prop()
    def psf_ysigma_nm(self) : # [nm]
        return 0.21 * 670 / 1.5
    @prop()
    def psf_theta_deg(self) : # [°]
        return 0.

    @property
    def psf_sigma_nm(self) : # [nm]
        return np.sqrt(self.psf_xsigma_nm * self.psf_ysigma_nm)
    @psf_sigma_nm.setter
    def psf_sigma_nm(self, value) :
        self.psf_xsigma_nm, self.psf_ysigma_nm = value, value
    @property
    def psf_wl_na_nm(self) : # [nm]
        return self.sigma_nm / 0.21
    @psf_wl_na_nm.setter
    def psf_wl_na_nm(self, value) :
        self.psf_sigma_nm = 0.21 * value
    @property
    def psf_radius_nm(self) : # nm
        return 0.61 * self.psf_wl_na_nm
    @psf_radius_nm.setter
    def psf_radius_nm(self, value) :
        self.psf_wl_na_nm = value / 0.61
    @property
    def psf_diameter_nm(self) : # nm
        return 1.22 * self.psf_wl_na_nm
    @psf_diameter_nm.setter
    def psf_diameter_nm(self, value) :
        self.psf_wl_na_nm = value / 1.22
    @property
    def psf_fwhm_nm(self) : # nm
        return 0.51 * self.psf_wl_na_nm
    @psf_fwhm_nm.setter
    def psf_fwhm_nm(self, value) :
        self.psf_wl_na_nm = value / 0.51



    # Backgrounds

    @property
    def mean_radius_pix(self) :
        rad_nm = self.camera.config.mean_radius_nm
        return rad_nm / self.pixel_nm[0], rad_nm / self.pixel_nm[1]

    @property
    def opening_radius_pix(self) :
        rad_nm = self.camera.config.opening_radius_nm
        return rad_nm / self.pixel_nm[0], rad_nm / self.pixel_nm[1]



    # Spatial kernel

    @property
    def spatial_subtract_factor(self) :
        return self.camera.config.spatial_subtract_factor

    @property
    def spatial_kernel_shape(self) :
        sigma_pix = max(self.psf_xsigma_nm, self.psf_ysigma_nm) / min(self.pixel_nm)
        if self.spatial_subtract_factor > 1 : sigma_pix *= self.spatial_subtract_factor
        sigma_pix = int(np.ceil(sigma_pix))
        return (sigma_pix * 6 + 1, sigma_pix * 6 + 1)

    @property
    def psf_kernel(self) :
        Y, X = coordinates(shape=self.spatial_kernel_shape, pixel=self.pixel_nm, grid=False)
        if self.psf_xtangents is not None and self.psf_ytangents is not None and self.psf_spline_coeffs is not None :
            from funclp import Spline2D
            tx, ty = np.asarray(self.psf_xtangents), np.asarray(self.psf_ytangents)
            coeffs = np.asarray(self.psf_spline_coeffs).reshape(ty.size-4, tx.size-4) # 4 = k + 1 (order + 1)
            spline = Spline2D(tx=tx, ty=ty, coeffs=coeffs)
            k = spline(X, Y)
        else :
            from funclp import Gaussian2D
            gaussian = Gaussian2D(sigx=self.psf_xsigma_nm, sigy=self.psf_ysigma_nm, theta=self.psf_theta_deg, pixx=self.pixel_nm[1], pixy=self.pixel_nm[0])
            k = gaussian(X, Y)
        k /= k.sum()
        return k.astype(np.float32)

    @property
    def spatial_subtract_kernel(self) :
        if self.spatial_subtract_factor <= 1 : return None
        pixel = self.pixel_nm[0] / self.spatial_subtract_factor, self.pixel_nm[1] / self.spatial_subtract_factor
        Y, X = coordinates(shape=self.spatial_kernel_shape, pixel=pixel, grid=False)
        if self.psf_xtangents is not None and self.psf_ytangents is not None and self.psf_spline_coeffs is not None :
            from funclp import Spline2D
            tx, ty = np.asarray(self.psf_xtangents), np.asarray(self.psf_ytangents)
            coeffs = np.asarray(self.psf_spline_coeffs).reshape(ty.size-4, tx.size-4) # 4 = k + 1 (order + 1)
            spline = Spline2D(tx=tx, ty=ty, coeffs=coeffs)
            k = spline(X, Y)
        else :
            from funclp import Gaussian2D
            gaussian = Gaussian2D(sigx=self.psf_xsigma_nm, sigy=self.psf_ysigma_nm, theta=self.psf_theta_deg, pixx=self.pixel_nm[1], pixy=self.pixel_nm[0])
            k = gaussian(X, Y)
        k /= k.sum()
        return k.astype(np.float32)

    @prop(cache=True)
    def spatial_kernel(self) :
        if self.spatial_subtract_factor <= 1 :
            return self.psf_kernel
        k = self.psf_kernel - self.spatial_subtract_kernel
        return k - k.mean()



    # Crops

    @property
    def default_crop_nm(self) : # nm
        h = self.psf_ysigma_nm / self.pixel_nm[0] * 8
        w = self.psf_xsigma_nm / self.pixel_nm[1] * 8
        return int(2*(h//2)+1) * self.pixel_nm[0], int(2*(w//2)+1) * self.pixel_nm[1]

    @property
    def crop_pix(self) : # pixel
        crop_nm = self.camera.config.crop_nm
        h = crop_nm / self.pixel_nm[0]
        w = crop_nm / self.pixel_nm[1]
        return int(2*(h//2)+1), int(2*(w//2)+1)



    # Registration

    @prop()
    def x_shift_nm(self) : # nm
        return 0.
    @prop()
    def y_shift_nm(self) : # nm
        return 0.
    @prop()
    def rotation_deg(self) : # deg
        return 0.
    @prop()
    def x_shear(self) :
        return 0.
    @prop()
    def y_shear(self) :
        return 0.



    # Image

    @prop()
    def wf_image(self) : # 2D image
        return None

# Adding Camera metadata
for data, _ in Camera.metadata :
    @property
    def camera_property(self, data=data) :
        return getattr(self.camera, data)
    setattr(Channel, data, camera_property)
for data in Camera.properties :
    @property
    def camera_property(self, data=data) :
        return getattr(self.camera, data)
    setattr(Channel, data, camera_property)


