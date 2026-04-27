#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

"""
Define channel-level optical and registration metadata.

The Channel class stores properties for individual optical channels within
a camera, including PSF parameters, registration transforms, and fit settings.
"""

import numpy as np

from arrlp import coordinates, transform_matrix
from corelp import prop
from smlmlp import Camera


class Channel:
    """
    Define channel-level optical and registration metadata.

    Parameters
    ----------
    camera : Camera
        Parent camera object.

    Attributes
    ----------
    camera_index : int
        Index of this channel within its camera.
    psf_sigma_nm : float
        Geometric mean of x and y PSF sigmas in nm.
    psf_wl_na_nm : float
        Effective wavelength/NA proxy in nm.
    spatial_kernel : ndarray
        Normalized spatial filtering kernel.
    """

    metadata = [
        ("flip", "Cameras"),
        ("psf_xsigma_nm", "Blinks"),
        ("psf_ysigma_nm", "Blinks"),
        ("psf_theta_deg", "Blinks"),
        ("psf_xtangents", "Data"),
        ("psf_ytangents", "Data"),
        ("psf_spline_coeffs", "Data"),
        ("psf_3d_xtangents", "Data"),
        ("psf_3d_ytangents", "Data"),
        ("psf_3d_ztangents", "Data"),
        ("psf_3d_spline_coeffs", "Data"),
        ("fit_theta", "Localizations"),
        ("fit_model", "Localizations"),
        ("x_shift_nm", "Registration"),
        ("y_shift_nm", "Registration"),
        ("rotation_deg", "Registration"),
        ("x_shear", "Registration"),
        ("y_shear", "Registration"),
        ("wf_image", "Data"),
    ]
    properties = [
        "channel_index",
        "psf_sigma_nm",
        "psf_radius_nm",
        "psf_diameter_nm",
        "psf_fwhm_nm",
        "spatial_kernel",
        "spatial_kernel_shape",
        "psf_kernel",
        "default_crop_nm",
        "crop_pix",
        "fit_init",
        "image_transform_matrix",
        "locs_transform_matrix",
    ]

    def __init__(self, camera):
        """Initialize the channel with its parent camera."""
        self.camera = camera

    @property
    def channel_index(self):
        """Return the index of this channel within its camera."""
        for i in range(self.camera.nchannels):
            if self.camera.channels[i] is self:
                return i

    # Bounding box

    @property
    def bbox(self):
        """Return the bounding box for this channel."""
        return self.bboxes[self.channel_index]

    @prop(iterable=2, dtype=bool)
    def flip(self):
        """Return whether to flip the channel data."""
        return False

    # PSF properties

    @prop()
    def psf_xtangents(self):
        """Return spline x tangents for PSF model."""
        return None

    @prop()
    def psf_ytangents(self):
        """Return spline y tangents for PSF model."""
        return None

    @prop()
    def psf_spline_coeffs(self):
        """Return spline coefficients for PSF model."""
        return None

    @prop()
    def psf_3d_xtangents(self):
        """Return spline x tangents for PSF 3D model."""
        return None

    @prop()
    def psf_3d_ytangents(self):
        """Return spline y tangents for PSF 3D model."""
        return None

    @prop()
    def psf_3d_ztangents(self):
        """Return spline z tangents for PSF 3D model."""
        return None

    @prop()
    def psf_3d_spline_coeffs(self):
        """Return spline coefficients for PSF 3D model."""
        return None

    @prop()
    def psf_xsigma_nm(self):
        """Return PSF sigma in x direction in nm."""
        return 0.21 * 670 / 1.5

    @prop()
    def psf_ysigma_nm(self):
        """Return PSF sigma in y direction in nm."""
        return 0.21 * 670 / 1.5

    @prop()
    def psf_theta_deg(self):
        """Return PSF rotation angle in degrees."""
        return 0.0

    @property
    def psf_sigma_nm(self):
        """Return geometric mean of PSF sigmas in nm."""
        return np.sqrt(self.psf_xsigma_nm * self.psf_ysigma_nm)

    @psf_sigma_nm.setter
    def psf_sigma_nm(self, value):
        """Set both x and y PSF sigmas to the same value."""
        self.psf_xsigma_nm, self.psf_ysigma_nm = value, value

    @property
    def psf_wl_na_nm(self):
        """Return effective wavelength/NA proxy in nm."""
        return self.psf_sigma_nm / 0.21

    @psf_wl_na_nm.setter
    def psf_wl_na_nm(self, value):
        """Set PSF sigma from wavelength/NA proxy."""
        self.psf_sigma_nm = 0.21 * value

    @property
    def psf_radius_nm(self):
        """Return PSF radius (Airy disk) in nm."""
        return 0.61 * self.psf_wl_na_nm

    @psf_radius_nm.setter
    def psf_radius_nm(self, value):
        """Set PSF parameters from radius."""
        self.psf_wl_na_nm = value / 0.61

    @property
    def psf_diameter_nm(self):
        """Return PSF diameter (Airy disk) in nm."""
        return 1.22 * self.psf_wl_na_nm

    @psf_diameter_nm.setter
    def psf_diameter_nm(self, value):
        """Set PSF parameters from diameter."""
        self.psf_wl_na_nm = value / 1.22

    @property
    def psf_fwhm_nm(self):
        """Return PSF full-width at half-maximum in nm."""
        return 0.51 * self.psf_wl_na_nm

    @psf_fwhm_nm.setter
    def psf_fwhm_nm(self, value):
        """Set PSF parameters from FWHM."""
        self.psf_wl_na_nm = value / 0.51

    # Background properties

    @property
    def mean_radius_pix(self):
        """Return mean radius for background estimation in pixels."""
        rad_nm = self.camera.config.mean_radius_nm
        return rad_nm / self.pixel_nm[0], rad_nm / self.pixel_nm[1]

    @property
    def opening_radius_pix(self):
        """Return radius for morphological opening in pixels."""
        rad_nm = self.camera.config.opening_radius_nm
        return rad_nm / self.pixel_nm[0], rad_nm / self.pixel_nm[1]

    # Spatial kernel

    @property
    def spatial_subtract_factor(self):
        """Return factor for spatial subtraction kernel."""
        return self.camera.config.spatial_subtract_factor

    @property
    def spatial_kernel_shape(self):
        """Return shape of the spatial kernel array."""
        sigma_pix = max(self.psf_xsigma_nm, self.psf_ysigma_nm) / min(self.pixel_nm)
        if self.spatial_subtract_factor > 1:
            sigma_pix *= self.spatial_subtract_factor
        sigma_pix = int(np.ceil(sigma_pix))
        return (sigma_pix * 6 + 1, sigma_pix * 6 + 1)

    @property
    def psf_kernel(self):
        """Return the normalized PSF kernel for this channel."""
        Y, X = coordinates(
            shape=self.spatial_kernel_shape, pixel=self.pixel_nm, grid=False
        )
        if (
            self.psf_xtangents is not None
            and self.psf_ytangents is not None
            and self.psf_spline_coeffs is not None
        ):
            from funclp import Spline2D

            tx, ty = np.asarray(self.psf_xtangents), np.asarray(
                self.psf_ytangents
            )
            coeffs = np.asarray(self.psf_spline_coeffs).reshape(
                ty.size - 4, tx.size - 4
            )
            spline = Spline2D(tx=tx, ty=ty, coeffs=coeffs)
            k = spline(X, Y)
        else:
            from funclp import Gaussian2D

            gaussian = Gaussian2D(
                sigx=self.psf_xsigma_nm,
                sigy=self.psf_ysigma_nm,
                theta=self.psf_theta_deg,
                pixx=self.pixel_nm[1],
                pixy=self.pixel_nm[0],
            )
            k = gaussian(X, Y)
        k /= k.sum()
        return k.astype(np.float32)

    @property
    def spatial_subtract_kernel(self):
        """Return the normalized spatial subtraction kernel if enabled."""
        if self.spatial_subtract_factor <= 1:
            return None
        pixel = (
            self.pixel_nm[0] / self.spatial_subtract_factor,
            self.pixel_nm[1] / self.spatial_subtract_factor,
        )
        Y, X = coordinates(shape=self.spatial_kernel_shape, pixel=pixel, grid=False)
        if (
            self.psf_xtangents is not None
            and self.psf_ytangents is not None
            and self.psf_spline_coeffs is not None
        ):
            from funclp import Spline2D

            tx, ty = np.asarray(self.psf_xtangents), np.asarray(
                self.psf_ytangents
            )
            coeffs = np.asarray(self.psf_spline_coeffs).reshape(
                ty.size - 4, tx.size - 4
            )
            spline = Spline2D(tx=tx, ty=ty, coeffs=coeffs)
            k = spline(X, Y)
        else:
            from funclp import Gaussian2D

            gaussian = Gaussian2D(
                sigx=self.psf_xsigma_nm,
                sigy=self.psf_ysigma_nm,
                theta=self.psf_theta_deg,
                pixx=self.pixel_nm[1],
                pixy=self.pixel_nm[0],
            )
            k = gaussian(X, Y)
        k /= k.sum()
        return k.astype(np.float32)

    @prop(cache=True)
    def spatial_kernel(self):
        """Return the effective spatial filtering kernel."""
        if self.spatial_subtract_factor <= 1:
            return self.psf_kernel
        k = self.psf_kernel - self.spatial_subtract_kernel
        return k - k.mean()

    # Crops

    @property
    def default_crop_nm(self):
        """Return default crop size in nm."""
        h = self.psf_ysigma_nm / self.pixel_nm[0] * 8
        w = self.psf_xsigma_nm / self.pixel_nm[1] * 8
        return int(2 * (h // 2) + 1) * self.pixel_nm[0], int(
            2 * (w // 2) + 1
        ) * self.pixel_nm[1]

    @property
    def crop_pix(self):
        """Return crop size in pixels."""
        crop_nm = self.camera.config.crop_nm
        h = crop_nm / self.pixel_nm[0]
        w = crop_nm / self.pixel_nm[1]
        return int(2 * (h // 2) + 1), int(2 * (w // 2) + 1)

    # Fit settings

    @prop()
    def fit_theta(self):
        """Return whether to fit theta parameter."""
        return False

    @prop()
    def fit_model(self):
        """Return the fitting model name."""
        return "isogauss"

    @fit_model.setter
    def fit_model(self, value):
        """Set the fitting model name."""
        value = str(value).lower()
        if value not in ["isogauss", "gauss", "spline"]:
            raise ValueError(f"{value} fitting model not recognized")
        self._fit_model = value

    @prop()
    def fit_init(self):
        """Return initial values for fitting based on model."""
        match self.fit_model:
            case "isogauss":
                return {"sig": self.psf_sigma_nm}
            case "gauss":
                return {
                    "sigx": self.psf_xsigma_nm,
                    "sigy": self.psf_ysigma_nm,
                    "theta": self.psf_theta_deg,
                    "theta_fit": self.fit_theta,
                }
            case "spline":
                return {
                    "tx": self.psf_3d_xtangents,
                    "ty": self.psf_3d_ytangents,
                    "tz": self.psf_3d_ztangents,
                    "coeffs": self.psf_3d_spline_coeffs,
                }
            case _:
                raise ValueError(
                    f"{self.fit_model} fitting model not recognized"
                )

    @fit_init.setter
    def fit_init(self, value):
        """Set initial fitting values, validating against model defaults."""
        init = self.fit_init
        if any(key not in init for key in value.keys()):
            raise KeyError("input fit initialization invalid")
        init.update(value)
        self._fit_init = init

    # Registration properties

    @prop()
    def x_shift_nm(self):
        """Return x shift in nm for registration."""
        return 0.0

    @prop()
    def y_shift_nm(self):
        """Return y shift in nm for registration."""
        return 0.0

    @prop()
    def rotation_deg(self):
        """Return rotation angle in degrees for registration."""
        return 0.0

    @prop()
    def x_shear(self):
        """Return x shear for registration."""
        return 0.0

    @prop()
    def y_shear(self):
        """Return y shear for registration."""
        return 0.0

    @prop()
    def image_transform_matrix(self):
        """Return transform matrix for image coordinates."""
        shape = self.bbox[3] - self.bbox[1], self.bbox[2] - self.bbox[0]
        return transform_matrix(
            shape=shape,
            shiftx=self.x_shift_nm / self.pixel_nm[1],
            shifty=self.y_shift_nm / self.pixel_nm[0],
            shearx=self.x_shear,
            sheary=self.y_shear,
            angle=self.rotation_deg,
        )

    @prop()
    def locs_transform_matrix(self):
        """Return transform matrix for localization coordinates."""
        shape = (self.bbox[3] - self.bbox[1]) * self.pixel_nm[0], (
            self.bbox[2] - self.bbox[0]
        ) * self.pixel_nm[1]
        return transform_matrix(
            shape=shape,
            shiftx=self.x_shift_nm,
            shifty=self.y_shift_nm,
            shearx=self.x_shear,
            sheary=self.y_shear,
            angle=self.rotation_deg,
        )

    # Image properties

    @prop()
    def wf_image(self):
        """Return widefield image for this channel."""
        if self.locs is not None:
            from smlmlp import image_pixel

            return image_pixel(self.locs.intensity, self.psf_sigma_nm, locs=self.locs)[
                0
            ]
        return None


# Adding Camera metadata dynamically
for data, _ in Camera.metadata:

    @property
    def camera_property(self, data=data):
        """Return camera property."""
        return getattr(self.camera, data)

    setattr(Channel, data, camera_property)


for data in Camera.properties:

    @property
    def camera_property(self, data=data):
        """Return camera property."""
        return getattr(self.camera, data)

    setattr(Channel, data, camera_property)
