#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np

from smlmlp import DataFrame, column


class pixels(DataFrame) :
    """Pixel-level dataframe aggregated from blinks."""

    @column(headers=['pixel'], dtype=np.uint32, fill=0, save=True, agg='min', index="blinks")
    def pix(self) :
        """Compute flattened pixel identifiers from xy coordinates."""
        return np.round(self.y / self.y_pixel) * self.x_shape + np.round(self.x / self.x_pixel)



    # Maps

    @column(headers=['wide field [photon.pix-2]'], dtype=np.float32, fill=0, save=True, agg='mean')
    def wf(self) :
        """Sample the wide-field image at localization positions."""
        from smlmlp import image_picker

        return image_picker(self.locs.config.wf_image, locs=self.locs)[0]

    @column(headers=['irradiance [photon.pix-2]'], dtype=np.float32, fill=0, save=True, agg='mean')
    def irradiance(self) :
        """Sample irradiance image or fall back to offset values."""
        if self.locs.config.irradiance_image is not None :
            from smlmlp import image_picker

            return image_picker(self.locs.config.irradiance_image, locs=self.locs)[0]
        return "os"
