#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class pixels(DataFrame) :
    '''
    Pixels dataframe
    '''

    @column(headers=['pixel'], dtype=np.uint32, save=True, agg='min', index="blinks")
    def pix(self) :
        return np.round(self.y / self.y_pixel) * self.x_shape + np.round(self.x / self.x_pixel)



    # Maps

    @column(headers=['wide field [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def wf(self) :
        from smlmlp import image_picker
        return image_picker(self.config.wf_image, locs=self.locs)[0]

    @column(headers=['irradiance [photon.pix-2]'], dtype=np.float32, save=True, agg='mean')
    def irradiance(self) :
        if self.config.irradiance_image is not None :
            from smlmlp import image_picker
            return image_picker(self.config.irradiance_image, locs=self.locs)[0]
        return "os"


