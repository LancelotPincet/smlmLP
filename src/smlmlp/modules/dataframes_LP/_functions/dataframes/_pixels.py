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

    @column(headers=['pixel'], save=True, agg='min', index="blinks")
    def pix(self:np.uint32) :
        from smlmlp import index_pixels
        return index_pixels(locs=self.locs)[0]



    # Maps

    @column(headers=['wide field [photons]'], save=True, agg='mean')
    def wf(self:np.float32) :
        from smlmlp import image_picker
        return image_picker(self.wf_image, locs=self.locs)[0]

    @column(headers=['irradiance [kw.cm-2]'], save=True, agg='mean')
    def irradiance(self:np.float32) :
        from smlmlp import image_picker
        return image_picker(self.irradiance_image, locs=self.locs)[0]


