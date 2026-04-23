#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import analysis
import numpy as np
import numba as nb



# %% Function
@analysis(df_name="detections")
def index_pixels(fr, *, cuda=False, parallel=False) :
    '''
    '''
    return np.round(self.y / self.y_pixel) * self.x_shape + np.round(self.x / self.x_pixel)
