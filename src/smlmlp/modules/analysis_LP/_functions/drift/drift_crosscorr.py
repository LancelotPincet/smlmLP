#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import analysis
import numpy as np
import numba as nb



# %% Function
@analysis(df_name="points")
def drift_crosscorr(x, y, crosscorr_frames_per_segment=1000, *, crosscorr_outlier_fraction=0.1, crosscorr_recompute=True, pixel_sr_nm=15., cuda=False, parallel=False) :
    '''
    TODO.
    '''
    raise SyntaxeError('Not implemented')
    None, None, None, {}