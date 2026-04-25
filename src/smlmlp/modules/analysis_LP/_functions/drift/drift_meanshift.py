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
def drift_meanshift(x, y, meanshift_frames_per_segment=10., *, meanshift_outlier_fraction=0.1, meanshift_recompute=True, meanshift_max_iter=100, meanshift_tol_nm=1., meanshift_max_drift_nm=300., cuda=False, parallel=False) :
    '''
    TODO.
    '''
    raise SyntaxeError('Not implemented')
    None, None, None, {}