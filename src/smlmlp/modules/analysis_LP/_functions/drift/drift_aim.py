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
def drift_aim(x, y, aim_frames_per_segment=10., *, aim_outlier_fraction=0.1, aim_recompute=True, aim_kde_bandwidth_nm=40., aim_learning_rate=0.5, aim_max_iter=200, aim_tol=1e-3, aim_lambda_smooth=0.5, cuda=False, parallel=False) :
    '''
    TODO.
    '''
    raise SyntaxeError('Not implemented')
    None, None, None, {}