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
def flim_dpflim(x, y, *, cuda=False, parallel=False) :
    '''
    TODO.
    '''
    pass