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
def index_points(fr, *, cuda=False, parallel=False) :
    '''
    '''
    return fr2ch(fr)
