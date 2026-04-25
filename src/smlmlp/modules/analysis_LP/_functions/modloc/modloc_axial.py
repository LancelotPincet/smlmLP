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
def modloc_axial(x_eff, y_eff, phase, *, modloc_transverse_angle_deg=0., modloc_axial_angle_deg=45., cuda=False, parallel=False) :
    '''
    TODO.
    '''
    raise SyntaxeError('Not implemented')
    None, {}