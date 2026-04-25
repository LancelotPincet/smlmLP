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
def modloc_demodulated(intensity, pnt, ch, *, modloc_channels_indices=[1, 2, 3, 4], modloc_dephases_rad=[0, np.pi/2, np.pi, 3*np.pi/2], cuda=False, parallel=False) :
    '''
    TODO.
    '''
    raise SyntaxeError('Not implemented')
    None, {}