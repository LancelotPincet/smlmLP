#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



import numpy as np
from smlmlp import analysis


@analysis(df_name="points")
def modloc_demodulated(intensity, pnt, ch, *, modloc_channels_indices=[1, 2, 3, 4], modloc_dephases_rad=[0, np.pi/2, np.pi, 3*np.pi/2], cuda=False, parallel=False) :
    """
    Placeholder for modloc demodulated.

    Raises
    ------
    SyntaxError
        Always raised because this analysis is not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")
