#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import analysis


@analysis(df_name="detections")
def modloc_axial(x_eff, y_eff, phase, *, modloc_transverse_angle_deg=0., modloc_axial_angle_deg=45., cuda=False, parallel=False) :
    """
    Placeholder for modloc axial.

    Raises
    ------
    SyntaxError
        Always raised because this analysis is not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")
