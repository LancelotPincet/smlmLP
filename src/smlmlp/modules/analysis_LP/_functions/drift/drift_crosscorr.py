#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import analysis


@analysis(df_name="points")
def drift_crosscorr(x, y, crosscorr_frames_per_segment=1000, *, crosscorr_outlier_fraction=0.1, crosscorr_recompute=True, pixel_sr_nm=15., cuda=False, parallel=False) :
    """
    Placeholder for drift crosscorr.

    Raises
    ------
    SyntaxError
        Always raised because this analysis is not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")
