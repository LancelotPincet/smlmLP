#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import analysis


@analysis(df_name="points")
def drift_comet(x, y, comet_frames_per_segment=10., *, comet_recompute=True, comet_max_drift_nm=300., comet_tol=1e-4, cuda=False, parallel=False) :
    """
    Placeholder for drift comet.

    Raises
    ------
    SyntaxError
        Always raised because this analysis is not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")
