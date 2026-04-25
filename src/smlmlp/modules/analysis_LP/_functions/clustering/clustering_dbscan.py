#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import analysis


@analysis(df_name="blinks")
def clustering_dbscan(xx, yy, *, dbscan_eps=50., dbscan_min_points=10., cuda=False, parallel=False) :
    """
    Placeholder for clustering dbscan.

    Raises
    ------
    SyntaxError
        Always raised because this analysis is not implemented yet.
    """
    raise SyntaxError("Not implemented yet.")
