#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
import numpy as np
import bottleneck as bn



@block()
def registrate_solve_redundant_affine(
    shiftx,
    shifty,
    angle,
    shearx,
    sheary,
    /,
    sigma_thresh=3.0,
    max_outliers=None,
    *,
    cuda=False,
    parallel=False,
):
    """
    TODO
    """