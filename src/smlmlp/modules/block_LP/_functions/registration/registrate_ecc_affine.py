#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_crosscorr, get_xp
import numpy as np



# %% Function
@block()
def registrate_ecc_shift(
    optimized,
    /,
    ref_pix=1.0,
    *,
    cuda=False,
    parallel=False,
):
    """
    Estimate redundant pairwise shifts from enhanced correlation coefficient images.

    """
    return TODO