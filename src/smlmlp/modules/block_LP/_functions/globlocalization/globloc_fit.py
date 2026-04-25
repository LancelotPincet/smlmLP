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
def globloc_fit(
    crops,
    X0,
    Y0,
    /,
    *,
    x_shift_nm=0.,
    y_shift_nm=0.,
    rotation_deg=0.,
    x_shear=0.,
    y_shear=0.,
    optimizer="lm",
    estimator="mle",
    distribution="poisson",
    channels_pixels_nm=1.0,
    channels_gains=1.0,
    channels_QE=1.0,
    cuda=False,
    parallel=False,
):
    """
    Create from the channels list the global channel where to do detections.
    """
    TODO