#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp, sortloop
import bottleneck as bn
import numpy as np
from scipy.stats import linregress



# %% Function
@block()
def detect_gain(channels, /, nbins=50, *, cuda=False, parallel=False) :
    '''
    This function calculates the gain maps.
    '''

    xp = get_xp(cuda)
    gain_maps, mean_maps, var_maps = [], [], []
    mean_bins, var_bins = [], []
    gains, fits = [], []

    for channel in channels :
        channel = xp.asarray(channel)
        med = xp.median(channel, axis=0) if cuda else bn.median(channel, axis=0)
        mad = xp.median(xp.abs(channel-med), axis=0) if cuda else bn.median(xp.abs(channel-med), axis=0)
        mean = med
        var = (1.4826 * mad)**2
        if cuda: mean, var = xp.asnumpy(mean), xp.asnumpy(var)
        gain_maps.append(mean / var)
        mean_maps.append(mean)
        var_maps.append(var)
        
        # Bin
        bins = np.linspace(mean.min(), mean.max(), nbins+1)
        digitized = np.digitize(mean, bins)
        mean_binned = []
        var_binned = []
        for _, _, mean_m, var_m in sortloop(digitized, mean, var) :
            if len(mean_m) > 10:
                mean_binned.append(mean_m.mean())
                var_binned.append(var_m.mean())
        mean_binned, var_binned = np.array(mean_binned), np.array(var_binned)
        mean_bins.append(mean_binned)
        var_bins.append(var_binned)

        # Fit
        mask = np.logical_and(mean_binned < np.percentile(mean, 66), mean_binned > np.percentile(mean, 33))
        fit = linregress(mean_binned[mask], (var_binned[mask]))
        gain = 1.0 / fit.slope
        gains.append(gain)
        fits.append(fit)

    return gains, dict(gain=gain_maps, mean=mean_maps, var=var_maps, mean_bins=mean_bins, var_bins=var_bins, fit=fits)