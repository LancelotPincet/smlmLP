#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, get_xp, img_gaussianfilter
from funclp import Exponential1
from stacklp import temporal_autocorr as stack_autocorr
import numpy as np
from scipy.optimize import curve_fit



# %% Function
@block()
def blink_temporal_on(channels, crop_fr=None, /, psf_sigma_nm=100., *, exposure_ms=50., channels_pixels_nm=100., cuda=False, parallel=False) :
    '''
    This function creates the temporal autocorrelation for on time measurements.
    '''

    # Optimization parameters
    xp = get_xp(cuda)

    # Get pixel
    channels_pixels_nm = Config(ncameras=len(channels), cameras_pixels_nm=channels_pixels_nm).cameras_pixels_nm
    
    # Get coordinates
    if crop_fr is None: crop_fr = int(len(channels) // 2 - 1)
    T = np.arange(crop_fr)
    t = T * exposure_ms

    results = {'ac': []}
    for i, (channel, pix) in enumerate(zip(channels, channels_pixels_nm)) :

        # Calculating autocorrelation
        gc()
        bkgd = img_gaussianfilter(channel, sigma=psf_sigma_nm[i] * 3, pixel=pix, cuda=cuda, parallel=parallel)
        bkgd = xp.minimum(bkgd, channel)
        channel = channel - bkgd
        f0 = int(channel.shape[0]//2)
        ac = stack_autocorr(channel, cuda=cuda, parallel=parallel)[f0 + 1: f0 + 1 + crop_fr]
        ac = ac.mean(axis=(1,2))
        ac -= ac[-1]
        ac /= ac[0]
        if cuda : ac = xp.asnumpy(ac)
        results['ac'].append(ac)

    # Averaging
    gc()
    y = np.zeros_like(results['ac'][0])
    for ac in results['ac'] :
        y += (ac / len(results['ac']))
    results['average'] = y

    # Fitting
    p0 = [1., -0.5] # tau, offset
    bounds = ([0., -1.], [len(y), 1.]) # (lower, upper)
    expodecay = Exponential1()
    func2fit = lambda t, tau, offset: expodecay(t, tau=tau, offset=offset)
    popt, _ = curve_fit(func2fit, T, y, p0=p0, bounds=bounds)
    tau, offset = popt
    results['fit'] = func2fit(T, *popt)
    results['time'] = t
    on_time = tau * exposure_ms # ms

    return on_time, results