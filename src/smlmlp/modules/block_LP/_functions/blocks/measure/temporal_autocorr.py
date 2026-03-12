#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, img_gaussianfilter
from funclp import Exponential1
from stacklp import temporal_autocorr as stack_autocorr



# %% Function
@block()
def temporal_autocorr(channels, psf_sigma, crop=None, exposure=1., channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the temporal autocorrelation for on time measurements.
    '''

    # Optimization parameters
    xp = get_cuda(cuda)
    parallel = False if cuda else parallel

    # Get pixel
    pixel = Config(nfiles=len(channels), pixel=channel_pixel).pixel
    
    # Get coordinates
    if crop is None: crop = int(len(channels) // 2 - 1)
    T = np.arange(crop)
    t = T * exposure
    mask = T != 0

    result = {'ac': []}
    for i, (channel, pix) in enumerate(zip(channels, pixel)) :

        # Calculating autocorrelation
        gc()
        bkgd = img_gaussianfilter(channel, sigma=psf_sigma[i] * 3, pixel=pix, cuda=cuda, parallel=parallel)
        bkgd = xp.minimum(bkgd, channel)
        channel = channel - bkgd
        f0 = int(channel.shape[0]//2)
        ac = stack_autocorr(channel, cuda=cuda, parallel=parallel)[f0 + 1: f0 + 1 + crop]
        ac = ac.mean(axis=(1,2))
        ac -= ac[-1]
        ac /= ac[0]
        if cuda : ac = xp.asnumpy(ac)
        result['ac'].append(ac)

    # Averaging
    gc()
    y = np.zeros_like(result['ac'][0])
    for ac in ac_list :
        y += (ac / len(result['ac']))
    result['average'] = y

    # Fitting
    p0 = [1., -0.5] # tau, offset
    bounds = ([0., -1.], [len(y), 1.]) # (lower, upper)
    expodecay = Exponential1()
    func2fit = lambda t, tau, offset: expodecay(t, tau=tau, offset=offset)
    popt, _ = curve_fit(func2fit, T, y, p0=p0, bounds=bounds)
    tau, offset = popt
    result['fitted'] = func2fit(T, *popt)
    result['time'] = t
    on_time = tau * exposure # ms

    return on_time, results