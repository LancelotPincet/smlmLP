#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, get_xp, img_autocorr, coordinates
from funclp import Gaussian2D
import numpy as np
from scipy.optimize import curve_fit



# %% Function
@block()
def spatial_autocorr(channels, crop=20, *,  channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # Optimization parameters
    xp = get_xp(cuda)
    parallel = False if cuda else parallel

    # Get pixel
    pixel = Config(nfiles=len(channels), pixel=channel_pixel).pixel

    # Coordinates
    Y, X = coordinates((2*crop+1, 2*crop+1), grid=True)
    mask = np.logical_and(Y != 0, X != 0)

    psf_sigma = []
    results = {'ac': [], 'fit': [], 'sigx': [], 'sigy': [], 'theta': []}
    for channel, pix in zip(channels, pixel) :

        # Calculating autocorrelation
        gc()
        bkgd = xp.median(channel, axis=(0,), keepdims=True)
        bkgd = xp.minimum(bkgd, channel)
        channel = channel - bkgd
        y0, x0 = int(channel.shape[1]//2), int(channel.shape[2]//2)
        ac = img_autocorr(channel, stacks=True, cuda=cuda, parallel=parallel)
        ac = ac[:, y0 - crop : y0 + crop + 1, x0 - crop : x0 + crop + 1]
        ac = ac.mean(axis=0)
        if cuda : ac = xp.asnumpy(ac)
        ac[crop, crop] = np.nan
        ac -= np.nanmin(ac)
        ac /= np.nanmax(ac)
        results['ac'].append(ac)

        # Fitting
        gc()
        _ac = ac[mask]
        yy = Y[mask] * pix[0] / min(pix)
        xx = X[mask] * pix[1] / min(pix)
        p0 = [2., 2., 1., 0., 0.] # sigx, sigy, amp, offset, theta [°]
        bounds = ([0., 0., 0.75, -0.25, 0.], [crop, crop, 1.25, 0.25, 90.]) # (lower, upper)
        gaus = Gaussian2D(pixx=1., pixy=1., sigx=2., sigy=2.)
        func2fit = lambda xy, sigx, sigy, amp, offset, theta: gaus(xy[0], xy[1], sigx=sigx, sigy=sigy, amp=amp, offset=offset, theta=theta)
        popt, _ = curve_fit(func2fit, (xx, yy), _ac, p0=p0, bounds=bounds)
        sigx, sigy, amp, offset, theta = popt
        sigx, sigy = sigx * min(pix), sigy * min(pix)
        results['fit'].append(func2fit((X * pix[1] / min(pix), Y * pix[0] / min(pix)), *popt))

        # Calculating PSF width
        results['sigx'].append(sigx)
        results['sigy'].append(sigy)
        results['theta'].append(theta)
        _psf_sigma = np.sqrt(sigx * sigy) / np.sqrt(2)
        psf_sigma.append(_psf_sigma)

    return psf_sigma, results