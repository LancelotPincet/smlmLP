#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, img_autocorr
from funclp import Gaussian2D



# %% Function
@block()
def spatial_autocorr(channels, crop=20, channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the spatial local mean background.
    '''

    # Optimization parameters
    xp = get_cuda(cuda)
    parallel = False if cuda else parallel

    # Get pixel
    pixel = Config(nfiles=len(channels), pixel=channel_pixel).pixel

    # Coordinates
    Y, X = coordinates((2*crop+1, 2*crop+1), grid=True)
    mask = np.logical_and(Y != 0, X != 0)

    psf_sigma = []
    results = {'ac': [], 'fit': [], 'sigx': [], 'sigy': [], 'theta': []}
    for i, (channel, pix) in enumerate(zip(channels, pixel)) :

        # Calculating autocorrelation
        gc()
        bkgd = xp.median(channel, axis=(0,), keepdims=True)
        bkgd = xp.minimum(bkgd, channel)
        channel = channel - bkgd
        y0, x0 = int(channel.shape[1]//2), int(channel.shape[2]//2)
        ac = img_autocorr(_stack, stacks=True, cuda=cuda, parallel=parallel)
        ac = ac[:, y0 - crop : y0 + crop + 1, x0 - crop : x0 + crop + 1]
        ac = ac.mean(axis=0)
        if cuda : ac = xp.asnumpy(ac)
        ac[spatial_crop, spatial_crop] = np.nan
        ac -= np.nanmin(ac)
        ac /= np.nanmax(ac)
        results['ac'].append(ac)

        # Fitting
        gc()
        _ac = ac[mask]
        yy = Y[mask]
        xx = X[mask]
        p0 = [2., 2., 1., 0., 0.] # sigx, sigy, amp, offset, theta [°]
        bounds = ([0., 0., 0.75, -0.25, 0.], [crop, crop, 1.25, 0.25, 45.]) # (lower, upper)
        gaus = Gaussian2D(pixx=1., pixy=1., sigx=2., sigy=2.)
        func2fit = lambda xy, sigx, sigy, amp, offset, theta: gaus(xy[0], xy[1], sigx=sigx, sigy=sigy, amp=amp, offset=offset, theta=theta)
        popt, _ = curve_fit(func2fit, (xx, yy), _ac, p0=p0, bounds=_bounds)
        sigx, sigy, amp, offset, theta = popt
        results['fit'].append(func2fit((X, Y), *popt))

        # Calculating PSF width
        theta_rad = np.radians(theta)
        sigma_y = np.sqrt((sigx * np.sin(theta_rad))**2 + (sigy * np.cos(theta_rad))**2) * pix[0]
        sigma_x = np.sqrt((sigx * np.cos(theta_rad))**2 + (sigy * np.sin(theta_rad))**2) * pix[1]
        results['sigx'].append(sigma_x)
        results['sigy'].append(sigma_y)
        results['theta'].append(theta)
        _psf_sigma = np.sqrt(sigma_x / np.sqrt(2) * sigma_y / np.sqrt(2))
        psf_sigma.append(_psf_sigma)

    return psf_sigma, results