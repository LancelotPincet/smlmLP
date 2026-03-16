#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, Config
from arrlp import gc, get_xp, img_autocorr, img_fft, img_ifft, coordinates
from funclp import Gaussian2D, Spline2D
import numpy as np
from scipy.optimize import curve_fit



# %% Function
@block()
def blink_psf(channels, /, crop=20, *,  channel_pixel=1., cuda=False, parallel=False) :
    '''
    This function creates the global psf image and characterizations from the channels spatial autocorrelations.
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
    results = {'ac': [], 'psf':[], 'fit': [], 'spline': [], 'sigx': [], 'sigy': [], 'theta': [], 'tx': [], 'ty': [], 'coeffs': []}
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
        mid, ac[crop, crop] = ac[crop, crop], np.nan
        ac -= np.nanmin(ac)
        maxi = np.nanmax(ac)
        ac[crop, crop] = mid
        ac /= maxi

        # Calculating PSF
        gc()
        psf = np.fft.fftshift(np.real(img_ifft(np.sqrt(np.abs(img_fft(ac))))))
        mid, psf[crop, crop] = psf[crop, crop], np.nan
        psf -= np.nanmin(psf)
        maxi = np.nanmax(psf)
        psf[crop, crop] = mid
        psf /= maxi

        # Fitting
        gc()
        _psf = psf[mask]
        yy = Y[mask] * pix[0] / min(pix)
        xx = X[mask] * pix[1] / min(pix)
        p0 = [1., 1., 1., 0., 0.] # sigx, sigy, amp, offset, theta [°]
        bounds = ([0., 0., 0.75, -0.25, 0.], [crop, crop, 1.25, 0.25, 90.]) # (lower, upper)
        gaus = Gaussian2D(pixx=1., pixy=1., sigx=2., sigy=2.)
        func2fit = lambda xy, sigx, sigy, amp, offset, theta: gaus(xy[0], xy[1], sigx=sigx, sigy=sigy, amp=amp, offset=offset, theta=theta)
        popt, _ = curve_fit(func2fit, (xx, yy), _psf, p0=p0, bounds=bounds)
        sigx, sigy, amp, offset, theta = popt
        sigx, sigy = sigx * min(pix), sigy * min(pix)
        _X, _Y = X * pix[1] / min(pix), Y * pix[0] / min(pix)
        fit = func2fit((_X, _Y), *popt)
        maxi = fit.max()
        fit = (fit-offset) / amp
        psf = (psf-offset) / amp
        
        # Spline
        yy = Y * pix[0]
        xx = X * pix[1]
        mask = np.ones(psf.shape, dtype=bool)
        mask[crop, :] = False
        mask[:, crop] = False
        _psf = psf[mask].reshape(2*crop, 2*crop)
        _yy = yy[mask].reshape(2*crop, 2*crop)
        _xx = xx[mask].reshape(2*crop, 2*crop)
        spline_model = Spline2D(_psf, _xx, _yy)
        spline = spline_model(xx, yy)



        # Appending to results
        psf_sigma.append(np.sqrt(sigx * sigy))
        results['ac'].append(ac)
        results['psf'].append(psf)
        results['fit'].append(fit)
        results['spline'].append(spline)
        results['sigx'].append(sigx)
        results['sigy'].append(sigy)
        results['theta'].append(theta)
        results['tx'].append(spline_model.tx)
        results['ty'].append(spline_model.ty)
        results['coeffs'].append(spline_model.coeffs)

    return psf_sigma, results