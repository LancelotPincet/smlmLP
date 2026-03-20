#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import get_xp, gc
import bottleneck as bn
from scipy.optimize import curve_fit
from funclp import Gaussian



# %% Function
@block()
def bkgd_snr(channels, bkgds, noise_corrections=None, /, channel_gain=0.25, *, nbins=60, cuda=False, parallel=False) :
    '''
    This function calculates assess the background quality from camera gain.
    '''

    xp = get_xp(cuda)
    if noise_corrections is None : noise_corrections = [1. for _ in range(len(channels))]

    snr_list, mean_list, std_list, histox_list, histoy_list, fitsig_list, fitmu_list = [], [], [], [], [], [], []
    for i in range(len(channels)) :
        channel = xp.asarray(channels[i])
        bkgd = xp.asarray(bkgds[i])
        noise_correction = noise_corrections[i]
        gain = channel_gain[i]

        # snr calculations
        noise = noise_correction * xp.sqrt(xp.maximum(bkgd / gain))
        signal = channel - bkgd
        snr = signal / noise
        del(noise)
        del(signal)
        snr_flat = snr.ravel()
        gc()
        snr_list.append(snr)

        # snr statistics
        snr_flat = snr.ravel()
        snr_mean = xp.median(snr_flat) if cuda else bn.median(snr_flat)
        snr_std = xp.median(xp.abs(snr_flat - snr_median)) / 0.6745 if cuda else bn.median(xp.abs(snr_flat - snr_median)) / 0.6745 # MAD
        mean_list.append(snr_mean)
        std_list.append(snr_std)

        # snr histogram
        histoy, edges = np.histogram(z_flat, bins=nbins+1, range=(-10, 10), density=True)
        histox = (edges[:-1] + edges[1:]) / 2
        gaussian = Gaussian(amp=histoy.max(), sig=snr_std, mu=snr_mean)
        func2fit = lambda x, amp, mu, sig : gaussian(x, amp=amp, mu=mu, sig=sig)
        mask = (histox > -5) & (histox < 5)
        p0 = [histoy.max(), snr_mean, snr_std] #amp, mu, sig
        popt, pcov = curve_fit(gaussian, histox[mask], histoy[mask], p0=p0)
        fit_sig, fit_mu = popt[2], popt[1]
        histox_list.append(histox)
        histoy_list.append(histoy)
        fitsig_list.append(fit_sig)
        fitmu_list.append(fit_mu)

    return snr_list, dict(mean=mean_list, std=std_list, histo_x=histox_list, histo_y=histoy_list, fit_sig=fitsig_list, fit_mu=fitmu_list)