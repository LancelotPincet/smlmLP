#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from funclp import LM, MLE, LSQ, Poisson, Normal, IsoGaussian
from arrlp import get_xp, nb_threads, coordinates
import numpy as np

SIGMA = 0.21*670/1.5

# %% Function
@block()
def locs_individual_isogaussfit(crops, X0, Y0, /, *, optimizer='lm', estimator='mle', distribution='poisson', channels_pixels_nm=1., channels_gains=1., channels_QE=1., cuda=False, parallel=False,
    channels_psf_sigmas_nm=SIGMA, # IsoGaussian
    ) :
    '''
    Calculates individual isogaussian fits of crops.
    '''

    # Calculate 
    try :
        if len(channels_pixels_nm) != len(crops) :
            if len(channels_pixels_nm) == 2 :
                channels_pixels_nm = [channels_pixels_nm for _ in range(len(crops))]
            else :
                raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_pixels_nm = [(channels_pixels_nm, channels_pixels_nm) for _ in range(len(crops))]
    try :
        if len(channels_gains) != len(crops) :
            raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_gains = [channels_gains for _ in range(len(crops))]
    try :
        if len(channels_QE) != len(crops) :
            raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_QE = [channels_QE for _ in range(len(crops))]

    # Optimizer
    match optimizer.lower() :
        case 'lm' :
            optimizer = LM
        case _ :
            raise SyntaxError(f'Optimizer {optimizer} is not recognized')

    # Distribution
    match distribution.lower() :
        case 'normal' :
            distribution = Normal()
        case 'poisson' :
            distribution = Poisson()
        case _ :
            raise SyntaxError(f'Distribution {distribution} is not recognized')

    # Estimator
    match estimator.lower() :
        case 'MLE' :
            estimator = MLE(distribution)
        case 'LSQ' :
            estimator = LSQ()
        case _ :
            raise SyntaxError(f'Estimator {estimator} is not recognized')
    
    # Function
    try :
        if len(channels_psf_sigmas_nm) != len(crops) :
            raise ValueError('channel_mean_radius_pix does not have the same length as channels')
    except TypeError:
        channels_psf_sigmas_nm = [channels_psf_sigmas_nm for _ in range(len(crops))]
    KW = [dict(
        sig = sig,
        pixx = pix[1],
        pixy = pix[0],
    ) for pix, sig in zip(channels_pixels_nm, channels_psf_sigmas_nm)]
    

    # Looping on crops
    xp = get_xp(cuda)
    Mux, Muy = [], []
    Amp, Offset = [], []
    Sigma = []
    for crop, x0, y0, pix, gain, qe, kw in zip(crops, X0, Y0, channels_pixels_nm, channels_gains, channels_QE, KW) :
        crop = xp.asarray(crop)
        N, Y, X = crop.shape
        yy, xx = coordinates(shape=(Y, X), pixel=pix, cuda=cuda)
        x0 = xp.asarray(x0) * pix[1]
        y0 = xp.asarray(y0) * pix[0]

        # Function
        mux = xp.full_like(x0, fill_value=(X-1)/2*pix[1])
        muy = xp.full_like(y0, fill_value=(Y-1)/2*pix[0])
        amp = xp.max(crop, axis=(1,2))
        offset=xp.min(crop, axis=(1,2))
        function = IsoGaussian(mux=mux, muy=muy, amp=amp, offset=offset, cuda=cuda, **kw)

        # Fit
        fit = optimizer(function, estimator)
        if cuda :
            fit(crop, xx, yy)
        else :
            with nb_threads(parallel) :
                fit(crop, xx, yy)
        
        # Get parameters
        mux, muy = function.mux, function.muy
        mux += x0
        muy += y0
        amp = function.amp / qe * gain
        offset = function.amp / qe * gain
        if cuda :
            mux = xp.asnumpy(mux)
            muy = xp.asnumpy(muy)
            amp = xp.asnumpy(amp)
            offset = xp.asnumpy(offset)
        Mux.append(mux)
        Muy.append(muy)
        Amp.append(amp)
        Offset.append(offset)

        # Get function specific parameters
        sig = function.sig
        if cuda :
            sig = xp.asnumpy(sig)
        Sigma.append(sig)
    
    # output
    output = dict(
        amp = np.hstack(Amp),
        offset = np.hstack(Offset),
        sigma = np.hstack(Sigma),
    )
    return np.hstack(Mux), np.hstack(Muy), output