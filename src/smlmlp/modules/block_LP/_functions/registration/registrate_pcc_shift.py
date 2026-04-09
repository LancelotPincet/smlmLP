#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from arrlp import img_crosscorr
import numpy as np



# %% Function
@block()
def registrate_pcc_shift(optimized, /, ref_pix=1., *, cuda=False, parallel=False) :
    '''
    This function calculates the redundant shifts between optimized channels.
    '''

    # Calculates ref_pix
    try :
        if len(ref_pix) != 2 :
            raise ValueError('ref_pix does not have 2 vaes (y, x)')
    except TypeError:
        ref_pix = (ref_pix, ref_pix)

    # Phase cross-correlation
    CC, shiftx, shifty = [], [], []
    for i in range(len(optimized)) :
        for j in range(i+1, len(optimized)) :
            cc = img_crosscorr(optimized[i], optimized[j], phase=True, cuda=cuda, parallel=parallel, stack=True)
            CC.append(cc)
            dx, dy = subpixel_peak_stack(cc, ref_pix=ref_pix)
            shiftx.append(dx)
            shifty.append(dy)
    return CC, shiftx, shifty



def subpixel_peak_stack(cc, ref_pix=(1., 1.)):
    nframes = cc.shape[0]
    shiftx = np.empty(nframes, dtype=float)
    shifty = np.empty(nframes, dtype=float)

    ny, nx = cc.shape[-2:]

    for k in range(nframes):
        c = cc[k]

        iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)

        if not (0 < iy < ny - 1 and 0 < ix < nx - 1):
            # fallback: no subpixel refinement if peak touches border
            dy_sub, dx_sub = 0.0, 0.0
        else:
            win = c[iy-1:iy+2, ix-1:ix+2].astype(float)
            dy_sub, dx_sub = subpixel_peak_2d(win)

        dy = ((iy - ny // 2) + dy_sub) * ref_pix[0]
        dx = ((ix - nx // 2) + dx_sub) * ref_pix[1]

        shiftx[k] = dx
        shifty[k] = dy

    return shiftx, shifty


def subpixel_peak_2d(win):

    # Coordinates relative to center
    y, x = np.mgrid[-1:2, -1:2]

    # Flatten
    X = np.column_stack([
        x.ravel()**2,
        y.ravel()**2,
        x.ravel()*y.ravel(),
        x.ravel(),
        y.ravel(),
        np.ones(9)
    ])
    Z = win.ravel()

    # Solve least squares for coefficients
    coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    a, b, c, d, e, f = coeffs

    # Solve for stationary point of quadratic surface
    A = np.array([[2*a, c],
                  [c,   2*b]])
    bvec = -np.array([d, e])

    try:
        offset = np.linalg.solve(A, bvec)
    except np.linalg.LinAlgError:
        offset = np.array([0.0, 0.0])  # fallback if singular

    dx_sub, dy_sub = offset
    return dy_sub, dx_sub
