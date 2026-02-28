#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import Hardware
from corelp import prop
import numpy as np



# %% Function
class Objectives(Hardware) :
    tosave = ["X", "TL", "NA", "immersion", "sample", "obj_transmission"]

    # Magnitude
    @property
    def magnitude(self) : # Real magnitude
        return self.TL / self.focal
    @magnitude.setter
    def magnitude(self, value) :
        self.focal = self.TL / value
    @prop(dtype=float)
    def X(self) : # constructor magnitude
        return 100.

    # Focal
    @prop(dtype=float)
    def TL(self) : # Tube lens focal distance [mm]
        return self.default_TL
    @property
    def default_TL(self) : # Default tube lens focal distance [mm]
        if self.constructor is None : return 200.
        return self.TLs[self.constructor]
    Tls = dict(Nikon=200, Olympus=180, Evident=180, Zeiss=165, Leica=200)
    @property
    def focal(self) : # Effective Focal length [mm]
        return self.default_TL / self.X
    @focal.setter
    def focal(self, value) :
        self.X = self.default_TL / value



    # Aperture
    @prop(dtype=float)
    def NA(self) :
        return 1.5
    @prop()
    def immersion(self) :
        return 1.515
    @immersion.setter
    def immersion(self, value) :
        if isinstance(value, str) : value = self.immsersions[value]
        self._immersion = float(value)
    immersions = dict(air=1., water=1.33, glycerol=1.4, silicon=1.4, oil=1.515)
    sample = 1.33
    @property
    def collection_cone(self) : # [pi.sr]
        if self.immersion > self.sample : return 2.
        return 2 * (1 - np.cos(np.arcsin(self.immersion/self.sample)))



    # Transmission
    @prop()
    def obj_transmission(self) :
        self.load_spectra('obj_transmission', 1.)



    # Models
    models = {
        ["X", "TL", "NA", "immersion", "sample", "obj_transmission"]
        'MRD01991' : dict(constructor='Nikon', X=100, NA=1.49, immersion='oil')
        'MRD01691' : dict(constructor='Nikon', X=60, NA=1.49, immersion='oil')
        'UPLAPO100XOHR' : dict(constructor='Evident', X=100, NA=1.5, immersion='oil'),
        'UPLXAPO100XO' : dict(constructor='Evident', X=100, NA=1.5, immersion='oil'),
        'UPLAPO60XOHR' : dict(constructor='Evident', X=100, NA=1.5, immersion='oil'),
    }