#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from corelp import prop
from smlmlp import Camera




# %% Function
class Channel :
    '''
    Defines channel instance
    '''

    metadata = []



    def __init__(self, camera) :
        self.camera = camera



# Adding Camera metadata
for data in Camera.metadata :
    @property
    def camera_property(self, data=data) :
        return getattr(self.camera, data)
    setattr(Channel, data, camera_property)



