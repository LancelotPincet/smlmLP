#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Config

"""
This class stores configuration values in an object that can be saved and loaded.
"""



# %% Libraries
from corelp import prop, selfkwargs
from pathlib import Path
import json
import numpy as np
from smlmlp import metadatum, Camera
from contextlib import ExitStack
import tifffile as tiff
from stacklp import shapetif



# %% Class
class Config() :
    '''
    This class stores configuration values in an object that can be saved and loaded.
    
    Parameters
    ----------
    a : int or float
        TODO.

    Attributes
    ----------
    _attr : int or float
        TODO.

    Examples
    --------
    >>> from smlmlp import Config
    ...
    >>> instance = Config(TODO)
    '''


    # Init
    def __init__(self, *tif_paths, config=None, **kwargs) :

        # Opening tif files
        self.nfiles = len(tif_paths)
        if self.nfiles == 0 : raise SyntaxError('Configuration object can only be initialized with tif files')

        # Loading previous configuration
        if config is not None :
            if isinstance(config, str) or isinstance(config,Path('').__class__):
                config_file = Path(config).with_suffix('.json')
                if config_file.exists() :
                    with open(config_file, "r") as file:
                        config = json.load(file)
                else :
                    raise SyntaxError(f'config path was not recognized: {config}')
            for group_name, group_dict in config.items() :
                selfkwargs(self, group_dict)
        selfkwargs(self, kwargs)

        # Opening files
        with ExitStack() as stack :
            tifs = [stack.enter_context(tiff.TiffFile(file)) for file in tif_paths]
            shapes = [shapetif(tif) for tif in tifs]

        # Check number of frames
        for shape in shapes :
            if shape[0] != shapes[0][0] :
                raise ValueError('All tiff files do not have the same number of frames which is not possible for a single SMLM acquisition')
        self.nframes = shapes[0][0]
        self.npixels = [shape[1:3] for shape in shapes]



    @property
    def metadata(self) :
        data = {}
        for group_name, group_list in metadatum.groups.items() :
            data_dict = {}
            data[group_name] = data_dict
            for datum in group_list :
                value = getattr(self, datum)
                data_dict[datum] = json_convert(value)
        return data



    def save(self, path, file=None) :
        path = Path(path)
        if file is not None : path = path / file
        path = path.with_suffix('.json')
        with open(path, 'w') as json_file:
            json.dump(self.metadata, json_file, indent=4)



    # --- Below are the metadata, organized by group---



    # Cameras

    @metadatum('Cameras')
    def nfiles(self) :
        return len(self.cameras)
    @nfiles.setter
    def nfiles(self, value) :
        self.cameras = [Camera() for _ in range(value)]

    @property
    def FOV_max(self) :
        ymin = min([camera.FOV_max[0] for camera in self.cameras])
        xmin = min([camera.FOV_max[1] for camera in self.cameras])
        return ymin, xmin

    @property
    def bbox(self) :
        FOV = self.FOV_max
        return [camera.FOV2bbox(FOV) for camera in self.cameras]

    @property
    def frame_bytes(self) : # gigabytes/frame
        return sum([camera.frame_bytes for camera in self.cameras])
    

    # Loads

    @metadatum('Loads', dtype=int)
    def nframes(self) :
        return 60000



# Adding Camera metadata
for data in Camera.metadata :
    @metadatum('Cameras', name=data)
    def camera_property(self, data=data) :
        return [getattr(camera, data) for camera in self.cameras]
    @camera_property.setter
    def camera_property(self, value, data=data) :
        if len(value) != self.nfiles : raise ValueError('Value set does not have same number of elements as cameras')
        for camera, v in zip(self.cameras, value) :
            setattr(camera, data, v)
    setattr(Config, data, camera_property)



# Convert to json function
def json_convert(value) :
    if isinstance(value, tuple) : value = list(value)
    if isinstance(value, list) : value = [json_convert(v) for v in value]
    if isinstance(value, np.ndarray) : value = list(value)
    return value



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)