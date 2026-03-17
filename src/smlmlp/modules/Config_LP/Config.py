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
from corelp import prop, selfkwargs, folder
from pathlib import Path
import json
import numpy as np
from smlmlp import metadatum, Camera, Channel
from contextlib import ExitStack
import tifffile as tiff
from stacklp import shapetif
from arrlp import coordinates



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

        # Loading previous configuration
        if config is not None :
            if isinstance(config, Config) :
                config = config.metadata
            if isinstance(config, str) or isinstance(config,Path('').__class__):
                config_file = Path(config).with_suffix('.json')
                if config_file.exists() :
                    with open(config_file, "r") as file:
                        config = json.load(file)
                    config_folder = config_file.parent / "_config_data"
                    if config_folder.exists() :
                        data = {}
                        for file in config_folder.glob('*.npy') :
                            key = file.stem
                            data[key] = np.load(file)
                        config["Data"] = data
                else :
                    raise SyntaxError(f'config path was not recognized: {config}')
            for group_dict in config.values() :
                selfkwargs(self, group_dict)
        selfkwargs(self, kwargs)

        # Opening files
        if len(tif_paths) :
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
                value = getattr(self, f'_{datum}', None)
                if value is not None :
                    data_dict[datum] = json_convert(value)
        return data



    def save(self, path, file=None) :
        path = Path(path)
        if file is not None : path = path / file
        path = path.with_suffix('.json')
        metadata = self.metadata
        data = metadata.pop('Data')
        with open(path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        config_folder = folder(path.parent / "_config_data", warning=False)
        for key, value in data.items() :
            np.save(config_folder / f"{key}.npy", np.asarray(value))




    # --- Below are the metadata, organized by group---



    # Cameras

    @metadatum('Cameras')
    def nfiles(self) :
        raise SyntaxError('Should not be called')
    @nfiles.setter
    def nfiles(self, value) :
        self.cameras = [Camera(self) for _ in range(int(value))]
    @property
    def _nfiles(self) :
        return len(self.cameras)

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



    # Channels

    @property
    def total_nchannels(self) :
        return sum(self.nchannels)

    @property
    def channels(self) :
        return [channel for camera in self.cameras for channel in camera.channels]



    # Loads

    @metadatum('Loads', dtype=int)
    def nframes(self) :
        return 60000



    # Time

    @metadatum('Cameras', dtype=float)
    def exposure(self) : # [ms]
        return 50.

    @metadatum('Blinks')
    def on_time(self) : # ms
        return 50.



    # Background configurations

    @metadatum('Backgrounds', dtype=float)
    def median_window(self) : # For temporal median [ms]
        return self.on_time * 10

    @metadatum('Backgrounds', dtype=float)
    def mean_radius(self) : # For spatial mean [nm]
        return max(self.psf_sigma) * 8

    @metadatum('Backgrounds', dtype=float)
    def mini_radius(self) : # For spatial mini [nm]
        return max(self.psf_sigma)



    # Temporal kernel

    @metadatum('Signals')
    def spatial_subtract_factor(self) :
        return 0.
    @spatial_subtract_factor.setter
    def spatial_subtract_factor(self, value) :
        self._spatial_subtract_factor = value
        for channel in self.channels:
            channel._spatial_kernel = None

    @metadatum('Signals')
    def temporal_subtract_factor(self) :
        return 0.
    @temporal_subtract_factor.setter
    def temporal_subtract_factor(self, value) :
        self._temporal_subtract_factor = value
        self._temporal_kernel = None

    @property
    def temporal_kernel_shape(self) :
        on_frames = self.on_time / self.exposure
        if self.temporal_subtract_factor > 1 : on_frames *= self.temporal_subtract_factor
        on_frames = int(np.ceil(on_frames))
        return (on_frames * 10 + 1),

    @property
    def on_time_kernel(self) :
        T, = coordinates(shape=self.temporal_kernel_shape, pixel=self.exposure)
        from funclp import Exponential1
        exponential = Exponential1(tau=self.on_time)
        k = exponential(np.abs(T))
        k /= k.sum()
        return k

    @property
    def temporal_subtract_kernel(self) :
        if self.temporal_subtract_factor <= 1 : return None
        exposure = self.exposure / self.temporal_subtract_factor
        T, = coordinates(shape=self.temporal_kernel_shape, pixel=exposure)
        from funclp import Exponential1
        exponential = Exponential1(tau=self.on_time)
        k = exponential(np.abs(T))
        k /= k.sum()
        return k

    @prop(cache=True)
    def temporal_kernel(self) :
        if self.temporal_subtract_factor <= 1 :
            return self.on_time_kernel
        k = self.on_time_kernel - self.temporal_subtract_kernel
        return k - k.mean()
        


# Adding Camera metadata
for data, group in Camera.metadata :
    @metadatum(group, name=data)
    def camera_property(self, data=data) :
        return [getattr(camera, data) for camera in self.cameras]
    @camera_property.setter
    def camera_property(self, value, data=data) :
        try :
            if len(value) != self.nfiles : raise ValueError('Value set does not have same number of elements as cameras')
        except TypeError :
            value = [value for _ in range(self.nfiles)]            
        for camera, v in zip(self.cameras, value) :
            setattr(camera, data, v)
    setattr(Config, data, camera_property)
    @property
    def _camera_property(self, data=data) :
        value = [getattr(camera, f'_{data}', None) for camera in self.cameras]
        bool_ = [v is None for v in value]
        return None if any(bool_) else value
    setattr(Config, f'_{data}', _camera_property)
    @property
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    setattr(Config, f'channel_{data}', channel_property)

# Adding Camera properties
for data in Camera.properties :
    @property
    def camera_property(self, data=data) :
        return [getattr(camera, data) for camera in self.cameras]
    @camera_property.setter
    def camera_property(self, value, data=data) :
        try :
            if len(value) != self.nfiles : raise ValueError('Value set does not have same number of elements as cameras')
        except TypeError :
            value = [value for _ in range(self.nfiles)]            
        for camera, v in zip(self.cameras, value) :
            setattr(camera, data, v)
    setattr(Config, data, camera_property)
    @property
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    setattr(Config, f'channel_{data}', channel_property)



# Adding Channel metadata
for data, group in Channel.metadata :
    @metadatum(group, name=data)
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    @channel_property.setter
    def channel_property(self, value, data=data) :
        try :
            if len(value) != self.total_nchannels : raise ValueError('Value set does not have same number of elements as channels')
        except TypeError :
            value = [value for _ in range(self.total_nchannels)]     
        for channel, v in zip(self.channels, value) :
            setattr(channel, data, v)
    setattr(Config, data, channel_property)
    @property
    def _channel_property(self, data=data) :
        value = [getattr(channel, f'_{data}', None) for channel in self.channels]
        bool_ = [v is None for v in value]
        return None if any(bool_) else value
    setattr(Config, f'_{data}', _channel_property)

# Adding Channel properties
for data in Channel.properties :
    @property
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    @channel_property.setter
    def channel_property(self, value, data=data) :
        try :
            if len(value) != self.total_nchannels : raise ValueError('Value set does not have same number of elements as channels')
        except TypeError :
            value = [value for _ in range(self.total_nchannels)]     
        for channel, v in zip(self.channels, value) :
            setattr(channel, data, v)
    setattr(Config, data, channel_property)





# Convert to json function
def json_convert(value) :
    if isinstance(value, bool) or isinstance(value, np.bool_) : value = bool(value)
    if isinstance(value, int) or isinstance(value, np.integer) : value = int(value)
    if isinstance(value, float) or isinstance(value, np.floating) : value = float(value)
    if isinstance(value, tuple) or isinstance(value, list) : value = [json_convert(v) for v in value]
    if isinstance(value, np.ndarray) :
        value = value.ravel()
        if np.issubdtype(value.dtype, np.bool_) :
            value = value.astype(bool)
        elif np.issubdtype(value.dtype, np.integer) :
            value = value.astype(int)
        elif np.issubdtype(value.dtype, np.floating) :
            value = value.astype(float)
        value = list(value)
    return value



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)