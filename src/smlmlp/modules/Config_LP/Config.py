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
        self.ncameras = max(len(tif_paths), 1)

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
            self.cameras_npixels = [shape[1:3] for shape in shapes]



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
    def ncameras(self) :
        raise SyntaxError('Should not be called')
    @ncameras.setter
    def ncameras(self, value) :
        self.cameras = [Camera(self) for _ in range(int(value))]
    @property
    def _ncameras(self) :
        return len(self.cameras)

    @property
    def FOV_max_um(self) :
        ymin = min([camera.FOV_max_um[0] for camera in self.cameras])
        xmin = min([camera.FOV_max_um[1] for camera in self.cameras])
        return ymin, xmin

    @property
    def cameras_bbox(self) :
        FOV = self.FOV_max_um
        return [camera.FOV2bbox(FOV) for camera in self.cameras]
    @property
    def frame_bytes(self) : # gigabytes/frame
        return sum([camera.frame_bytes for camera in self.cameras])



    # Channels

    @property
    def nchannels(self) :
        return sum(self.cameras_nchannels)

    @property
    def channels(self) :
        return [channel for camera in self.cameras for channel in camera.channels]



    # Loads

    @metadatum('Loads', dtype=int)
    def nframes(self) :
        return 60000



    # Time

    @metadatum('Cameras', dtype=float)
    def exposure_ms(self) : # [ms]
        return 50.

    @metadatum('Blinks')
    def on_time_ms(self) : # ms
        return 50.



    # Background configurations

    @metadatum('Backgrounds', dtype=float)
    def median_window_ms(self) : # For temporal median [ms]
        return self.on_time_ms * 10
    
    @property
    def median_window_fr(self) :
        return int(round(self.median_window_ms / self.exposure_ms))
    @median_window_fr.setter
    def median_window_fr(self, value) :
        self.median_window_ms = value * self.exposure_ms

    @metadatum('Backgrounds', dtype=bool)
    def do_temporal_median(self) : # For temporal median
        return True

    @metadatum('Backgrounds', dtype=float)
    def mean_radius_nm(self) : # For spatial mean [nm]
        return max(self.channels_psf_sigmas_nm) * 8

    @metadatum('Backgrounds', dtype=bool)
    def do_spatial_mean(self) : # For spatial mean
        return True

    @metadatum('Backgrounds', dtype=float)
    def opening_radius_nm(self) : # For spatial opening [nm]
        return max(self.channels_psf_sigmas_nm) * 2

    @metadatum('Backgrounds', dtype=bool)
    def do_spatial_opening(self) : # For spatial opening
        return False



    # Signal configurations

    @metadatum('Signals', dtype=bool)
    def do_spatial_filter(self) : # For spatial filter
        return True
    @metadatum('Signals', dtype=bool)
    def do_temporal_filter(self) : # For temporal filter
        return False

    @metadatum('Signals')
    def spatial_subtract_factor(self) :
        return 2.5
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
        on_frames = self.on_time_ms / self.exposure_ms
        if self.temporal_subtract_factor > 1 : on_frames *= self.temporal_subtract_factor
        on_frames = int(np.ceil(on_frames))
        return (on_frames * 10 + 1),

    @property
    def on_time_kernel(self) :
        T, = coordinates(shape=self.temporal_kernel_shape, pixel=self.exposure_ms)
        from funclp import Exponential1
        exponential = Exponential1(tau=self.on_time_ms)
        k = exponential(np.abs(T))
        k /= k.sum()
        return k.astype(np.float32)

    @property
    def temporal_subtract_kernel(self) :
        if self.temporal_subtract_factor <= 1 : return None
        exposure = self.exposure_ms / self.temporal_subtract_factor
        T, = coordinates(shape=self.temporal_kernel_shape, pixel=exposure)
        from funclp import Exponential1
        exponential = Exponential1(tau=self.on_time_ms)
        k = exponential(np.abs(T))
        k /= k.sum()
        return k.astype(np.float32)

    @prop(cache=True)
    def temporal_kernel(self) :
        if self.temporal_subtract_factor <= 1 :
            return self.on_time_kernel
        k = self.on_time_kernel - self.temporal_subtract_kernel
        return k - k.mean()



    # Detection

    @metadatum('Detections')
    def snr_thresh(self) :
        return 4.



    # Crops

    @metadatum('Crops')
    def crop_nm(self) : # nm
        h = max([cz[0] for cz in self.channels_default_crops_nm])
        w = max([cz[1] for cz in self.channels_default_crops_nm])
        return max(h, w)



def get_datas(data) :
    """ adds 's' to data paramter caring for units"""
    suffix = ''
    for unit in ['_nm', '_um', '_mm', '_deg', '_rad', '_us', '_ms', '_s', '_pix', '_fr'] :
        if data.endswith(unit) :
            data, suffix = data[:-len(unit)], data[-len(unit):]
            break
    if not data.endswith('s') and not data.endswith('x') and data[-1].upper() != data[-1] : data = f'{data}s'
    return data + suffix



# Adding Camera metadata
for data, group in Camera.metadata :
    datas = get_datas(data)
    @metadatum(group, name=f'cameras_{datas}')
    def camera_property(self, data=data) :
        return [getattr(camera, data) for camera in self.cameras]
    @camera_property.setter
    def camera_property(self, value, data=data) :
        try :
            if len(value) != self.ncameras : raise ValueError('Value set does not have same number of elements as cameras')
        except TypeError :
            value = [value for _ in range(self.ncameras)]            
        for camera, v in zip(self.cameras, value) :
            setattr(camera, data, v)
    setattr(Config, f'cameras_{datas}', camera_property)
    @property
    def _camera_property(self, data=data) :
        value = [getattr(camera, f'_{data}', None) for camera in self.cameras]
        bool_ = [v is None for v in value]
        return None if any(bool_) else value
    setattr(Config, f'_cameras_{datas}', _camera_property)
    @property
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    setattr(Config, f'channels_{datas}', channel_property)

# Adding Camera properties
for data in Camera.properties :
    datas = get_datas(data)
    @property
    def camera_property(self, data=data) :
        return [getattr(camera, data) for camera in self.cameras]
    @camera_property.setter
    def camera_property(self, value, data=data) :
        try :
            if len(value) != self.ncameras : raise ValueError('Value set does not have same number of elements as cameras')
        except TypeError :
            value = [value for _ in range(self.ncameras)]            
        for camera, v in zip(self.cameras, value) :
            setattr(camera, data, v)
    setattr(Config, f'cameras_{datas}', camera_property)
    @property
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    setattr(Config, f'channels_{datas}', channel_property)



# Adding Channel metadata
for data, group in Channel.metadata :
    datas = get_datas(data)
    @metadatum(group, name=f'channels_{datas}')
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    @channel_property.setter
    def channel_property(self, value, data=data) :
        try :
            if len(value) != self.nchannels : raise ValueError('Value set does not have same number of elements as channels')
        except TypeError :
            value = [value for _ in range(self.nchannels)]     
        for channel, v in zip(self.channels, value) :
            setattr(channel, data, v)
    setattr(Config, f'channels_{datas}', channel_property)
    @property
    def _channel_property(self, data=data) :
        value = [getattr(channel, f'_{data}', None) for channel in self.channels]
        bool_ = [v is None for v in value]
        return None if any(bool_) else value
    setattr(Config, f'_channels_{datas}', _channel_property)

# Adding Channel properties
for data in Channel.properties :
    datas = get_datas(data)
    @property
    def channel_property(self, data=data) :
        return [getattr(channel, data) for channel in self.channels]
    @channel_property.setter
    def channel_property(self, value, data=data) :
        try :
            if len(value) != self.nchannels : raise ValueError('Value set does not have same number of elements as channels')
        except TypeError :
            value = [value for _ in range(self.nchannels)]     
        for channel, v in zip(self.channels, value) :
            setattr(channel, data, v)
    setattr(Config, f'channels_{datas}', channel_property)





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