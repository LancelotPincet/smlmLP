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
    def __init__(self, *tif_paths, config=None, locs=None, **kwargs) :

        # Attach locs object
        self.locs = locs

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
                            data[key] = np.load(file, allow_pickle=True)
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
            np.save(config_folder / f"{key}.npy", np.array(value, dtype=object))




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
    def frame_bytes(self) : # gigabytes/frame
        return sum([camera.frame_bytes for camera in self.cameras])



    # Channels

    @property
    def nchannels(self) :
        return sum(self.cameras_nchannels)

    @property
    def channels(self) :
        return [channel for camera in self.cameras for channel in camera.channels]

    @property
    def channels_shapes(self) :
        return [(bb[3]-bb[1], bb[2]-bb[0]) for bboxes in self.cameras_bbox for bb in bboxes]


    # Loads

    @metadatum('Loads', dtype=int)
    def cuda(self) :
        return 0

    @metadatum('Loads', dtype=int)
    def parallel(self) :
        return 0

    @metadatum('Loads', dtype=int)
    def nframes(self) :
        return 60000

    @metadatum('Loads', dtype=int)
    def loaded(self) :
        return 256

    @property
    def pad(self) :
        median_window_fr = int(self.median_window_ms / self.exposure_ms)
        temporal_kernel_length = int(self.temporal_kernel_shape[0])
        pad_max = int((self.loaded - 1) // 2)
        return min(max(median_window_fr, temporal_kernel_length), pad_max)

    @property
    def chunk(self) :
        return self.loaded - self.pad * 2



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



    # Localization

    @metadatum('Localizations')
    def optimizer(self) : # string
        return "lm"
    @optimizer.setter
    def optimizer(self, value) :
        assert type(value) is str
        assert value.lower() in ['lm']
        self._optimizer = value.lower()

    @metadatum('Localizations')
    def estimator(self) : # string
        return "mle"
    @estimator.setter
    def estimator(self, value) :
        assert type(value) is str
        assert value.lower() in ["mle", "lse"]
        self._estimator = value.lower()

    @metadatum('Localizations')
    def distribution(self) : # string
        return "poisson"
    @distribution.setter
    def distribution(self, value) :
        assert type(value) is str
        assert value.lower() in ["poisson", "normal"]
        self._distribution = value.lower()

    @metadatum('Localizations')
    def model(self) : # string
        return "isogaussian"
    @model.setter
    def model(self, value) :
        assert type(value) is str
        assert value.lower() in ["isogaussian", "gaussian", "spline"]
        self._model = value.lower()



    # Effective values

    @metadatum('Effective')
    def intensity_channels(self) :
        return [idx for idx in range(1, self.nchannels + 1)]
    @intensity_channels.setter
    def intensity_channels(self, value) :
        if np.ndim(value) == 0 : 
            self._intensity_channels = [value]
        else :
            self._intensity_channels = value

    @metadatum('Effective')
    def x_channels(self) :
        return [idx for idx in range(1, self.nchannels + 1)]
    @x_channels.setter
    def x_channels(self, value) :
        if np.ndim(value) == 0 : 
            self._x_channels = [value]
        else :
            self._x_channels = value

    @metadatum('Effective')
    def y_channels(self) :
        return [idx for idx in range(1, self.nchannels + 1)]
    @y_channels.setter
    def y_channels(self, value) :
        if np.ndim(value) == 0 : 
            self._y_channels = [value]
        else :
            self._y_channels = value

    @metadatum('Effective')
    def z_channels(self) :
        return [idx for idx in range(1, self.nchannels + 1)]
    @z_channels.setter
    def z_channels(self, value) :
        if np.ndim(value) == 0 : 
            self._z_channels = [value]
        else :
            self._z_channels = value



    # Methods

    @metadatum('Methods')
    def x_method(self) : # string
        return "fit"
    @x_method.setter
    def x_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["det", "fit", "modloc", "timeloc"]
        self._x_method = value.lower()

    @metadatum('Methods')
    def y_method(self) : # string
        return "fit"
    @y_method.setter
    def y_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["det", "fit", "modloc", "timeloc"]
        self._y_method = value.lower()

    @metadatum('Methods')
    def z_method(self) : # string
        return "fit"
    @z_method.setter
    def z_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["fit", "astig", "biplane", "donald", "modloc", "timeloc", "miet", "qtirf"]
        self._z_method = value.lower()

    @metadatum('Methods')
    def azimuth_method(self) : # string
        return "polar3d"
    @azimuth_method.setter
    def azimuth_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["polar2d", "polar3d"]
        self._azimuth_method = value.lower()

    @metadatum('Methods')
    def tilt_method(self) : # string
        return "polar3d"
    @tilt_method.setter
    def tilt_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["polar3d"]
        self._tilt_method = value.lower()

    @metadatum('Methods')
    def phase_method(self) : # string
        return "demodulated"
    @phase_method.setter
    def phase_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["demodulated", "sequential"]
        self._phase_method = value.lower()

    @metadatum('Methods')
    def lifetime_method(self) : # string
        return "iflim"
    @lifetime_method.setter
    def lifetime_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["tcspc", "iflim", "dpflim"]
        self._lifetime_method = value.lower()

    @metadatum('Methods')
    def drift_method(self) : # string
        return "none"
    @drift_method.setter
    def drift_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["none", "crosscorr", "comet", "aim", "meanshift"]
        self._drift_method = value.lower()

    @metadatum('Methods')
    def demix_method(self) : # string
        return "flux"
    @demix_method.setter
    def demix_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["flux", "spectral", "lifetime"]
        self._demix_method = value.lower()

    @metadatum('Methods')
    def demix2d_method(self) : # string
        return "spectral"
    @demix_method.setter
    def demix_method(self, value) :
        assert type(value) is str
        assert value.lower() in ["spectral"]
        self._demix_method = value.lower()



    # Targets
    
    @metadatum('Targets')
    def dyes(self) : # string list
        return ['unknown']
    @dyes.setter
    def dyes(self, value) :
        for v in value :
            assert type(v) is str
        value = [v.lower() for v in value]
        self._dyes = value
    @property
    def ndyes(self) :
        return len(self.ndyes)



    # Image

    @metadatum('Data')
    def irradiance_image(self) : # 2D image
        if self.locs is not None :
            from smlmlp import image_pixel
            return image_pixel(self.os, None, locs=self.locs)[0]
        return None



    # Association

    @metadatum('Association')
    def channel_association_radius_nm(self) : # nm
        return 30.

    @metadatum('Association')
    def blink_association_radius_nm(self) : # nm
        if self.locs is not None :
            from smlmlp import associate_consecutive_frames_radius
            return associate_consecutive_frames_radius(locs=self.locs)[0]
        return 30.

    @metadatum('Association')
    def blink_z_association_radius_nm(self) : # nm
        return 100.

    @metadatum('Association')
    def track_association_radius_nm(self) : # nm
        return 500.



    # Ratio

    @metadatum('Ratio')
    def spectral_x_channels(self) : # nm
        return [idx for idx in range(1, self.nchannels)] if self.nchannels > 1 else [self.nchannels]

    @metadatum('Ratio')
    def spectral_y_channels(self) : # nm
        return [self.nchannels]

    @metadatum('Ratio')
    def biplane_x_channels(self) : # nm
        return [idx for idx in range(1, self.nchannels)] if self.nchannels > 1 else [self.nchannels]

    @metadatum('Ratio')
    def biplane_y_channels(self) : # nm
        return [self.nchannels]

    @metadatum('Ratio')
    def donald_x_channels(self) : # nm
        return [idx for idx in range(1, self.nchannels)] if self.nchannels > 1 else [self.nchannels]

    @metadatum('Ratio')
    def donald_y_channels(self) : # nm
        return [self.nchannels]

    @metadatum('Ratio')
    def iflim_x_channels(self) : # nm
        return [idx for idx in range(1, self.nchannels)] if self.nchannels > 1 else [self.nchannels]

    @metadatum('Ratio')
    def iflim_y_channels(self) : # nm
        return [self.nchannels]



    # Modloc

    @metadatum('Modloc')
    def modloc_transverse_angle_deg(self) : # deg
        return 0.

    @metadatum('Modloc')
    def modloc_axial_angle_deg(self) : # nm
        return 0.

    @metadatum('Modloc')
    def modloc_channels_indices(self) : # nm
        return [1, 2, 3, 4]

    @metadatum('Modloc')
    def modloc_sequential_frames(self) : # nm
        return 1

    @metadatum('Modloc')
    def modloc_dephases_rad(self) : # nm
        return [0, np.pi/2, np.pi, 3*np.pi/2]



    # Calibrations

    @metadatum('Data')
    def x_timeloc_calibration(self) : # nm / Hz
        return None

    @metadatum('Data')
    def y_timeloc_calibration(self) : # nm / Hz
        return None

    @metadatum('Data')
    def z_timeloc_calibration(self) : # nm / Hz
        return None

    @metadatum('Data')
    def z_astig_calibration(self) : # nm / ratio
        return None

    @metadatum('Data')
    def z_biplane_calibration(self) : # nm / ratio
        return None

    @metadatum('Data')
    def z_donald_calibration(self) : # nm / ratio
        return None

    @metadatum('Data')
    def z_miet_calibration(self) : # nm / ns
        return None

    @metadatum('Data')
    def z_qtirf_calibration(self) : # nm / ns
        return None

    @metadatum('Data')
    def lifetime_iflim_calibration(self) : # ns / ratio
        return None



    # Time
    
    @metadatum('Time')
    def frames_per_sequence(self) :
        return 1000

    @metadatum('Time')
    def zstack_speed(self) : # nm / frame
        return 10.



    # Drifts
    
    @metadatum('Drift')
    def aim_(self) :
        return 10.

    @metadatum('Drift')
    def comet_(self) :
        return 10.

    @metadatum('Drift')
    def crosscorr_(self) :
        return 10.

    @metadatum('Drift')
    def meanshift_(self) :
        return 10.



    # Clusters

    @metadatum('Clusters')
    def dbscan_eps(self) : # nm
        return 50.

    @metadatum('Clusters')
    def dbscan_min_points(self) :
        return 10.

    

def get_datas(data) :
    """ adds 's' to data parameter caring for units"""
    suffix = ''
    for unit in ['_nm', '_um', '_mm', '_deg', '_rad', '_us', '_ms', '_s', '_pix', '_fr'] :
        if data.endswith(unit) :
            data, suffix = data[:-len(unit)], data[-len(unit):]
            break
    if data.endswith('us') : data = f'{data[:-2]}i'
    elif data.endswith('ex') : data = f'{data[:-2]}ices'
    elif data.endswith('ix') : data = f'{data[:-2]}ices'
    elif data.endswith('ox') : data = f'{data[:-2]}oxes'
    elif not data.endswith('s') and not data.endswith('x') and data[-1].upper() != data[-1] : data = f'{data}s'
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