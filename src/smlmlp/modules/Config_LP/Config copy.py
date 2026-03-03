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
from smlmlp import metadatum
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
    def __init__(self, config=None, **kwargs) :

        if config is not None :

            # Open file
            if isinstance(config, str) or isinstance(config,Path('').__class__):
                config_file = Path(config).with_suffix('.json')
                if config_file.exists() :
                    with open(config_file, "r") as file:
                        config = json.load(file)
                else :
                    raise SyntaxError(f'config path was not recognized: {config}')
            selfkwargs(self, config)
        selfkwargs(self, kwargs)

    @property
    def metadata(self) :
        return {datum : getattr(self, datum, None) for group in metadatum.groups for datum in group}

    def save(self, path, file=None) :
        path = Path(path)
        if file is not None : path = path / file
        path = path.with_suffix('.json')
        with open(path, 'w') as json_file:
            json.dump(self.metadata, json_file, indent=4)

    def file_update(self, *tif_paths) :
        with ExitStack() as stack :
            tifs = [stack.enter_context(tiff.TiffFile(file)) for file in tif_paths]
            shapes = [shapetif(tif) for tif in tifs]

            # Check number of frames
            nframes = None
            for shape in shapes :
                if nframes is None : nframes = shape[0]
                if shape[0] != nframes :
                    raise ValueError('All tiff files do not have the same number of frames which is not possible for a single SMLM acquisition')
            self.nframes = nframes

            # Define shape
            # TODO




    # --- Below are the metadata, organized by group---



    # Cameras
    



    # Image size
    @property
    def shape(self) :
        return (self.yshape, self.xshape)
    @shape.setter
    def shape(self, value) :
        self.yshape, self.xshape = int(value), int(value)
    @metadatum("Image size", unit='pixel', ui=False)
    def xshape(self) :
        return 1024
    @metadatum("Image size", unit='pixel', ui=False)
    def yshape(self) :
        return 1024
    @metadatum("Image size", unit='µm', ui=False)
    def xFOV(self) : # µm
        return self.xshape * self.xpixel * 1e-3
    @metadatum("Image size", unit='µm', ui=False)
    def yFOV(self) : # µm
        return self.yshape * self.xpixel * 1e-3
    

    # 
    @metadatum(unit='ypix, xpix')
    def cropshape(self) :
        return (int(round(self.PSF_field/self.ypixel)), int(round(self.PSF_field/self.xpixel)))



    # PSF
    @metadatum('PSF', unit='nm')
    def wavelength(self) : # Fluorescence wavelength [nm]
        return 671.
    @metadatum('PSF')
    def NA(self) : # Objective Numerical Aperture
        return 1.5
    @metadatum('PSF', ui=['air', 'water', 'glycerol', 'silicon', 'oil'])
    def immersion_medium(self) : # Optical_index of immersion_medium
        return 'oil'
    @metadatum('PSF', ui=False)
    def optical_index(self) :
        dic = {'air': 1., 'water': 1.33, 'glycerol': 1.4, 'silicon': 1.4, 'oil': 1.515}
        return dic[self.immersion_medium]
    @metadatum('PSF')
    def PSF_fact(self) : # Experimental PSF growth factor
        return 2.
    @property
    def PSF_sigma(self) : #PSF gaussian sigma [nm]
        return 0.21 * self.wavelength / self.NA * self.PSF_fact
    @property
    def PSF_field(self) :
        return 1.22 * self.wavelength / self.NA * self.PSF_fact
    @property
    def DOF(self) : #Depth of Field [nm]
        delta_wave = self.immersion_medium * self.wavelength / self.NA**2
        delta_geom = self.immersion_medium * self.pixel / self.NA
        return delta_wave + delta_geom



    #Photon count
    @metadatum("Photon count")
    def ADU(self) : # Analog Digital Unit of camera
        return 0.25
    @metadatum("Photon count")
    def QE(self) : # Quantum efficiency of camera at wavelength
        return 0.72
    @metadatum("Photon count")
    def camera_gain(self) : # Camera gain
        return 1.
    @metadatum("Photon count")
    def camera_compensation(self) : # Camera readout noise compensation
        return 0.



    #Density
    @metadatum("Localization density", unit='loc/µm²')
    def density(self) : # density of localization [loc/µm²]
        return 0.1
    @metadatum("Localization density", unit='nm', ui=False)
    def density_radius(self) :
        return np.sqrt(1 / self.density / np.pi) * 1000 # [nm]
    @density_radius.setter
    def density_radius(self, value) :
        self.density = 1 / (value / 1000)**2 / np.pi
    @property
    def PSF_influence(self) : # mean radius of influence of a PSF [nm]
        return np.exp(((np.log(3 * self.PSF_sigma) + np.log(self.density_radius)) / 2)) # logarithmic average



    #Photophysics
    @metadatum("Photophysics", unit='ms')
    def exposure(self) : # camera exposure time [ms]
        return 50. 
    @metadatum("Photophysics", ui=['STORM', 'PAINT', 'PALM'])
    def imaging_modality(self) :
        return 'STORM'
    @metadatum("Photophysics", unit='ms')
    def tau_ON(self) : # Mean ON time of blinking events
        dic = dict(STORM=15., PAINT=200., PALM=15.)
        if self.imaging_modality in dic : return dic[self.imaging_modality]
        return self.exposure



    # Defines number of loops and hom many frames to charge
    @property
    def frame_bytes(self) : # gigabytes/frame
        return np.sum([(size[0] * size[1] * dtype.itemsize / 1024 ** 3) for size, dtype in zip(self.sizes,self.dtypes)])
    chunk_fraction = 1/25 # fraction of available ram to define how many frames to load at once
    @property
    def chunk_in_memory(self):
        return int(self.memory*self.chunk_fraction // self.frame_bytes) +1
    pad = 0
    @metadatum(unit='frames')
    def load(self) : # number of frames loaded in stack
        if getattr(self,'_load',None) is not None :
            return max(1,min(self.nframes+2*self.pad,self._load))
        elif getattr(self,'_chunk',None) is not None :
            return self._chunk + 2*self.pad
        return self.chunk + 2*self.pad
    @load.setter
    def load(self,value) :
        self._load = value
        self._chunk = None
    @metadatum(unit='frames')
    def chunk(self) : # number of frames to process at each loop
        if getattr(self,'_chunk',None) is not None :
            return max(1,min(self.nframes,self._chunk))
        elif getattr(self,'_load',None) is not None :
            return self._load - 2*self.pad
        return min(self.nframes, self.chunk_in_memory)
    @chunk.setter
    def chunk(self,value) :
        self._chunk = value
        self._load = None
    @property
    def nloops(self) :
        return int(np.ceil(self.nframes/self.chunk))



    # Defines how to divide into channels
    _nchannels = None #Number of channels
    @metadatum(unit='channels')
    def nchannels(self) :
        if self._channel2load is not None :
            return len(self._channel2load) #If channels defined takes number directly
        return self.nfiles #If not defined, 1 channel / load

    _channel2load = None
    @metadatum(unit='n°load')
    def channel2load(self) : #For each channel, load index in which to get it
        if self.nfiles == 1 : #If only 1 file
            return list(iterable(0, self.nchannels))
        return list(np.arange(self.nfiles))
    @property
    def load2channels(self) : #For each load, gives channel indices
        channel2load = self.channel2load
        n = np.max(channel2load)+1
        return [[ch for ch, ld in enumerate(channel2load) if load==ld] for load in range(n)]
    @property
    def channel2channel(self) :
        return [[ch] for ch in range(self.nchannels)]

    @metadatum(unit='(x0,y0,x1,y1)')
    def channel_bbox(self) : #(x0, y0, x1, y1)
        if self.nchannels == self.nfiles : #Takes full frame
            return [(0, 0, self.sizes[load][0], self.sizes[load][1]) for load in self.channel2load]
        if self.nchannels == 1 : #Takes full frame
            return [(0, 0, self.sizes[0][0], self.sizes[0][1])]
        if self.nchannels == 2 and self.nfiles == 1: # Divides along x, sCMOS rapid direction
            return [
                    (0, 0, self.sizes[0][0]//2, self.sizes[0][1]),
                    (self.sizes[0][0] - self.sizes[0][0]//2, 0, self.sizes[0][0], self.sizes[0][1]),
                    ]
        if self.nchannels == 4 and self.nfiles == 1: # Divides along x and y, sCMOS rapid direction
            return [
                    (0, 0, self.sizes[0][0]//2, self.sizes[0][1]//2),
                    (self.sizes[0][0] - self.sizes[0][0]//2, 0, self.sizes[0][0], self.sizes[0][1]//2),
                    (0, self.sizes[0][1] - self.sizes[0][1]//2, self.sizes[0][0]//2, self.sizes[0][1]),
                    (self.sizes[0][0] - self.sizes[0][0]//2, self.sizes[0][1] - self.sizes[0][1]//2, self.sizes[0][0], self.sizes[0][1]),
                    ]
        if self.nchannels == 4 and self.nfiles == 2: # Divides along x on the two files, sCMOS rapid direction
            return [
                    (0, 0, self.sizes[0][0]//2, self.sizes[0][1]),
                    (self.sizes[0][0] - self.sizes[0][0]//2, 0, self.sizes[0][0], self.sizes[0][1]),
                    (0, 0, self.sizes[1][0]//2, self.sizes[1][1]),
                    (self.sizes[1][0] - self.sizes[1][0]//2, 0, self.sizes[1][0], self.sizes[1][1]),
                    ]
        if self.nchannels == 5 and self.nfiles == 2 :
            return [
                    (0, 0, self.sizes[0][0]//2, self.sizes[0][1]//2),
                    (self.sizes[0][0] - self.sizes[0][0]//2, 0, self.sizses[0][0], self.sizes[0][1]//2),
                    (0, self.sizes[0][1] - self.sizes[0][1]//2, self.sizes[0][0]//2, self.sizes[0][1]),
                    (self.sizes[0][0] - self.sizes[0][0]//2, self.sizes[0][1] - self.sizes[0][1]//2, self.sizes[0][0], self.sizes[0][1]),
                    (0, 0, self.sizes[1][0], self.sizes[1][1])
                    ]
        return [(0, 0, self.sizes[0][0], self.sizes[0][1]) for _ in self.channel2load] #If nothing, just gives first full frame n times



    #Channel info
    @property
    def channel_shapes(self) :
        return [(y1-y0, x1-x0) for x0, y0, x1, y1 in self.channel_bbox]
    @property
    def channel_sizes(self) :
        return [(shape[1], shape[0]) for shape in self.channel_shapes]
    @property
    def channel_xpixels(self) :
        return [self.xpixels[load] for load in self.channel2load]
    @property
    def channel_ypixels(self) :
        return [self.ypixels[load] for load in self.channel2load]
    @property
    def channel_pixels(self) :
        return [np.sqrt(xpix*ypix) for xpix, ypix in zip(self.channel_xpixels, self.channel_ypixels)]
    @metadatum(unit='xpix, ypix')
    def channel_cropsizes(self) :
        return [(np.uint8(round(self.PSF_field/xpix)),np.uint8(round(self.PSF_field/ypix))) for xpix, ypix in zip(self.channel_xpixels, self.channel_ypixels)]
    @property
    def channel_cropshapes(self) :
        return [(size[1], size[0]) for size in self.channel_cropsizes]



    #Translations
    @metadatum(unit='nm')
    def channel_shifts(self) : #[(shiftx, shifty) for each channel] [nm]
        return [(0., 0.) for _ in range(self.nchannels)]

    #Shearing
    @metadatum(unit='nm/nm')
    def channel_shears(self) : #[(shearx, sheary) for each channel] []
        return [(0., 0.) for _ in range(self.nchannels)]

    #Rotations
    @metadatum(unit='°')
    def channel_angles(self) : #[theta for each channel] [°]
        return [0. for _ in range(self.nchannels)]

    #Zoom
    @metadatum(unit='')
    def channel_scales(self) : #[(scalex, scaley) for each channel] []
        return [(1., 1.) for _ in range(self.nchannels)]

    #Transform matrix
    @initproperty
    def transform_matrices(self) :
        matrices = [transform_matrix(shape, shiftx, shifty, shearx, sheary, angle, scalex, scaley)
                for shape, (shiftx, shifty), (shearx, sheary), angle, (scalex, scaley) in zip(self.channel_shapes, self.channel_shifts, self.channel_shears, self.channel_angles, self.channel_scales)]
        return [matrix if (matrix != np.eye(3)).any() else None for matrix in matrices]
    @metadatum(init=True)
    def do_transform(self) :
        for matrix in self.transform_matrices :
            if matrix is not None :
                return True
        return False



    @metadatum(init=True, unit='nm')
    def local_window(self) :
        return int(round(self.PSF_influence*2))



    #temporal_median
    tau_fact = 25 #Number of tau_ON to consider temporal window
    @metadatum(unit='frames')
    def temporal_window(self) :
        return int(np.ceil(self.tau_ON*self.tau_fact/self.exposure))



    #Detection parameters
    @defaultproperty
    def noise_factsmin(self) : #minimum noise factor for thresholding detected image
        return 1
    @defaultproperty
    def noise_factsstep(self) : #noise factor step for thresholding detected image
        return 0.5
    @defaultproperty
    def noise_factsmax(self) : #maximum noise factor for thresholding detected image
        return 2.001
    @metadatum(init=True)
    def noise_facts(self) : #can be set as a list if needed for different configs
        return np.arange(self.noise_factsmin, self.noise_factsmax, self.noise_factsstep)
    @metadatum(init=True)
    def area_min(self) :
        return 5
    @metadatum(init=True)
    def area_max(self) :
        radius = 2*int(np.ceil(2*self.PSF_sigma))+1
        return int(np.sum((ROI(field=radius, pixel=self.pixel, center=True).R <= radius)))


    #filters to apply
    @metadatum()
    def detfilters2apply(self) : #List of filter names to apply
        return ['close_borders']

    #filters to apply
    @metadatum()
    def locfilters2apply(self) : #List of filter names to apply
        return ['close_borders']



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)