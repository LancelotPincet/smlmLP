#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import Hardware
from corelp import prop



# %% Function
class Camera(Hardware) :
    tosave = ['pixel_cam', 'npixels', 'magnitude', 'nchannels', 'read_out', 'exposure', 'bits', 'ADU', 'QE']

    # Camera pixels
    @prop(iterable=2, dtype=float) 
    def pixel_cam(self) : # (y, x) µm
        return 6.5
    @prop(iterable=2, dtype=int)
    def npixels(self) :# (y, x)
        return 2304



    # Sample pixel size
    @prop(iterable=2, dtype=float)
    def magnitude(self) : # (y, x)
        if self.config is None : raise ValueError('Cannot access default xmagnitude without full configuration object')
        return self.pixel_cam[0] / self.config.objective.magnitude / 100. * 1e3, self.pixel_cam[1] / self.config.objective.magnitude / 100. * 1e3
    @property
    def pixel(self) : # (y, x) nm
        if self.config is None : raise ValueError('Cannot access sample plane pixel size without full configuration object')
        return self.pixel_cam[0] / self.config.objective.magnitude / self.magnitude[0] * 1e3, self.pixel_cam[1] / self.config.objective.magnitude / self.magnitude[1] * 1e3
    @pixel.setter
    def pixel(self, value):
        try :
            if len(value) != 2 : raise ValueError(f'Pixel values cannot have {len(value)} values')
        except TypeError :
            value = [value, value]
        self.magnitude = self.pixel_cam[0] / self.config.objective.magnitude / value[0] * 1e3, self.pixel_cam[1] / self.config.objective.magnitude / value[1] * 1e3, 



    # Channels
    @prop(dtype=int)
    def nchannels(self) :
        return 1
    @property
    def FOV_max(self) :
        ny, nx = self.npixels
        match self.nchannels :
            case 1 : # Full frame
                pass
            case 2 : # Divide along x direction (fast sCMOS axis, we want each channel to have the same time stamps per line)
                nx = nx - nx%2 # Make it even
                nx = nx / 2
            case 3 : # Divide along x direction, same reason
                nx = nx - nx%3 # Make it 3 multiple
                nx = nx / 3
            case 4 : # 4 quadrants
                nx = nx - nx%2 # Make it even
                ny = ny - ny%2 # Make it even
                nx, ny = nx / 2, ny / 2
            case _ : raise ValueError('Dividing a camera image into more than 4 channels is not supported.')
        return int(ny) * self.pixel[0] * 1e-3, int(nx) * self.pixel[1] * 1e-3
    @property
    def FOV(self) :
        if self.config is None : return self.FOV_max
        return self.config.FOV
    @property
    def shape(self) : # shape per channel (y, x)
        return (int(self.FOV[0] * 1e3 / self.pixel[0]), int(self.FOV[1] * 1e3 / self.pixel[1]))
    @property
    def bbox(self) :
        ny, nx = self.npixels # total number of pixels in frames
        sy, sx = self.shape # shape of each channel
        match self.nchannels:
            case 1: # Center a single crop
                x0 = (nx - sx) // 2
                y0 = (ny - sy) // 2
                return [(x0, y0, x0 + sx, y0 + sy)]
            case 2: # Two horizontal channels (split x into 2)
                total_w = 2 * sx
                x_start = (nx - total_w) // 2
                y0 = (ny - sy) // 2
                return [(x_start + i * sx, y0, x_start + (i + 1) * sx, y0 + sy) for i in range(2)]
            case 3: # Three horizontal channels (split x into 3)
                total_w = 3 * sx
                x_start = (nx - total_w) // 2
                y0 = (ny - sy) // 2
                return [ (x_start + i * sx, y0, x_start + (i + 1) * sx, y0 + sy) for i in range(3)]
            case 4: # 2x2 grid (split x and y)
                total_w = 2 * sx
                total_h = 2 * sy
                x_start = (nx - total_w) // 2
                y_start = (ny - total_h) // 2
                boxes = []
                for j in range(2):      # y direction
                    for i in range(2):  # x direction
                        x0 = x_start + i * sx
                        y0 = y_start + j * sy
                        boxes.append((x0, y0, x0 + sx, y0 + sy))
                return boxes
        case _: raise ValueError("Dividing a camera image into more than 4 channels is not supported.")



    # Exposure time
    @prop(dtype=float)
    def read_out(self) : #µs/line
        return 19.
    @property
    def min_exposure(self) :
        return self.read_out * 1e-3 * self.npixels[1] # x direction
    @prop(dtype=float)
    def exposure(self) : #µs/line
        return self.min_exposure
    @exposure.setter
    def exposure(self, value) :
        if value < self.min_exposure : raise ValueError('Exposure time set is smaller than minimum defined with read-out')
        self._exposure = float(value)
    @property
    def frequency(self) : # Hz
        return 1000 / self.exposure
    @frequency.setter
    def frequency(self, value) :
        self.exposure = 1000 / value



    # Data
    @prop(dtype=int)
    def bits(self) :
        return 16
    @property
    def frame_bytes(self) : # gigabytes/frame
        return self.npixels[0] * self.npixels[1] * self.bits / 8 / 1024**3
    @property
    def data_flux(self) : # megabytes / s
        return self.frame_bytes * self.frequency * 1024**3
    


    # Photons counting
    @prop(dtype=float)
    def ADU(self) :
        return 0.25
    @prop()
    def QE(self) :
        self.load_spectra('QE', 0.8)
    


    # Models
    models = {
        'ORCA-Fusion-BT' : dict(constructor='Hamamatsu', pixel_cam=6.5, npixels=2304, read_out=19., bits=16, ADU=0.25),
        'ORCA-Fusion' : dict(constructor='Hamamatsu', pixel_cam=6.5, npixels=2304, read_out=19., bits=16, ADU=0.25),
        'ORCA-Flash4.0' : dict(constructor='Hamamatsu', pixel_cam=6.5, npixels=2304, read_out=19., bits=16, ADU=0.25),
        'Kinetix' : dict(constructor='Teledyne Photometrics', pixel_cam=6.5, npixels=3200, bits=16, ADU=0.22),
        'iXon897Life' : dict(constructor='Andor', pixel_cam=16., npixels=512, bits=16),
        'Panda4.2' : dict(constructor='PCO', pixel_cam=6.5., npixels=2048, bits=16),
    }