#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from corelp import prop



# %% Function
class Camera :
    '''
    Defines camera instance
    '''

    metadata = [ # (metadata, group)
        ("nchannels", "Cameras"),
        ("npixels", "Cameras"),
        ("bits", "Cameras"),
        ("pixel_nm", "Cameras"),
        ("constructor_gain", "Cameras"),
        ("experimental_gain", "Detections"),
        ("read_noise", "Detections"),
        ("QE", "Cameras"),
        ]
    properties = ['camera_index', 'gain']



    def __init__(self, config) :
        self.config = config



    # Channels
    @property
    def camera_index(self) :
        for i in range(self.config.ncameras) :
            if self.config.cameras[i] is self :
                return i

    @property
    def channels(self) :
        if not hasattr(self, '_channels') : self.nchannels = 1
        return self._channels
    @property
    def nchannels(self) :
        return len(self.channels)
    @nchannels.setter
    def nchannels(self, value) :
        from smlmlp import Channel
        self._channels = [Channel(self) for _ in range(int(value))]
    @property
    def _nchannels(self) :
        channels = getattr(self, '_channels', None)
        return None if channels is None else len(channels)



    # Pixels

    @prop(iterable=2, dtype=int)
    def npixels(self) : # (y, x)
        return 2304

    @prop(iterable=2, dtype=float)
    def pixel_nm(self) : # (y, x) [nm]
        return 100.



    # Data

    @prop(dtype=int)
    def bits(self) :
        return 16

    @property
    def frame_bytes(self) : # gigabytes/frame
        return self.npixels[0] * self.npixels[1] * self.bits / 8 / 1024**3
    


    # Photons counting
    @property
    def gain(self) :
        return self.experimental_gain if self.experimental_gain is not None else self.constructor_gain

    @prop(dtype=float)
    def constructor_gain(self) : # e-/ADU (Analog to Digital Unit)
        return 0.25

    @prop(dtype=float)
    def experimental_gain(self) : # e-/ADU (Analog to Digital Unit)
        return None

    @prop(dtype=float)
    def read_noise(self) : # ADU (Analog to Digital Unit)
        return 0.

    @prop(dtype=float)
    def QE(self) : # Quantum Efficiency
        0.8
    


    @property
    def FOV_max_um(self) : # (y, y) [µm]
        ny, nx = self.npixels
        match self.nchannels :
            case 1 : # Full frame
                pass
            case 2 : # Divide along x direction (fast sCMOS axis, we want each channel to have the same time stamps per line)
                nx = nx - nx%2 # Make it even
                nx = nx // 2
            case 3 : # Divide along x direction, same reason
                nx = nx - nx%3 # Make it 3 multiple
                nx = nx // 3
            case 4 : # 4 quadrants
                nx = nx - nx%2 # Make it even
                ny = ny - ny%2 # Make it even
                nx, ny = nx // 2, ny // 2
            case _ : raise ValueError('Dividing a camera image into more than 4 channels is not supported.')
        return int(ny) * self.pixel_nm[0] * 1e-3, int(nx) * self.pixel_nm[1] * 1e-3

    def FOV2bbox(self, FOV) : # (x0, y0, x1, y1)
        ny, nx = self.npixels # total number of pixels in frames
        sy, sx = int(FOV[0] * 1e3 / self.pixel_nm[0]), int(FOV[1] * 1e3 / self.pixel_nm[1]) # shape of each channel
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

