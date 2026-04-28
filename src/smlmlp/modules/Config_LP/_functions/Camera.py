#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

"""
Define camera-level acquisition and optics metadata.

The Camera class stores properties for individual cameras including
pixel size, gain, quantum efficiency, and channel bounding boxes.
"""

from corelp import prop


class Camera:
    """
    Define camera-level acquisition and optics metadata.

    Parameters
    ----------
    config : Config
        Parent configuration object.

    Attributes
    ----------
    camera_index : int
        Index of this camera within the configuration.
    channels : list
        List of Channel objects for this camera.
    gain : float
        Effective gain (e-/ADU).
    bboxes : list
        List of bounding boxes for each channel.
    """

    metadata = [
        ("nchannels", "Cameras"),
        ("npixels", "Cameras"),
        ("bits", "Cameras"),
        ("pixel_nm", "Cameras"),
        ("constructor_gain", "Cameras"),
        ("experimental_gain", "Detections"),
        ("read_noise", "Detections"),
        ("QE", "Cameras"),
    ]
    properties = ["camera_index", "gain", "bboxes"]

    def __init__(self, config):
        """Initialize the camera with its parent configuration."""
        self.config = config

    # Channels

    @property
    def camera_index(self):
        """Return the index of this camera within its configuration."""
        for i in range(self.config.ncameras):
            if self.config.cameras[i] is self:
                return i

    @property
    def channels(self):
        """Return the list of channels for this camera."""
        if not hasattr(self, "_channels"):
            self.nchannels = 1
        return self._channels

    @property
    def nchannels(self):
        """Return the number of channels for this camera."""
        return len(self.channels)

    @nchannels.setter
    def nchannels(self, value):
        """Set the number of channels, creating Channel objects."""
        from smlmlp import Channel

        self._channels = [Channel(self) for _ in range(int(value))]

    @property
    def _nchannels(self):
        """Return the stored number of channels or None if not set."""
        channels = getattr(self, "_channels", None)
        return None if channels is None else len(channels)

    # Pixels

    @prop(iterable=2, dtype=int)
    def npixels(self):
        """Return the number of pixels in (y, x) format."""
        return 2304

    @prop(iterable=2, dtype=float)
    def pixel_nm(self):
        """Return the pixel size in nm in (y, x) format."""
        return 100.0

    # Data

    @prop(dtype=int)
    def bits(self):
        """Return the bit depth of the camera."""
        return 16

    @property
    def frame_bytes(self):
        """Return the frame size in gigabytes."""
        return self.npixels[0] * self.npixels[1] * self.bits / 8 / 1024**3

    # Photon counting

    @property
    def gain(self):
        """Return effective gain, preferring experimental over constructor."""
        g = (
            self.experimental_gain
            if self.experimental_gain is not None and self.experimental_gain > 0
            else self.constructor_gain
        )
        return abs(g)

    @prop(dtype=float)
    def constructor_gain(self):
        """Return constructor-specified gain in e-/ADU."""
        return 0.25

    @prop(dtype=float)
    def experimental_gain(self):
        """Return experimentally measured gain in e-/ADU."""
        return None

    @prop(dtype=float)
    def read_noise(self):
        """Return read noise in ADU."""
        return 0.0

    @prop(dtype=float)
    def QE(self):
        """Return quantum efficiency."""
        return 0.8

    # Bounding box

    @property
    def bboxes(self):
        """Return bounding boxes for each channel derived from FOV."""
        FOV = self.config.FOV_max_um
        return self.FOV2bbox(FOV)

    @property
    def FOV_max_um(self):
        """Return maximum field of view in microns."""
        ny, nx = self.npixels
        match self.nchannels:
            case 1:
                pass
            case 2:
                nx = nx - nx % 2
                nx = nx // 2
            case 3:
                nx = nx - nx % 3
                nx = nx // 3
            case 4:
                nx = nx - nx % 2
                ny = ny - ny % 2
                nx, ny = nx // 2, ny // 2
            case _:
                raise ValueError(
                    "Dividing a camera image into more than 4 channels is not supported."
                )
        return int(ny) * self.pixel_nm[0] * 1e-3, int(nx) * self.pixel_nm[1] * 1e-3

    def FOV2bbox(self, FOV):
        """Convert field of view to per-channel bounding boxes.

        Parameters
        ----------
        FOV : tuple
            Field of view in (y, x) microns.

        Returns
        -------
        list
            List of (x0, y0, x1, y1) bounding boxes.
        """
        ny, nx = self.npixels
        sy = int(FOV[0] * 1e3 / self.pixel_nm[0])
        sx = int(FOV[1] * 1e3 / self.pixel_nm[1])
        match self.nchannels:
            case 1:
                x0 = (nx - sx) // 2
                y0 = (ny - sy) // 2
                return [(x0, y0, x0 + sx, y0 + sy)]
            case 2:
                total_w = 2 * sx
                x_start = (nx - total_w) // 2
                y0 = (ny - sy) // 2
                return [
                    (x_start + i * sx, y0, x_start + (i + 1) * sx, y0 + sy)
                    for i in range(2)
                ]
            case 3:
                total_w = 3 * sx
                x_start = (nx - total_w) // 2
                y0 = (ny - sy) // 2
                return [
                    (x_start + i * sx, y0, x_start + (i + 1) * sx, y0 + sy)
                    for i in range(3)
                ]
            case 4:
                total_w = 2 * sx
                total_h = 2 * sy
                x_start = (nx - total_w) // 2
                y_start = (ny - total_h) // 2
                boxes = []
                for j in range(2):
                    for i in range(2):
                        x0 = x_start + i * sx
                        y0 = y_start + j * sy
                        boxes.append((x0, y0, x0 + sx, y0 + sy))
                return boxes
            case _:
                raise ValueError(
                    "Dividing a camera image into more than 4 channels is not supported."
                )