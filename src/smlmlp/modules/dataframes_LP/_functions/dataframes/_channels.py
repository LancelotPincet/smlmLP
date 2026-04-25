#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import DataFrame, column
import numpy as np



# %% Function
class channels(DataFrame) :
    '''
    Channels dataframe
    '''

    @column(headers=['channel'], dtype=np.uint8, save=False, agg='min', index="detections")
    def ch(self) :
        if self.locs.config.nchannels == 1 :
            return np.ones(self.locs.ndetections, dtype=np.uint8)
        else :
            from smlmlp import lost_channels
            return lost_channels(locs=self.locs)[0]
    
    

    # --- Camera ---

    @column(headers=['x pixel [nm]'], dtype=np.float32, save=False, agg='mean')
    def x_pixel(self) :
        return np.asarray([self.locs.config.channels_pixels_nm[i-1][1] for i in self.ch], dtype=np.float32)

    @column(headers=['y pixel [nm]'], dtype=np.float32, save=False, agg='mean')
    def y_pixel(self) :
        return np.asarray([self.locs.config.channels_pixels_nm[i-1][0] for i in self.ch], dtype=np.float32)

    @column(headers=['x shape [pix]'], dtype=np.uint8, save=False, agg='mean')
    def x_shape(self) :
        return np.asarray([self.locs.config.channels_shape[i-1][1] for i in self.ch], dtype=np.uint8)

    @column(headers=['y shape [pix]'], dtype=np.uint8, save=False, agg='mean')
    def y_shape(self) :
        return np.asarray([self.locs.config.channels_shape[i-1][0] for i in self.ch], dtype=np.uint8)

    @column(headers=['bits'], dtype=np.uint8, save=False, agg='mean')
    def bits(self) :
        return np.asarray([self.locs.config.channels_bits[i-1] for i in self.ch], dtype=np.uint8)

    @column(headers=['gain'], dtype=np.float32, save=False, agg='mean')
    def gain(self) :
        return np.asarray([self.locs.config.channels_gains[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['read noise'], dtype=np.float32, save=False, agg='mean')
    def read_noise(self) :
        return np.asarray([self.locs.config.channels_read_noises[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['qe'], dtype=np.float32, save=False, agg='mean')
    def qe(self) :
        return np.asarray([self.locs.config.channels_QE[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['camera index'], dtype=np.float32, save=False, agg='mean')
    def cam(self) :
        return np.asarray([self.locs.config.channels_camera_indices[i-1] for i in self.ch], dtype=np.uint8)

    @column(headers=['x flip'], dtype=np.bool_, save=False, agg='max')
    def x_flip(self) :
        return np.asarray([self.locs.config.channels_flips[i-1][1] for i in self.ch], dtype=np.bool_)

    @column(headers=['y flip'], dtype=np.bool_, save=False, agg='max')
    def y_flip(self) :
        return np.asarray([self.locs.config.channels_flips[i-1][0] for i in self.ch], dtype=np.bool_)



    # --- PSF ---

    @column(headers=['psf x sigma [nm]'], dtype=np.float32, save=False, agg='mean')
    def psf_x_sigma(self) :
        return np.asarray([self.locs.config.channels_psf_xsigmas_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['psf y sigma [nm]'], dtype=np.float32, save=False, agg='mean')
    def psf_y_sigma(self) :
        return np.asarray([self.locs.config.channels_psf_ysigmas_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['psf theta [deg]'], dtype=np.float32, save=False, agg='mean')
    def psf_theta(self) :
        return np.asarray([self.locs.config.channels_psf_thetas_deg[i-1] for i in self.ch], dtype=np.float32)



    # --- Registration ---

    @column(headers=['registration x shift [nm]'], dtype=np.float32, save=False, agg='mean')
    def x_shift(self) :
        return np.asarray([self.locs.config.channels_x_shift_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration y shift [nm]'], dtype=np.float32, save=False, agg='mean')
    def y_shift(self) :
        return np.asarray([self.locs.config.channels_y_shift_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration rotation [deg]'], dtype=np.float32, save=False, agg='mean')
    def rotation(self) :
        return np.asarray([self.locs.config.channels_rotation_deg[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration x shear'], dtype=np.float32, save=False, agg='mean')
    def x_shear(self) :
        return np.asarray([self.locs.config.channels_x_shear[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration y shear'], dtype=np.float32, save=False, agg='mean')
    def y_shear(self) :
        return np.asarray([self.locs.config.channels_y_shear[i-1] for i in self.ch], dtype=np.float32)



    # --- Crops ---

    @column(headers=['x cropshape [pix]'], dtype=np.float32, save=False, agg='mean')
    def x_cropshape(self) :
        return np.asarray([self.locs.config.channels_crops_pix[i-1][1] for i in self.ch], dtype=np.float32)

    @column(headers=['y cropshape [pix]'], dtype=np.float32, save=False, agg='mean')
    def y_cropshape(self) :
        return np.asarray([self.locs.config.channels_crops_pix[i-1][1] for i in self.ch], dtype=np.float32)
    
    @column(headers=['x cropsize [nm]'], dtype=np.float32, save=False, agg='mean')
    def x_cropsize(self) :
        return self.x_cropshape * self.x_pixel

    @column(headers=['y cropsize [nm]'], dtype=np.float32, save=False, agg='mean')
    def y_cropsize(self) :
        return self.y_cropshape * self.y_pixel

