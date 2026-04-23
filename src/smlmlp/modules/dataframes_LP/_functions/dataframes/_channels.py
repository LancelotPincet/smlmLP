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

    @column(headers=['channel'], save=False, agg='min', index="detections")
    def ch(self:np.uint8) :
        if self.locs.config.nchannels == 1 :
            return np.ones(self.locs.ndetections, dtype=np.uint8)
        else :
            from smlmlp import index_channels
            return index_channels(locs=self.locs)[0]
    
    

    # --- Camera ---

    @column(headers=['x pixel [nm]'], save=False, agg='mean')
    def x_pixel(self:np.float32) :
        return np.asarray([self.locs.config.channels_pixels_nm[i-1][1] for i in self.ch], dtype=np.float32)

    @column(headers=['y pixel [nm]'], save=False, agg='mean')
    def y_pixel(self:np.float32) :
        return np.asarray([self.locs.config.channels_pixels_nm[i-1][0] for i in self.ch], dtype=np.float32)

    @column(headers=['x shape [pix]'], save=False, agg='mean')
    def x_shape(self:np.uint8) :
        return np.asarray([self.locs.config.channels_shape[i-1][1] for i in self.ch], dtype=np.uint8)

    @column(headers=['y shape [pix]'], save=False, agg='mean')
    def y_shape(self:np.uint8) :
        return np.asarray([self.locs.config.channels_shape[i-1][0] for i in self.ch], dtype=np.uint8)

    @column(headers=['bits'], save=False, agg='mean')
    def bits(self:np.uint8) :
        return np.asarray([self.locs.config.channels_bits[i-1] for i in self.ch], dtype=np.uint8)

    @column(headers=['gain'], save=False, agg='mean')
    def gain(self:np.float32) :
        return np.asarray([self.locs.config.channels_gains[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['read noise'], save=False, agg='mean')
    def read_noise(self:np.float32) :
        return np.asarray([self.locs.config.channels_read_noises[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['qe'], save=False, agg='mean')
    def qe(self:np.float32) :
        return np.asarray([self.locs.config.channels_QE[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['camera index'], save=False, agg='mean')
    def cam(self:np.uint8) :
        return np.asarray([self.locs.config.channels_camera_indices[i-1] for i in self.ch], dtype=np.uint8)

    @column(headers=['x flip'], save=False, agg='max')
    def x_flip(self:np.bool_) :
        return np.asarray([self.locs.config.channels_flips[i-1][1] for i in self.ch], dtype=np.bool_)

    @column(headers=['y flip'], save=False, agg='max')
    def y_flip(self:np.bool_) :
        return np.asarray([self.locs.config.channels_flips[i-1][0] for i in self.ch], dtype=np.bool_)



    # --- PSF ---

    @column(headers=['psf x sigma [nm]'], save=False, agg='mean')
    def psf_x_sigma(self:np.float32) :
        return np.asarray([self.locs.config.channels_psf_xsigmas_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['psf y sigma [nm]'], save=False, agg='mean')
    def psf_y_sigma(self:np.float32) :
        return np.asarray([self.locs.config.channels_psf_ysigmas_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['psf theta [deg]'], save=False, agg='mean')
    def psf_theta(self:np.float32) :
        return np.asarray([self.locs.config.channels_psf_thetas_deg[i-1] for i in self.ch], dtype=np.float32)



    # --- Registration ---

    @column(headers=['registration x shift [nm]'], save=False, agg='mean')
    def x_shift(self:np.float32) :
        return np.asarray([self.locs.config.channels_x_shift_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration y shift [nm]'], save=False, agg='mean')
    def y_shift(self:np.float32) :
        return np.asarray([self.locs.config.channels_y_shift_nm[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration rotation [deg]'], save=False, agg='mean')
    def rotation(self:np.float32) :
        return np.asarray([self.locs.config.channels_rotation_deg[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration x shear'], save=False, agg='mean')
    def x_shear(self:np.float32) :
        return np.asarray([self.locs.config.channels_x_shear[i-1] for i in self.ch], dtype=np.float32)

    @column(headers=['registration y shear'], save=False, agg='mean')
    def y_shear(self:np.float32) :
        return np.asarray([self.locs.config.channels_y_shear[i-1] for i in self.ch], dtype=np.float32)



    # --- Crops ---

    @column(headers=['x cropshape [pix]'], save=False, agg='mean')
    def x_cropshape(self:np.float32) :
        return np.asarray([self.locs.config.channels_crops_pix[i-1][1] for i in self.ch], dtype=np.float32)

    @column(headers=['y cropshape [pix]'], save=False, agg='mean')
    def y_cropshape(self:np.float32) :
        return np.asarray([self.locs.config.channels_crops_pix[i-1][1] for i in self.ch], dtype=np.float32)
    
    @column(headers=['x cropsize [nm]'], save=False, agg='mean')
    def x_cropsize(self:np.float32) :
        return self.x_cropshape * self.x_pixel

    @column(headers=['y cropsize [nm]'], save=False, agg='mean')
    def y_cropsize(self:np.float32) :
        return self.y_cropshape * self.y_pixel
