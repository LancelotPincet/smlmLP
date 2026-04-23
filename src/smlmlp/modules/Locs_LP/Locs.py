#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-02-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : Locs

"""
This class define objects corresponding to localizations sets for one experiment.
"""



# %% Libraries
from corelp import folder, selfkwargs, prop
from smlmlp import open_df, save_df, LocsReceiver, DetsDataFrame, Config
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import nullcontext
import importlib



# %% Class
class Locs(LocsReceiver) :
    '''
    This class define objects corresponding to localizations sets for one experiment.

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
    >>> from smlmlp import Locs
    ...
    >>> instance = Locs(TODO)
    '''


    # Creates Locs objects
    def __new__(cls, source=None, /, **kwargs) :

        #If Locs object is given as input returns it directly
        if isinstance(source, cls) :
            return source

        #Creates a new Locs object
        return object.__new__(cls)



    # Initialize with any object
    df_dict = None
    def __init__(self, source=None, config=None, **kwargs) :
        self.df_dict = dict(detections=DetsDataFrame(self))
        selfkwargs(self, kwargs)

        #Changes attributes from kwargs values
        if isinstance(source, Locs) :
            return #break if already a Locs object

        #Opening data
        if source is not None :
            self.open(source)

        # Config
        if config is not None :
            self.config_source = config
        if isinstance(config, Config) :
            self.config = config
        else :
            timeit = self.printer.timeit('opening config file') if self.printer is not None else nullcontext()
            with timeit :
                self.config = Config(config=self.config_source, **self.config_kwargs)



    # Config
    @prop()
    def config_source(self) :
        return None
    @config_source.setter
    def config_source(self, value) :
        if getattr(self, '_config_source', None) is not None :
            raise ValueError('Config source is defined twice')
        self._config_source = value
    @property
    def config_kwargs(self) :
        return dict(ncameras=1) if self.config_source is None else dict()



    # Detections properties
    @property
    def detections(self) :
        return self.df_dict['detections']
    @property
    def ndetections(self) :
        return len(self.detections)



    # Analysis
    @prop(cache=True)
    def time(self) :
        return {}
    printer = None # rootlp printing



    # Opening
    def open(self, source) :
        if source is None :
            return
        elif isinstance(source, pd.DataFrame) :
            open_df(self, source, self.printer)
        elif isinstance(source, dict) :
            selfkwargs(self, source)
        elif isinstance(source, str) or isinstance(source,Path('').__class__) :
            path = Path(source)
            if path.is_dir() : # folder : open csv and json
                stem = path.stem
                detection_file = path / f'{stem}_[detections].csv'
                if not detection_file.exists() :
                    raise ValueError('Cannot open foler without "detections" file')
                self.open(detection_file)
                for file in path.glob(f'{stem}_[*].csv') :
                    if file == detection_file : continue
                    self.open(file)
                config_file = path / f'{stem}_[metadata].json'
                if config_file.exists() :
                    self.open(config_file)
            else :
                if path.suffix == '.csv' :
                    self.open(pd.read_csv(path))
                elif path.suffix == '.json' :
                    self.config_source = path
                else :
                    raise SyntaxError(f'Locs cannot open file {path}')
        elif isinstance(source, list) or isinstance(source, tuple) :
            for value in source :
                self.open(value)
        else :
            raise SyntaxError('source type was not recognized for Locs')



    # Saving
    def save(self, path, file=None) :
        path = Path(path) if file is None else Path(path) / file
        saving_folder = folder(path.with_suffix(''), warning=False)
        stem = saving_folder.stem
        mainpath = saving_folder / f'{stem}.csv'
        
        # Saving df
        for df_name, df in self.df_dict.items() :
            if df is None or len(df.columns) == 0 : continue
            path = mainpath.with_stem(f'{stem}_[{df_name}]')
            save_df(df, path, df.head2save, self.printer)

        # Saving metadata
        mainpath = saving_folder / f'{stem}_[metadata].json'
        self.config.save(mainpath)



    # Filter
    def filter(self, *filter_names, mask=None, df_name="detections") :
        base_df = self.df_dict[df_name]
        filters = [getattr(base_df, name, None) for name in filter_names]
        if mask is not None : filters += [mask]
        mask = np.logical_and.reduce(filters)
        base_df.keep = mask
        df_list = [df.loc[getattr(df, 'keep')] for df in self.df_dict.values()]
        for df in df_list :
            df.drop(columns=['filter'], inplace=True)
        return Locs(df_list, config=self.config)



    # Split
    def split(self, nlocs=2) :
        locs_list = []
        for i in range(nlocs) :
            mask = self.blinks.fr % nlocs == i
            newloc = self.filter(mask=mask, df_name="blinks")
            locs_list.append(newloc)
        return tuple(locs_list)



    # Combine
    def combine(self, *locs_list, col_name="ch") :
        locs_list = [self] + list(locs_list)
        detections = pd.concat([locs.detections for locs in locs_list], ignore_index=True)
        newlocs = Locs(detections, config=self.config)
        newcol = np.hstacks([np.full(locs.ndetections, fill_value=i+1, dtype=np.uint8) for i, locs in enumerate(locs_list)])
        setattr(newlocs.detections, col_name, newcol)
        return newlocs



    # Crop
    def crop(self, xmin, ymin, xmax, ymax) :
        x, y = self.detections.x, self.detections.y
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        return self.filter(mask=mask)



# Image
for img in ["wf", "irradiance"] :
    @property
    def img_prop(self) :
        image = getattr(self.config, f'{img}_image', None)
        if image is None :
            module = importlib.import_module("smlmlp")
            function = getattr(module, f"image_{img}")
            image = function(locs=self.locs)[0]
            setattr(self.config, f'{img}_image', image)
        return image
    @img_prop.setter
    def img_prop(self, value) :
        setattr(self.config, f'{img}_image', value)
    setattr(Locs, f'{img}_image', img_prop)




# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)