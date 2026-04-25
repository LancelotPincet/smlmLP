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

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

from corelp import folder, prop, selfkwargs
from smlmlp import Config, DetsDataFrame, LocsReceiver, columns, open_df, save_df



class Locs(LocsReceiver) :
    """
    Store localization dataframes and metadata for one experiment.

    Parameters
    ----------
    source : Locs, pandas.DataFrame, path-like, list[pandas.DataFrame], or None, default=None
        Localization source to open.
    config : Config, path-like, dict, or None, default=None
        Configuration source passed to Config.
    **kwargs
        Additional attributes assigned on the instance.

    Attributes
    ----------
    df_dict : dict
        Mapping of dataframe names to dataframe instances.
    config : Config
        Configuration attached to this localization set.

    Examples
    --------
    >>> from smlmlp import Locs
    >>> instance = Locs()
    """


    # Creates Locs objects
    def __new__(cls, source=None, config=None, **kwargs) :

        """Create or reuse an instance."""
        if isinstance(source, cls) :
            return source

        return object.__new__(cls)



    # Initialize with any object
    df_dict = None
    def __init__(self, source=None, config=None, **kwargs) :
        """Initialize the object."""
        self.df_dict = dict(detections=DetsDataFrame(self))
        selfkwargs(self, kwargs)

        if isinstance(source, Locs) :
            return

        if source is not None :
            self.open(source)

        # Config
        if config is not None :
            self.config_source = config
        timeit = self.printer.timeit('opening config file') if self.printer is not None else nullcontext()
        with timeit :
            self.config = Config(config=self.config_source, **self.config_kwargs)



    # Config
    @prop()
    def config_source(self) :
        """Return config source."""
        return None
    @config_source.setter
    def config_source(self, value) :
        """Set config source."""
        if getattr(self, '_config_source', None) is not None :
            raise ValueError('Config source is defined twice')
        self._config_source = value
    @property
    def config_kwargs(self) :
        """Return config kwargs."""
        return dict(ncameras=1) if self.config_source is None else dict()



    # Detections properties
    @property
    def detections(self) :
        """Return detections."""
        return self.df_dict['detections']
    @property
    def ndetections(self) :
        """Return ndetections."""
        return len(self.detections)



    # Analysis
    @prop(cache=True)
    def time(self) :
        """Return time."""
        return {}
    @property
    def times(self) :
        """Alias timing storage used by analysis decorators."""
        return self.time
    printer = None # rootlp printing



    # Opening
    def open(self, source) :
        """Open localization data from a supported source."""
        if source is None :
            return
        elif isinstance(source, pd.DataFrame) :
            open_df(self, source, self.printer)
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
            source = list(source)
            dets_idx = None
            for idx, df in enumerate(source) :
                if not isinstance(df, pd.DataFrame) : raise ValueError('Can only open lists/tuples of DataFrames')
                index = df.index.name
                if index is not None :
                    index = index.replace('"', '')
                    index = index.replace("'", "")
                else :
                    index = 'detection'
                col_index = columns.headers[index]
                df_name = col_index.df_name
                if df_name == "detections" :
                    if dets_idx is not None :
                        raise ValueError('Multiple detections dataframe to open which is not possible')
                    dets_idx = idx
            if dets_idx is None :
                raise ValueError('No detections dataframe to open which is not possible')
            dets = source.pop(dets_idx)
            source = [dets] + source
            for value in source :
                self.open(value)
        else :
            raise SyntaxError('source type was not recognized for Locs')



    # Saving
    def save(self, path, file=None) :
        """Save localization dataframes and metadata to a folder."""
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
        """Return a new Locs object containing rows matching all filters."""
        base_df = self.df_dict[df_name]
        filters = [getattr(base_df, name, None) for name in filter_names]
        if mask is not None : filters += [mask]
        mask = np.logical_and.reduce(filters)
        base_df.keep = mask
        df_list = [df.loc[getattr(df, 'keep')] for df in self.df_dict.values()]
        for df in df_list :
            df.drop(columns=['filter'], inplace=True)
        return Locs(df_list, config=self.config, printer=self.printer)



    # Split
    def split(self, nlocs=2) :
        """Split localizations by frame modulo ``nlocs``."""
        locs_list = []
        for i in range(nlocs) :
            mask = self.blinks.fr % nlocs == i
            newloc = self.filter(mask=mask, df_name="blinks")
            locs_list.append(newloc)
        return tuple(locs_list)



    # Combine
    def combine(self, *locs_list, col_name="ch") :
        """Combine this Locs object with others and label their source."""
        locs_list = [self] + list(locs_list)
        detections = pd.concat([locs.detections for locs in locs_list], ignore_index=True)
        newlocs = Locs(detections, config=self.config, printer=self.printer)
        newcol = np.hstack([np.full(locs.ndetections, fill_value=i+1, dtype=np.uint8) for i, locs in enumerate(locs_list)])
        setattr(newlocs.detections, col_name, newcol)
        return newlocs



    # Crop
    def crop(self, xmin, ymin, xmax, ymax) :
        """Crop detections to an inclusive xy rectangle."""
        x, y = self.detections.x, self.detections.y
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        return self.filter(mask=mask)

if __name__ == "__main__":
    from corelp import test

    test(__file__)
