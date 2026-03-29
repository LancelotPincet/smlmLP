#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import columns, DataFrame, MainDataFrame, LocsDataFrame
import numpy as np



# %% Class
class column() :
    '''
    This object is a decorator to apply to localization columns.
    
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
    >>> from smlmlp import column
    ...
    >>> instance = column(TODO)
    '''



    def __init__(self, *, headers, save=True, index=False, agg='mean'):
        ''' First called, bracket of decorator '''
        self.headers = headers #List of possible headers, first is default one
        self.header = headers[0]
        self.save = save #True to save this column in files
        self.index = index # True if index is the index of the class
        self.agg = agg # Aggregation method



    def __call__(self, func) :
        ''' Called after initialization '''

        #Get column info
        self.func = func #Default behavior function
        self.col = func.__name__ #column shortcut name
        self.dtype = func.__annotations__['self'] # dtype of column

        #Updating columns dictionnaries
        if self.col in columns: raise SyntaxError(f"Column {self.col} is defined twice")
        columns[self.col] = self
        for header in self.headers :
            if header in columns.headers: raise SyntaxError(f"Header {header} is defined twice")
            columns.headers[header] = self

        return self



    def __set_name__(self, cls, name):
        ''' Called when assigned to a class '''
        self.cls = cls # dataframe object
        if self.index :
            self.df_name = f'{self.header}s'
            if self.df_name != self.cls.__name__ :
                raise SyntaxError(f'Index column {self.col} does not coincide with DataFrame name {self.cls.__name__}')
            self.cls.index_name = self.header

        # Assign column dict
        if not hasattr(self.cls, "columns_list") : self.cls.columns_dict = {}
        self.cls.columns_dict[self.col] = self

        # Assign saving list
        if self.index :
            for header in self.headers :
                MainDataFrame.head2save.append(header)
        else :
            if not hasattr(self.cls, "head2save") : self.cls.head2save = []
            for header in self.headers :
                self.cls.head2save.append(header)
        

        # Get column object from dataframe instance
        @property
        def _col(instance) :
            return self
        setattr(self.cls, f'_{self.col}', _col)

        # Belongs
        if self.index or issubclass(self.cls, MainDataFrame) :
            setattr(MainDataFrame, f'{self.col}_mine', True)
            setattr(DataFrame, f'{self.col}_mine', False)
        else :
            setattr(MainDataFrame, f'{self.col}_mine', False)
            setattr(DataFrame, f'{self.col}_mine', True)
            

        # Set on MainDataFrame
        if issubclass(self.cls, MainDataFrame) :
            @property
            def merged_col(df) :
                if self.header not in df.columns :
                    dets = df.locs.detections
                    df[self.header] = dets.groupby(df.index.name)[self.header].agg(self.agg).to_numpy()
                return df[self.header].to_numpy()
            if hasattr(DataFrame, self.col) : raise SyntaxError(f'{self.col} cannot be defined twice in DataFrame')
            setattr(DataFrame, self.col, merged_col)

        
        # Set on DataFrame
        else :
            if self.index :
                @property
                def index_col(dets) :
                    if self.header not in dets.columns :
                        return None
                    return dets[self.header].to_numpy()
                @index_col.setter
                def index_col(dets, value) :
                    if value is None :
                        dets.drop(columns=[self.header], inplace=True)
                        dets.df_dict.pop(self.df_name)
                    else :
                        array = np.asarray(value, dtype=self.dtype)
                        dets[self.header] = array
                if hasattr(MainDataFrame, self.col) : raise SyntaxError(f'{self.col} cannot be defined twice in MainDataFrame')
                setattr(MainDataFrame, self.col, index_col)
                @property
                def get_df(locs) :
                    if self.df_name not in locs.df_dict :
                        from smlmlp import dataframes
                        locs.df_dict[self.df_name] = dataframes[self.df_name](locs)
                    return locs.df_dict[self.df_name]
                if hasattr(LocsDataFrame, self.df_name) : raise SyntaxError(f'{self.df_name} cannot be defined twice in LocsDataFrame')
                setattr(LocsDataFrame, self.df_name, get_df)
                @property
                def len_df(locs) :
                    df = getattr(locs, self.df_name)
                    return len(df)
                if hasattr(LocsDataFrame, f'n{self.df_name}') : raise SyntaxError(f'n{self.df_name} cannot be defined twice in LocsDataFrame')
                setattr(LocsDataFrame, f'n{self.df_name}', len_df)
            else :
                @property
                def spread_col(dets) :
                    if self.header not in dets.columns :
                        df = getattr(dets.locs, self.cls.__name__)
                        dets[self.header] = dets[df.index.name].map(df[self.header]).to_numpy()
                    return dets[self.header].to_numpy()
                if hasattr(MainDataFrame, self.col) : raise SyntaxError(f'{self.col} cannot be defined twice in LocsDataFrame')
                setattr(MainDataFrame, self.col, spread_col)
                


    def __get__(self, instance, cls):
        # Good practice getter
        if instance is None:
            return self

        # Index
        if self.index :
            return instance.index.to_numpy()

        # Gets from dataframe
        if self.header in instance.columns :
            return instance[self.header].to_numpy()

        # Automatic calculation
        newcol = self.func(instance)
        if newcol is None :
            return None

        # Substitute
        if isinstance(newcol, str) :
            return getattr(instance, newcol)
        
        # Calculated
        setattr(instance, self.col, newcol)
        return getattr(instance, self.col)



    def __set__(self, instance, value):
        
        # Index
        if self.index :
            if value is None : raise SyntaxError('Setting Dataframe index to None is not possible')
            return instance.index.to_numpy()

        #Removing column from dataframe (value=None)
        if value is None :
            instance.drop(columns=[self.header], inplace=True)

        #Setting column
        array = np.asarray(value, dtype=self.dtype)
        instance[self.header] = array

