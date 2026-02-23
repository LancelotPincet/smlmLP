#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import columns, DataFrame, LocsDataFrame
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



    def __init__(self, *, dtype, headers, save=True, index=False, agg='mean'):
        ''' First called, bracket of decorator '''
        self.dtype = dtype
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

        #Updating columns dictionnaries
        if self.col in columns: raise SyntaxError(f"Column {self.col} is defined twice")
        columns[self.col] = self
        for header in self.headers :
            if header in columns.headers: raise SyntaxError(f"Header {header} is defined twice")
            columns.headers[header] = self.col

        return self



    def __set_name__(self, cls, name):
        ''' Called when assigned to a class '''
        self.cls = cls
        if self.index : self.cls.set_index(self.col, inplace=True)

        # Assign column list
        if not hasattr(self.cls, "columns_list") : self.cls.columns_list = []
        self.cls.columns_list.append(self)

        # Get column object from dataframe instance
        @property
        def _col(instance) :
            return self
        setattr(self.cls, f'_{self.col}', _col)

        # Set on LocsDataFrame
        if self.cls is LocsDataFrame :
            @property
            def merged_col(df) :
                if self.header not in df.columns :
                    locs = df.locs
                    df[self.header] = locs.groupby(df.index.name)[self.header].agg(self.agg).to_numpy()
                return df[self.header].to_numpy()
            setattr(DataFrame, self.col, merged_col)
        
        # Set on DataFrame
        else :
            @property
            def spread_col(locs) :
                if self.header not in df.columns :
                    df = getattr(locs, self.cls.__name__)
                    locs[self.header] = locs[df.index.name].map(df).to_numpy()
                return locs[self.header].to_numpy()
            setattr(LocsDataFrame, self.col, merged_col)
                


    def __get__(self, instance, cls):

        # Gets from dataframe
        if self.col in instance.columns :
            return instance[self.header].to_numpy()

        # Automatic calculation
        newcol = self.func(instance)
        if newcol is None :
            raise ValueError(f'{self.col} is not defined')

        # Substitute
        if isinstance(newcol, str) :
            return getattr(instance, newcol)
        
        # Calculated
        setattr(instance, self.col, newcol)
        return getattr(instance, self.col)



    def __set__(self, instance, value):
        
        #Removing column from dataframe (value=None)
        if value is None :
            instance.drop(columns=[self.header], inplace=True)

        #Setting column
        array = np.asarray(value, dtype=self.dtype)
        df[self.header] = array
