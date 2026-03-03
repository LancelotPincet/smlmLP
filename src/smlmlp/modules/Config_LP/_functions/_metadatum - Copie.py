#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



def metadatum(group, unit=None, type=None, ui=None, name=None):
    '''
    Decorator used in Config class when defining a metadatum property
    will store the datum in the metadata dictionnary
    '''

    def decorator(func) :
        datum = func.__name__ if name is None else name
        if group not in metadatum.groups : metadatum.groups[group] = {} 
        metadatum.groups[group][datum] = datum # Groups of metadata
        metadatum.ui[datum] = ui

        #Getter
        def getter(self):
            _attribut = getattr(self, f'_{datum}',None)
            if _attribut is not None : return _attribut
            _attribute = func(self)
            return _attribute

        #Setter
        def setter(self, value):
            if type is not None :
                value = type(value)
            if isinstance(ui, list) :
                if value not in ui : raise SyntaxError(f'{value} is not in {ui}')
            setattr(self, f'_{datum}', value)

        #Deleter
        def deleter(self):
            setattr(self, f'_{datum}', None)

        #Property
        return property(getter, setter, deleter)
    return decorator

#Adding attributes
metadatum.groups = {}
metadatum.units = {}
metadatum.ui = {}
