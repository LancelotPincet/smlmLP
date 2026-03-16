#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



def metadatum(group, dtype=None, name=None, iterable=None):
    '''
    Decorator used in Config class when defining a metadatum property
    will store the datum in the metadata dictionnary
    '''

    def decorator(func) :
        datum = func.__name__ if name is None else name
        if group not in metadatum.groups : metadatum.groups[group] = []
        metadatum.groups[group].append(datum) # Groups of metadata

        #Getter
        def getter(self):
            _attribut = getattr(self, f'_{datum}', None)
            if _attribut is not None : return _attribut
            _attribute = func(self)
            return _attribute

        #Setter
        def setter(self, value):
            if iterable is not None :
                try :
                    if len(value) != iterable :
                        raise ValueError(f'Cannot have lenght {len(value)}')
                except TypeError :
                    value = [value for _ in range(iterable)]
                if dtype is not None :
                    value = [dtype(v) for v in value]
            else :
                if dtype is not None :
                    value = dtype(value)
            setattr(self, f'_{datum}', value)

        #Deleter
        def deleter(self):
            setattr(self, f'_{datum}', None)

        #Property
        return property(getter, setter, deleter)
    return decorator

#Adding attributes
metadatum.groups = {}
