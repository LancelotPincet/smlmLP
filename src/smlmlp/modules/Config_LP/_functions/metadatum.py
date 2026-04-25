#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

def metadatum(group, dtype=None, name=None, iterable=None):
    """
    Build a metadata property for the Config class.

    Parameters
    ----------
    group : str
        Metadata group name.
    dtype : type or None, default=None
        Optional value conversion callable.
    name : str or None, default=None
        Optional metadata name overriding the decorated function name.
    iterable : int or None, default=None
        Required iterable length. Scalars are broadcast when provided.

    Returns
    -------
    callable
        Decorator returning a property object.
    """

    def decorator(func) :
        """Create the metadata property for one decorated function."""
        datum = func.__name__ if name is None else name
        if group not in metadatum.groups : metadatum.groups[group] = []
        metadatum.groups[group].append(datum)

        def getter(self):
            """Return getter."""
            _attribut = getattr(self, f'_{datum}', None)
            if _attribut is not None : return _attribut
            _attribute = func(self)
            return _attribute

        def setter(self, value):
            """Return setter."""
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

        def deleter(self):
            """Return deleter."""
            setattr(self, f'_{datum}', None)

        return property(getter, setter, deleter)
    return decorator


metadatum.groups = {}
