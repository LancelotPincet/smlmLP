#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet

import numpy as np
from arrlp import get_xp


def split_channel_origins(crops, X0, Y0, ch, *, cuda=False):
    """Split detection-aligned crop origins into one array per crop channel."""
    positions = channel_positions(crops, ch)
    X0_channels = split_detection_values(X0, crops, positions, cuda=cuda, name="X0")
    Y0_channels = split_detection_values(Y0, crops, positions, cuda=cuda, name="Y0")
    return X0_channels, Y0_channels, positions


def channel_positions(crops, ch):
    """Return one-based channel positions matching each crop stack."""
    n_channels = len(crops)
    crop_lengths = [len(crop) for crop in crops]
    total = sum(crop_lengths)
    if ch is None:
        if n_channels > 1: raise ValueError("ch is required when crops has several channels")
        return [np.arange(total)]

    ch_np = asnumpy(ch)
    if ch_np.ndim != 1: raise ValueError("ch must be a one-dimensional vector")
    if len(ch_np) != total: raise ValueError("ch must match the total number of crops")
    if total and (ch_np.min() < 1 or ch_np.max() > n_channels):
        raise ValueError("Channel indices must be one-based and within crops.")

    positions = []
    for channel_index, crop_length in enumerate(crop_lengths, start=1):
        pos = np.flatnonzero(ch_np == channel_index)
        if len(pos) != crop_length: raise ValueError("ch channel counts must match crop stack lengths")
        positions.append(pos)
    return positions


def split_detection_values(values, crops, positions, *, cuda=False, name="value"):
    """Split a one-dimensional detection-aligned value vector by channel."""
    xp = get_xp(cuda)
    values = xp.asarray(values)
    if values.ndim != 1: raise ValueError(f"{name} must be a one-dimensional vector")
    if len(values) != sum(len(crop) for crop in crops):
        raise ValueError(f"{name} must match the total number of crops")
    return [values[xp.asarray(pos)] for pos in positions]


def stack_channel_values(values, positions):
    """Stack per-channel outputs back into the detection order defined by positions."""
    if len(values) == 0: return np.empty(0, dtype=np.float32)
    values = [asnumpy(value) for value in values]
    total = sum(len(pos) for pos in positions)
    dtype = np.result_type(*[np.asarray(value).dtype for value in values], np.float32)
    output = np.empty(total, dtype=dtype)
    for value, pos in zip(values, positions):
        output[pos] = value
    return output


def asnumpy(array):
    """Return a NumPy view or copy for CPU-side indexing."""
    if hasattr(array, "get"): return array.get()
    return np.asarray(array)
