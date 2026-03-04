#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block
from contextlib import ExitStack
import gc
import tifffile as tiff
from stacklp import shapetif
import numpy as np



# %% Function
@block()
def load_data(*tif_paths, chunk=None, pad=0, bbox=None, iterator=range) :
    '''
    This generator loads SMLM raw data in chunks from various tif files.
    '''

    with ExitStack() as stack :
        block.times = {} # Reset timings of blocks
        tifs = [stack.enter_context(tiff.TiffFile(file)) for file in tif_paths]
        shapes = [shapetif(tif) for tif in tifs]
        nfiles = len(tifs)
        if nfiles < 1 : raise SyntaxError('Must define at least one tiff file to load')
        if bbox is None : bbox = [[(0, 0, shape[2], shape[1]),] for shape in shapes]
        nchannels = [len(box) for box in bbox]
        if len(nchannels) != nfiles : raise ValueError('Did not give the same amount of bbox as files')

        # Check number of frames
        nframes = None
        for shape in shapes :
            if nframes is None : nframes = shape[0]
            if len(shape) != 3 :
                raise ValueError(f'Tiff files for SMLM data should have 3 dimensions (time, y, x), not {shape}')
            if shape[0] != nframes :
                raise ValueError('All tiff files do not have the same number of frames which is not possible for a single SMLM acquisition')

        # Correct parameters
        if chunk is None : chunk = nframes
        if nframes < chunk :
            chunk = nframes
        if nframes <= pad :
            pad = nframes - 1
        nloops = int(np.ceil(nframes / chunk))

        # Allocating memory
        shapes = [shape[1:3] for shape in shapes] # Remove number of frames
        dtypes = [tif.pages.get(0).dtype for tif in tifs] # Get dtype of each file
        loads = [np.empty(shape=(chunk + 2 * pad, *shape), dtype=dtype) for shape, dtype in zip(shapes, dtypes)]

        # loops
        for loop in iterator(nloops) :
            gc.collect()  # Force garbage collection

            #             <-- data flux <--          
            # | 00pad01 |     0chunk1     | 10pad11 | BEFORE
            # |                loaded               |
            # |unk1     | 10pad11 |    new|array    | AFTER

            # Calculate borders
            chunk0, chunk1 = loop * chunk, (loop+1) * chunk -1
            pad00, pad01 = chunk0 - pad, chunk0 - 1
            pad10, pad11 = chunk1 + 1, chunk1 + pad
            borders = dict(chunk0=chunk0, chunk1=chunk1, pad00=pad00, pad01=pad01, pad10=pad10, pad11=pad11)



            # --- Load ---



            for tif, load in zip(tifs, loads) :
                
                # Defining views
                array_pad0 = load[:pad]
                array_chunk = load[-pad-chunk:len(load)-pad]
                array_pad1 = load[len(load)-pad:]

                # First loop scenario
                if pad > 0 and loop == 0 :
                    tif.asarray(key=slice(0, pad, 1), out=array_pad1)

                # Transfering already loaded data
                if pad > 0 :
                    np.copyto(array_pad0, array_chunk[-pad:])
                    np.copyto(array_chunk[:pad,:,:], array_pad1)

                # Loading chunk data
                pos0 = min(chunk1 + pad - chunk + 1, nframes)
                pos1 = min(chunk1 + 1, nframes)
                delta = pos1 - pos0
                if pos0 < nframes :
                    tif.asarray(key=slice(pos0, pos1, 1), out=array_chunk[pad:pad+delta,:,:])

                # Loading pad1 data
                if pad > 0 :
                    pos0 = min(pad10, nframes)
                    pos1 = min(pad11 + 1, nframes)
                    delta = pos1 - pos0
                    if pos0 < nframes :
                        tif.asarray(key=slice(pos0, pos1, 1), out=array_pad1[:delta,:,:])

            # Slicing if end of acquisition
            if pad11 + 1 > nframes :
                loads = [load[:nframes - pad11 - 1] for load in loads]



            # --- Dividing loads arrays into channels ---



            # Make channel views
            channels = []
            for load, box in zip(loads, bbox) :
                for bb in box :
                    # bb = (x0, y0, x1, y1) slicing --> [y0:y1, x0:x1]
                    x0, y0, x1, y1 = bb
                    channel = load[:, y0:y1, x0:x1]
                    channels.append(channel)



            # Return the generator value
            yield channels, borders