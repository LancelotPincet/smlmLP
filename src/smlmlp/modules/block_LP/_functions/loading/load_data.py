#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



from smlmlp import block
from contextlib import ExitStack
from arrlp import gc
import tifffile as tiff
from stacklp import shapetif
import numpy as np
from concurrent.futures import ThreadPoolExecutor



@block()
def load_data(
    *tif_paths,
    chunk=None,
    pad=0,
    cameras_bboxeses=None,
    memmap=True,
    flip=None,
    cuda=False,
    parallel=False,
    iterator=range,
):
    """
    Load SMLM raw TIFF data in chunks and yield per-channel views.

    This generator loads one or more TIFF stacks chunk by chunk, optionally
    with temporal padding on both sides of each chunk. Each loaded file can be
    split into several channels using bounding boxes, and channel views can
    optionally be flipped along the y and/or x axes.

    Parameters
    ----------
    *tif_paths : str
        Paths to the TIFF files to load.
    chunk : int or None, optional
        Number of frames to load per iteration. If ``None``, the full
        acquisition is loaded at once.
    pad : int, optional
        Number of frames of temporal padding to include before and after each
        chunk.
    cameras_bboxeses : sequence or None, optional
        Per-file list of channel bounding boxes. For each file, each bounding
        box must be given as ``(x0, y0, x1, y1)`` and will be applied as
        ``[:, y0:y1, x0:x1]``.
    memmap : bool, optional
        Whether to try loading TIFF files through ``tifffile.memmap``.
    flip : sequence or None, optional
        Optional per-channel flip configuration. Each entry must be a pair
        ``(flip_y, flip_x)`` of booleans.
    cuda : bool, optional
        Unused in this function. It is kept for block API consistency.
    parallel : bool, optional
        Unused in this function. It is kept for block API consistency.
    iterator : callable, optional
        Iterator factory used to loop over chunk indices. This can be used,
        for example, to wrap the iteration with a progress bar.

    Yields
    ------
    tuple
        A tuple ``(channels, info)`` where:

        - ``channels`` is the list of per-channel views for the current chunk,
        - ``info`` is a dictionary containing reusable intermediate results.

        The dictionary contains the following keys:

        ``'chunk0'``
            First frame index of the current chunk, using 0-based indexing.
        ``'chunk1'``
            Last frame index of the current chunk, using 0-based indexing.
        ``'pad00'``
            First frame index of the left padding region.
        ``'pad01'``
            Last frame index of the left padding region.
        ``'pad10'``
            First frame index of the right padding region.
        ``'pad11'``
            Last frame index of the right padding region.

    Raises
    ------
    SyntaxError
        If no TIFF file path is provided.
    ValueError
        If the number of bounding-box lists does not match the number of
        files.
    ValueError
        If an input TIFF does not have exactly three dimensions.
    ValueError
        If the input TIFF files do not all have the same number of frames.

    Notes
    -----
    When memory mapping is available for a file, chunk extraction is performed
    directly on the mapped array. Otherwise, an internal temporary array is
    filled using explicit reads and temporal padding transfers.

    Examples
    --------
    Iterate over one file without chunking:

    >>> for channels, info in load_data("movie.tif", memmap=False):
    ...     print(len(channels), info["chunk0"], info["chunk1"])

    Iterate over two files with chunking and padding:

    >>> bbox = [
    ...     [(0, 0, 64, 64)],
    ...     [(0, 0, 32, 32), (32, 0, 64, 32)],
    ... ]
    >>> for channels, info in load_data(
    ...     "cam1.tif",
    ...     "cam2.tif",
    ...     chunk=500,
    ...     pad=20,
    ...     cameras_bboxeses=bbox,
    ... ):
    ...     print(info["chunk0"], info["chunk1"], len(channels))
    """
    cameras_bboxes = cameras_bboxeses

    with ExitStack() as stack:
        # Reset timings stored by the block decorator machinery.
        block.times = {}

        # Open all TIFF files inside the managed context.
        tifs = [stack.enter_context(tiff.TiffFile(file)) for file in tif_paths]
        shapes = [shapetif(tif) for tif in tifs]
        nfiles = len(tifs)

        if nfiles < 1:
            raise SyntaxError("Must define at least one tiff file to load")

        # Default to one full-frame channel per file when no bounding boxes are
        # provided.
        if cameras_bboxes is None:
            cameras_bboxes = [[(0, 0, shape[2], shape[1])] for shape in shapes]

        nchannels = [len(box) for box in cameras_bboxes]
        if len(nchannels) != nfiles:
            raise ValueError("Did not give the same amount of bbox as files")

        # Validate input stack shapes and ensure all files belong to the same
        # acquisition length.
        nframes = None
        for shape in shapes:
            if nframes is None:
                nframes = shape[0]

            if len(shape) != 3:
                raise ValueError(
                    f"Tiff files for SMLM data should have 3 dimensions (time, y, x), not {shape}"
                )

            if shape[0] != nframes:
                raise ValueError(
                    "All tiff files do not have the same number of frames which is not possible for a single SMLM acquisition"
                )

        # Normalize chunk and padding parameters to the acquisition length.
        if chunk is None:
            chunk = nframes
        if nframes < chunk:
            chunk = nframes
        if nframes <= pad:
            pad = nframes - 1

        nloops = int(np.ceil(nframes / chunk))

        # Try to memory-map each file when requested.
        if memmap:
            mmaps = []
            for tif_path in tif_paths:
                try:
                    mmap = tiff.memmap(tif_path)
                except ValueError:
                    mmap = None
                mmaps.append(mmap)
        else:
            mmaps = [None for _ in range(nfiles)]

        # Allocate temporary load buffers only for files that are not memory
        # mapped.
        frame_shapes = [shape[1:3] for shape in shapes]
        dtypes = [tif.pages.get(0).dtype for tif in tifs]
        loads = [
            np.empty((chunk + 2 * pad, *shape), dtype=dtype) if mmap is None else None
            for shape, dtype, mmap in zip(frame_shapes, dtypes, mmaps)
        ]

        # Iterate over chunk indices.
        for loop in iterator(nloops):
            gc()

            # Temporal layout:
            #
            #             <-- data flux <--
            # | 00pad01 |     0chunk1     | 10pad11 | BEFORE
            # |                loaded               |
            # |unk1     | 10pad11 |    new|array    | AFTER

            # Compute the chunk and padding boundaries in 0-based indexing.
            chunk0 = loop * chunk
            chunk1 = (loop + 1) * chunk - 1
            pad00 = chunk0 - pad
            pad01 = chunk0 - 1
            pad10 = chunk1 + 1
            pad11 = chunk1 + pad

            info = {
                "chunk0": chunk0,
                "chunk1": chunk1,
                "pad00": pad00,
                "pad01": pad01,
                "pad10": pad10,
                "pad11": pad11,
            }

            # Build the views on the memory-mapped arrays for the current chunk
            # and its padding region.
            mmaps_chunks = [
                mmap[max(0, pad00):min(pad11 + 1, nframes)] if mmap is not None else None
                for mmap in mmaps
            ]

            # Fill the temporary buffers for non-memory-mapped files.
            with ThreadPoolExecutor(max_workers=nfiles) as pool:
                futures = [
                    pool.submit(
                        _load_one_tif,
                        tif,
                        load,
                        loop,
                        chunk,
                        pad,
                        nframes,
                        chunk1,
                        pad10,
                        pad11,
                    )
                    for tif, load, mmap in zip(tifs, loads, mmaps)
                    if mmap is None
                ]

                for future in futures:
                    future.result()

            # Trim the temporary buffers when the right padding extends beyond
            # the end of the acquisition.
            if pad11 + 1 > nframes:
                loads = [
                    load[:nframes - pad00] if load is not None else None
                    for load in loads
                ]

            # Split each loaded array into channel views according to the
            # provided bounding boxes.
            channels = []
            count = 0

            for load, mmap, box in zip(loads, mmaps_chunks, cameras_bboxes):
                for bb in box:
                    x0, y0, x1, y1 = bb
                    channel = (
                        load[:, y0:y1, x0:x1]
                        if mmap is None else
                        mmap[:, y0:y1, x0:x1]
                    )

                    if flip is not None:
                        if flip[count][0]:
                            channel = channel[:, ::-1, :]
                        if flip[count][1]:
                            channel = channel[:, :, ::-1]

                    channels.append(channel)
                    count += 1

            yield channels, info



def _load_one_tif(tif, load, loop, chunk, pad, nframes, chunk1, pad10, pad11):
    """Load one TIFF chunk with temporal padding into a preallocated buffer."""
    # Define the three buffer regions: left padding, main chunk, right padding.
    array_pad0 = load[:pad]
    array_chunk = load[-pad - chunk:len(load) - pad]
    array_pad1 = load[len(load) - pad:]

    # Special case for the first iteration: initialize the right-padding buffer
    # with the first frames so it can be shifted into place.
    if pad > 0 and loop == 0:
        tif.asarray(key=slice(0, pad, 1), out=array_pad1)

    # Reuse previously loaded data by shifting the chunk and padding buffers.
    if pad > 0:
        np.copyto(array_pad0, array_chunk[-pad:], casting="no")
        np.copyto(array_chunk[:pad, :, :], array_pad1, casting="no")

    # Load the new chunk frames.
    pos0 = min(chunk1 + pad - chunk + 1, nframes)
    pos1 = min(chunk1 + 1, nframes)
    delta = pos1 - pos0

    if pos0 < nframes:
        tif.asarray(key=slice(pos0, pos1, 1), out=array_chunk[pad:pad + delta, :, :])

    # Load the new right-padding frames.
    if pad > 0:
        pos0 = min(pad10, nframes)
        pos1 = min(pad11 + 1, nframes)
        delta = pos1 - pos0

        if pos0 < nframes:
            tif.asarray(key=slice(pos0, pos1, 1), out=array_pad1[:delta, :, :])
