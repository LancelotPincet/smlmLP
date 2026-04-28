#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
from smlmlp import block, computer, load_chunking
from contextlib import ExitStack
from arrlp import gc
import tifffile as tiff
from stacklp import shapetif
import numpy as np
from concurrent.futures import ThreadPoolExecutor



# %% Function
@block()
def load_data(*tif_paths, chunk=None, pad=0, cameras_bboxes=None,
              memmap=True, flip=None, owner="unknown", cuda=False, parallel=False, iterator=range):
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
        acquisition is requested at once. The effective value can be reduced
        when the resource server grants fewer RAM/VRAM tokens than requested.
    pad : int, optional
        Number of frames of temporal padding to include before and after each
        chunk.
    cameras_bboxes : sequence or None, optional
        Per-file list of channel bounding boxes. For each file, each bounding
        box must be given as ``(x0, y0, x1, y1)`` and will be applied as
        ``[:, y0:y1, x0:x1]``.
    memmap : bool, optional
        Whether to try loading TIFF files through ``tifffile.memmap``.
    flip : sequence or None, optional
        Optional per-channel flip configuration. Each entry must be a pair
        ``(flip_y, flip_x)`` of booleans.
    owner : str, optional
        Name inserted into the resource-token owner field when reservations are
        created.
    cuda : bool, optional
        Whether to reserve VRAM tokens in addition to RAM tokens.
    parallel : bool or int, optional
        Worker-core setting forwarded to :func:`smlmlp.load_chunking` for memory
        estimation.
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
        ``'chunk_requested'``
            Chunk size requested before resource-token negotiation.
        ``'chunk'``
            Effective chunk size after resource-token negotiation.
        ``'ram_tokens_requested'``
            RAM tokens requested for one full chunk including side padding.
        ``'ram_tokens_minimum'``
            Minimum RAM tokens required for one frame plus side padding.
        ``'ram_tokens_granted'``
            RAM tokens granted by the resource client.
        ``'vram_tokens_requested'``
            VRAM tokens requested when ``cuda=True``, otherwise ``0``.
        ``'vram_tokens_minimum'``
            Minimum VRAM tokens required when ``cuda=True``, otherwise ``0``.
        ``'vram_tokens_granted'``
            VRAM tokens granted when ``cuda=True``, otherwise ``0``.
        ``'ram_token'`` and ``'vram_token'``
            Raw resource-token dictionaries returned by the resource clients.

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
    ValueError
        If ``flip`` does not match the number of channels, or if chunking inputs
        are invalid.
    RuntimeError
        If RAM or VRAM tokens cannot be reserved, or if ``cuda=True`` is used
        while CUDA is unavailable.

    Notes
    -----
    1. TIFF files are opened, their shapes are read, and the file count,
       bounding-box count, dimensionality, and shared frame count are checked.
    2. Missing bounding boxes are replaced by one full-frame channel per file;
       ``chunk`` and ``pad`` are clamped to the acquisition length.
    3. Per-file frame costs are estimated as float32 data and passed to
       :func:`smlmlp.load_chunking` to compute the requested RAM/VRAM tokens.
    4. RAM tokens are reserved, VRAM tokens are reserved when ``cuda=True``, and
       token release callbacks are registered on the context stack.
    5. Granted token counts are passed back to :func:`smlmlp.load_chunking`; if
       the computer grants less than requested, the effective chunk size and
       loop count are recalculated before buffers are allocated.
    6. Each file is exposed either through ``tifffile.memmap`` or through a
       reusable temporary array filled by explicit chunk and padding reads.
    7. Each loaded file view is split by bounding box, optional channel flips
       are applied, and the channel list is yielded with chunk/resource info.

    Examples
    --------
    Iterate over one temporary file without chunking:

    >>> import numpy as np
    >>> import tifffile as tiff
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from smlmlp import load_data
    >>> with TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "movie.tif"
    ...     stack = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    ...     tiff.imwrite(path, stack, photometric="minisblack")
    ...     chunks = list(load_data(str(path), memmap=False))
    ...     info = chunks[0][1]
    ...     len(chunks), len(chunks[0][0]), info["chunk0"], info["chunk1"]
    (1, 1, 0, 2)

    Split two files into channels while chunking the acquisition:

    >>> with TemporaryDirectory() as tmp:
    ...     cam1, cam2 = Path(tmp) / "cam1.tif", Path(tmp) / "cam2.tif"
    ...     stack1 = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    ...     stack2 = np.arange(4 * 4 * 6, dtype=np.uint16).reshape(4, 4, 6)
    ...     tiff.imwrite(cam1, stack1, photometric="minisblack")
    ...     tiff.imwrite(cam2, stack2, photometric="minisblack")
    ...     bboxes = [[(0, 0, 2, 2)], [(0, 0, 3, 2), (3, 0, 6, 2)]]
    ...     loaded = load_data(str(cam1), str(cam2), chunk=2,
    ...                        cameras_bboxes=bboxes, memmap=False)
    ...     chunks = list(loaded)
    ...     [channel.shape for channel in chunks[0][0]]
    [(2, 2, 2), (2, 2, 3), (2, 2, 3)]
    """

    with ExitStack() as stack:
        # Reset block timings
        block.times = {}

        # Open TIFFs
        if len(tif_paths) < 1: raise SyntaxError("Must define at least one tiff file to load")
        tifs = [stack.enter_context(tiff.TiffFile(file)) for file in tif_paths]
        shapes = [shapetif(tif) for tif in tifs]
        nfiles = len(tifs)

        # Validate stack shapes
        nframes = None
        for shape in shapes:
            if nframes is None: nframes = shape[0]
            if len(shape) != 3: raise ValueError(f"Tiff files for SMLM data should have 3 dimensions (time, y, x), not {shape}")
            if shape[0] != nframes: raise ValueError("All tiff files do not have the same number of frames which is not possible for a single SMLM acquisition")
        if nframes < 1: raise ValueError("Tiff files for SMLM data should contain at least one frame")

        # Normalize channel boxes and chunking inputs
        if cameras_bboxes is None: cameras_bboxes = [[(0, 0, shape[2], shape[1])] for shape in shapes]
        if len(cameras_bboxes) != nfiles: raise ValueError("Did not give the same amount of bbox as files")
        nchannels = sum(len(box) for box in cameras_bboxes)
        if flip is not None and len(flip) != nchannels: raise ValueError("flip does not have the same length as channels")
        chunk = nframes if chunk is None else int(chunk)
        chunk = min(max(chunk, 1), nframes)
        pad = min(max(int(pad), 0), nframes - 1)

        # Get frame sizes
        float32_gb = np.dtype(np.float32).itemsize / 1024**3
        channels_frames_pixels = [int(np.prod(shape[1:3])) for shape in shapes]
        channels_frames_sizes_gb = [fp * float32_gb for fp in channels_frames_pixels]

        # Request resource tokens
        chunk_requested = chunk
        chunk, ram, vram, ram_min, vram_min, _ = load_chunking(
            channels_frames_sizes_gb, chunks=chunk, pad=pad, cuda=cuda, parallel=parallel)
        ram_owner = f"load_data:{owner}:ram"
        vram_owner = f"load_data:{owner}:vram"
        ram_token = _take_tokens(computer.ram, ram, ram_owner, ram_min)
        stack.callback(_release_token, computer.ram, ram_token)
        vram_token = None
        if cuda:
            vram_token = _take_tokens(computer.vram, vram, vram_owner, vram_min)
            stack.callback(_release_token, computer.vram, vram_token)

        # Recalculate chunking from granted tokens
        ram_granted = min(ram, _granted_tokens(ram_token, ram))
        vram_granted = min(vram, _granted_tokens(vram_token, vram)) if cuda else None
        chunk, ram_effective, vram_effective, _, _, chunking_info = load_chunking(
            channels_frames_sizes_gb, chunks=chunk_requested, pad=pad,
            ram_tokens=ram_granted, vram_tokens=vram_granted, cuda=cuda, parallel=parallel)
        vram_requested = vram if cuda else 0
        vram_minimum = vram_min if cuda else 0
        vram_effective = vram_effective if cuda else 0
        vram_granted = vram_granted if cuda else 0
        resource_info = dict(
            chunk_requested=chunk_requested, chunk=chunk, ram_token=ram_token, vram_token=vram_token,
            ram_tokens_requested=ram, ram_tokens_minimum=ram_min, ram_tokens_granted=ram_granted,
            ram_tokens_effective=ram_effective, vram_tokens_effective=vram_effective,
            vram_tokens_requested=vram_requested, vram_tokens_minimum=vram_minimum,
            vram_tokens_granted=vram_granted, **chunking_info)

        nloops = int(np.ceil(nframes / chunk))

        # Open memory maps
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

        # Allocate load buffers
        frame_shapes = [shape[1:3] for shape in shapes]
        dtypes = [tif.pages.get(0).dtype for tif in tifs]
        loads = [
            np.empty((chunk + 2 * pad, *shape), dtype=dtype) if mmap is None else None
            for shape, dtype, mmap in zip(frame_shapes, dtypes, mmaps)
        ]

        # Iterate chunks
        for loop in iterator(nloops):
            gc()

            # Temporal layout:
            #
            #             <-- data flux <--
            # | 00pad01 |     0chunk1     | 10pad11 | BEFORE
            # |                loaded               |
            # |unk1     | 10pad11 |    new|array    | AFTER

            # Compute boundaries
            chunk0 = loop * chunk
            chunk1 = (loop + 1) * chunk - 1
            pad00 = chunk0 - pad
            pad01 = chunk0 - 1
            pad10 = chunk1 + 1
            pad11 = chunk1 + pad

            # Build chunk info
            info = dict(resource_info,
                        chunk0=chunk0, chunk1=chunk1, pad00=pad00,
                        pad01=pad01, pad10=pad10, pad11=pad11)

            # Slice memory maps
            mmaps_chunks = [
                mmap[max(0, pad00):min(pad11 + 1, nframes)] if mmap is not None else None
                for mmap in mmaps
            ]

            # Load non-memmap buffers
            with ThreadPoolExecutor(max_workers=nfiles) as pool:
                futures = [
                    pool.submit(
                        _load_one_tif, tif, load,
                        loop, chunk, pad, nframes,
                        chunk1, pad10, pad11,
                    )
                    for tif, load, mmap in zip(tifs, loads, mmaps)
                    if mmap is None
                ]

                for future in futures:
                    future.result()

            # Trim end overflow
            if pad11 + 1 > nframes:
                loads = [load[:nframes - pad00] if load is not None else None for load in loads]

            # Split channels
            channels = []
            count = 0

            for load, mmap, box in zip(loads, mmaps_chunks, cameras_bboxes):
                for bb in box:
                    x0, y0, x1, y1 = bb
                    source = load if mmap is None else mmap
                    channel = source[:, y0:y1, x0:x1]

                    if flip is not None:
                        if flip[count][0]: channel = channel[:, ::-1, :]
                        if flip[count][1]: channel = channel[:, :, ::-1]

                    channels.append(channel)
                    count += 1

            yield channels, info



def _take_tokens(client, value, owner, minimum):
    """Reserve resource tokens and fail when the grant is below the minimum."""
    token = client.take(value, owner, minimum=minimum)
    if token is None or not token.get("ok", False):
        reason = None if token is None else token.get("reason", "unknown reason")
        raise RuntimeError(f"Could not reserve {owner} tokens: {reason}")
    if _granted_tokens(token, value) + 1e-12 < minimum:
        _release_token(client, token)
        raise RuntimeError(f"Could not reserve the minimum {owner} tokens")
    return token



def _granted_tokens(token, requested):
    """Return the granted token count, defaulting to the requested count."""
    return float(token.get("granted_tokens", requested)) if token is not None else 0.0



def _release_token(client, token):
    """Release a resource token when it carries a reservation id."""
    if isinstance(token, dict) and "reservation_id" in token: client.release(token)



def _load_one_tif(tif, load, loop, chunk, pad, nframes, chunk1, pad10, pad11):
    """Load one TIFF chunk with temporal padding into a preallocated buffer."""
    # Split load buffer
    array_pad0 = load[:pad]
    array_chunk = load[-pad - chunk:len(load) - pad]
    array_pad1 = load[len(load) - pad:]

    # Prime right padding
    if pad > 0 and loop == 0: tif.asarray(key=slice(0, pad, 1), out=array_pad1)

    # Shift reusable data
    if pad > 0:
        np.copyto(array_pad0, array_chunk[-pad:], casting="no")
        np.copyto(array_chunk[:pad, :, :], array_pad1, casting="no")

    # Load new chunk frames
    pos0 = min(chunk1 + pad - chunk + 1, nframes)
    pos1 = min(chunk1 + 1, nframes)
    delta = pos1 - pos0

    if pos0 < nframes:
        tif.asarray(key=slice(pos0, pos1, 1), out=array_chunk[pad:pad + delta, :, :])

    # Load new right padding
    if pad > 0:
        pos0 = min(pad10, nframes)
        pos1 = min(pad11 + 1, nframes)
        delta = pos1 - pos0

        if pos0 < nframes:
            tif.asarray(key=slice(pos0, pos1, 1), out=array_pad1[:delta, :, :])



if __name__ == "__main__":
    from corelp import test
    test(__file__)
