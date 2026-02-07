"""Core rechunking algorithm stuff."""
# import copy
from typing import List, Optional, Sequence, Tuple, Iterator, Generator
import numpy as np
import itertools
# from time import time
from math import prod, lcm, ceil
from collections import Counter, deque
from collections.abc import Callable
from itertools import count
from bisect import bisect

########################################################
### Parameters

composite_numbers = (1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280, 720720, 1081080, 1441440, 2162160)

########################################################
### Functions


def guess_chunk_shape(shape: Tuple[int, ...], itemsize: int, target_chunk_size: int = 2**21) -> Tuple[int, ...]:
    """
    Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as target_chunk_size. Chunks will be assigned to the highest composite number within the target_chunk_size. Using composite numbers will benefit the rehunking process as there is a very high likelihood that the least common multiple of two composite numbers will be significantly lower than the product of those two numbers.

    Parameters
    ----------
    shape: tuple of ints
        Shape of the array.
    itemsize: int
        The byte size of the data type. It must be a numpy bytes size: 1, 2, 4, or 8
    target_chunk_size: int
        The maximum size per chunk in bytes.

    Returns
    -------
    tuple of ints
        shape of the chunk
    """
    ndims = len(shape)

    if ndims > 0:

        if not all(isinstance(v, int) for v in shape):
            raise TypeError('All values in the shape must be ints.')

        chunks = list(shape)
        
        idx = 0
        while True:
            chunk_bytes = prod(chunks)*itemsize

            if chunk_bytes <= target_chunk_size * 1.5:
                break

            if prod(chunks) == 1:
                break

            # Find the largest composite number <= current_dim / 2
            current_dim = chunks[idx % ndims]
            search_val = (current_dim // 2) - 1
            pos = bisect(composite_numbers, search_val)
            
            if pos == 0:
                new_val = 1
            else:
                new_val = composite_numbers[pos - 1]
            
            chunks[idx % ndims] = new_val
            idx += 1

        return tuple(chunks)
    else:
        return ()


def get_slice_min_max(read_slices, write_slices):
    """
    Function to get the max start position and the min stop position.
    """
    slices = tuple(slice(max(rs.start, ws.start), min(rs.stop, ws.stop)) for rs, ws in zip(read_slices, write_slices))

    return slices


def chunk_range(
    chunk_start: Tuple[int, ...], chunk_stop: Tuple[int, ...], chunk_step: Tuple[int, ...], include_partial_chunks=True, clip_ends=True,
) -> Iterator[Tuple[slice, ...]]:
    """
    Generator like the Python range function, but for multiple dimensions and it returns tuples of slices.

    Parameters
    ----------
    chunk_start: tuple of int
        The start positions of the chunks.
    chunk_stop: tuple of int
        The stop positions of the chunks.
    chunk_step: tuple of int
        The chunking step.
    include_partial_chunks: bool
        Should partial chunks be included? True by default.
    clip_ends: bool
        Only applies when include_partial_chunks == True. Should the chunks be clipped to the overall extents? True by default.

    Returns
    -------
    Generator with tuples of slices
    """
    if chunk_start is None:
        chunk_start = tuple(0 for i in range(len(chunk_stop)))

    if include_partial_chunks:
        start_ranges = [cs * (sc//cs) for cs, sc in zip(chunk_step, chunk_start)]
    else:
        start_ranges = [cs * (((sc - 1)//cs) + 1) for cs, sc in zip(chunk_step, chunk_start)]

    ranges = [range(sr, ec, cs) for ec, cs, sr in zip(chunk_stop, chunk_step, start_ranges)]

    for indices in itertools.product(*ranges):
        # print(indices)
        inside = True
        res = []
        for i, ec, cs, sc in zip(indices, chunk_stop, chunk_step, chunk_start):
            stop = i + cs
            if stop > ec:
                if clip_ends:
                    stop = ec
                inside = False

            start = i
            if start < sc:
                if clip_ends:
                    start = sc
                inside = False

            res.append(slice(start, stop))

        if inside or include_partial_chunks:
            yield tuple(res)


def calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape):
    """
    Calculates the minimum ideal read chunk shape between a source and target.
    """
    return tuple(lcm(i, s) for i, s in zip(source_chunk_shape, target_chunk_shape))


def calc_ideal_read_chunk_mem(ideal_read_chunk_shape, itemsize):
    """
    Calculates the minimum ideal read chunk memory between a source and target.
    """
    return int(prod(ideal_read_chunk_shape) * itemsize)


def calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem):
    """
    Calculates the optimum read chunk shape given a maximum amount of available memory.

    Parameters
    ----------
    source_chunk_shape: tuple of int
        The source chunk shape
    target_chunk_shape: tuple of int
        The target chunk shape
    itemsize: int
        The byte length of the data type.
    max_mem: int
        The max allocated memory to perform the chunking operation in bytes.

    Returns
    -------
    optimal chunk shape: tuple of ints
    """
    max_cells = max_mem // itemsize
    source_len = len(source_chunk_shape)
    target_len = len(target_chunk_shape)

    if source_len != target_len:
        raise ValueError('The source_chunk_shape and target_chunk_shape do not have the same number of dims.')

    tot_source = prod(source_chunk_shape)
    if tot_source >= max_cells:
        return source_chunk_shape

    # Calculate ideal (LCM) shape
    ideal_chunks = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)
    tot_ideal = prod(ideal_chunks)

    if tot_ideal <= max_cells:
        return ideal_chunks

    # If ideal doesn't fit, we need to find a multiple of source_chunk_shape
    # that fits in max_mem and approximates the aspect ratio of ideal_chunks.
    
    # Calculate how many source chunks we can fit
    capacity = max_cells // tot_source
    if capacity < 1:
        return source_chunk_shape # Should have been caught by tot_source >= max_cells
    
    # Calculate the scaling factor for each dimension from source to ideal
    # ideal_chunks[i] = k_i * source_chunk_shape[i]
    k_factors = [i // s for i, s in zip(ideal_chunks, source_chunk_shape)]
    
    # We want to find new factors n_i <= k_i such that prod(n_i) <= capacity
    # To preserve aspect ratio, we want n_i proportional to k_i.
    
    total_k = prod(k_factors)
    scale = (capacity / total_k) ** (1.0 / source_len)
    
    new_factors = [max(1, int(k * scale)) for k in k_factors]
    
    # Refine new_factors to ensure prod(new_factors) <= capacity
    while prod(new_factors) > capacity:
        # Shrink the largest factor > 1
        idx = max(range(source_len), key=lambda i: new_factors[i])
        new_factors[idx] = max(1, new_factors[idx] - 1)

    # Grow to fill remaining capacity
    while True:
        candidates = [i for i in range(source_len) if new_factors[i] < k_factors[i]]
        if not candidates:
             break
             
        # Heuristic: Grow the one with largest (k/n) ratio (most compressed)
        candidates.sort(key=lambda i: k_factors[i]/new_factors[i], reverse=True)
        
        grew = False
        curr_prod = prod(new_factors)
        for idx in candidates:
             if curr_prod * (new_factors[idx] + 1) // new_factors[idx] <= capacity:
                 new_factors[idx] += 1
                 grew = True
                 break # Re-evaluate from top
        
        if not grew:
            break

    # Trim waste: Reduce factors if they exceed what's needed for the target chunks covered
    final_factors = []
    for n, s, t in zip(new_factors, source_chunk_shape, target_chunk_shape):
        m = (n * s) // t
        if m == 0:
            # Limit to covering 1 target chunk if possible
            req_n = (t + s - 1) // s
            final_factors.append(min(n, req_n))
        else:
            # Limit to covering m target chunks
            req_n = (m * t + s - 1) // s
            final_factors.append(req_n)

    return tuple(f * s for f, s in zip(final_factors, source_chunk_shape))


def calc_n_chunks_per_read(source_chunk_shape, source_read_chunk_shape):
    """

    """
    return prod(tuple(nc//sc for nc, sc in zip(source_read_chunk_shape, source_chunk_shape)))


def calc_n_chunks(shape, chunk_shape):
    """

    """
    return prod(ceil(s / c) for s, c in zip(shape, chunk_shape))


def calc_n_reads_simple(shape, source_chunk_shape, target_chunk_shape):
    """
    Brute force chunking read count. Every target chunk must iterate over every associated source chunk. This should be considered the maximum number of reads between a source and target (most inefficient). The number of writes is the total number of chunks in the target.

    Parameters
    ----------
    shape: tuple of ints
        The shape of the source dataset, which will also be the shape of the target dataset.
    dtype: np.dtype
        The numpy data type of the source/target.
    source_chunk_shape: tuple of ints
        The chunk_shape of the source.
    target_chunk_shape: tuple of ints
        The chunk_shape of the target.

    Returns
    -------
    int
        Count of the number of reads
    """
    chunk_start = tuple(0 for i in range(len(shape)))
    read_counter = count()

    for write_chunk in chunk_range(chunk_start, shape, target_chunk_shape):
        write_chunk_start = tuple(rc.start for rc in write_chunk)
        write_chunk_stop = tuple(rc.stop for rc in write_chunk)
        for chunk_slice in chunk_range(write_chunk_start, write_chunk_stop, source_chunk_shape):
            next(read_counter)

    return next(read_counter)


def _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel=None):
    """
    Internal generator that yields rechunking plan entries. Each entry is a tuple:
        (group_type, read_chunks, write_chunks, group_start)

    - group_type: 'bulk' (reads fill a buffer, writes extract from it) or 'single' (one write chunk, reads clipped to it)
    - read_chunks: list of tuple-of-slices for source reads (in target coordinate space)
    - write_chunks: list of tuple-of-slices for target writes (in target coordinate space)
    - group_start: tuple of ints, reference point for buffer offset calculations
    """
    source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)
    ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)

    chunk_start = tuple(0 for i in range(len(shape)))

    if sel is None:
        target_shape = shape
    else:
        for s, sh in zip(sel, shape):
            if s.start < 0 or s.stop > sh:
                raise ValueError('The selection must be a subset of the source.')
            if s.step is not None and s.step != 1:
                raise ValueError('The selection slices must have a step of 1 or None.')
        target_shape = tuple(s.stop - s.start for s in sel)

    if source_read_chunk_shape == ideal_read_chunk_shape:
        ## Ideal case: read chunks fill the buffer exactly, each source chunk is read once
        for read_chunk_grp in chunk_range(chunk_start, target_shape, source_read_chunk_shape):
            grp_start = tuple(s.start for s in read_chunk_grp)
            grp_stop = tuple(s.stop for s in read_chunk_grp)

            read_chunks = list(chunk_range(grp_start, grp_stop, source_chunk_shape))
            write_chunks = list(chunk_range(grp_start, grp_stop, target_chunk_shape))

            yield ('bulk', read_chunks, write_chunks, grp_start)

    else:
        ## Constrained case: buffer is smaller than ideal, some source chunks may be read multiple times
        written_chunks = set()

        for write_chunk in chunk_range(chunk_start, target_shape, target_chunk_shape):
            write_chunk_start = tuple(s.start for s in write_chunk)
            if write_chunk_start not in written_chunks:
                write_chunk_stop = tuple(s.stop for s in write_chunk)

                read_chunk_start = tuple(rc * (wc//rc) for wc, rc in zip(write_chunk_start, source_chunk_shape))
                read_chunk_stop = tuple(min(max(rcs + rc, wc), sh) for rcs, rc, wc, sh in zip(read_chunk_start, source_read_chunk_shape, write_chunk_stop, target_shape))

                read_chunks = list(chunk_range(read_chunk_start, read_chunk_stop, source_chunk_shape, True, False))

                if all(stop - start <= rcs for start, stop, rcs in zip(read_chunk_start, read_chunk_stop, source_read_chunk_shape)):
                    ## Bulk: read region fits in buffer, can serve multiple write chunks
                    is_end_chunk = any(wc.stop == ts for wc, ts in zip(write_chunk, target_shape))
                    write_chunks = []
                    for write_chunk1 in chunk_range(write_chunk_start, read_chunk_stop, target_chunk_shape, include_partial_chunks=is_end_chunk, clip_ends=False):
                        write_chunk2 = tuple(slice(wc.start, min(wc.stop, s)) for wc, s in zip(write_chunk1, target_shape))
                        if all(all((wc.stop - wcs <= src, wc.start < wc.stop)) for wcs, wc, src in zip(read_chunk_start, write_chunk2, source_read_chunk_shape)):
                            write_chunk1_start = tuple(s.start for s in write_chunk2)
                            if write_chunk1_start not in written_chunks:
                                write_chunks.append(write_chunk2)
                                written_chunks.add(write_chunk1_start)

                    yield ('bulk', read_chunks, write_chunks, read_chunk_start)

                else:
                    ## Single: read region exceeds buffer, handle one write chunk at a time
                    written_chunks.add(write_chunk_start)

                    overlapping = [rc for rc in read_chunks if all(cc.start < cc.stop for cc in get_slice_min_max(rc, write_chunk))]
                    yield ('single', overlapping, [write_chunk], write_chunk_start)


def calc_n_reads_rechunker(shape: Tuple[int, ...], itemsize: int,  source_chunk_shape: Tuple[int, ...], target_chunk_shape: Tuple[int, ...], max_mem: int, sel=None) -> Tuple[int, int]:
    """
    This function calculates the total number of reads (and writes) using the more optimized rechunking algorithm. It optimises the rechunking by using an in-memory numpy ndarray with a size defined by the max_mem provided by the user.

    Parameters
    ----------
    shape: tuple of ints
        The shape of the source dataset, which will also be the shape of the target dataset.
    itemsize: int
        The byte length of the data type.
    source_chunk_shape: tuple of ints
        The chunk_shape of the source.
    target_chunk_shape: tuple of ints
        The chunk_shape of the target.
    max_mem: int
        The max allocated memory to perform the chunking operation in bytes.
    sel: tuple of slices
        A subset selection of the source in the form of a tuple of slices. The starts and stops must be within the shape of the source.

    Returns
    -------
    tuple
        number of reads, number of writes
    """
    n_reads = 0
    n_writes = 0
    for group_type, read_chunks, write_chunks, group_start in _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel):
        n_reads += len(read_chunks)
        n_writes += len(write_chunks)
    return n_reads, n_writes


def rechunker(source: Callable, shape: Tuple[int, ...], dtype: np.dtype, source_chunk_shape: Tuple[int, ...], target_chunk_shape: Tuple[int, ...], max_mem: int, sel=None, itemsize: int=None) -> Iterator[Tuple[Tuple[slice, ...], np.ndarray]]:
    """
    This function takes a source dataset function with a specific chunk_shape and returns a generator that converts to a new chunk_shape. It optimises the rechunking by using an in-memory numpy ndarray with a size defined by the max_mem provided by the user.

    Parameters
    ----------
    source: callable function/method
        The source function/method to read the dataset/array. The function must have a single parameter input as a tuple of slices to retrieve an array chunk of data.
    shape: tuple of ints
        The shape of the source dataset, which will also be the shape of the target dataset unless sel is passed.
    dtype: np.dtype
        The numpy data type of the source/target.
    source_chunk_shape: tuple of ints
        The chunk_shape of the source.
    target_chunk_shape: tuple of ints
        The chunk_shape of the target.
    max_mem: int
        The max allocated memory to perform the chunking operation in bytes. This will only be as large as necessary for an optimum size chunk for the rechunking.
    sel: tuple of slices or None
        A subset selection of the source in the form of a tuple of slices. The starts and stops must be within the shape of the source.
    itemsize: int or None
        The byte length of the data type. Only necessary to explicitly assign when using numpy StringDTypes.

    Returns
    -------
    Generator
        tuple of the target slices to the np.ndarray of data
    """
    if not isinstance(itemsize, int):
        itemsize = dtype.itemsize

    source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)
    buffer_shape = tuple(max(s, t) for s, t in zip(source_read_chunk_shape, target_chunk_shape))
    mem_arr1 = np.zeros(buffer_shape, dtype=dtype)

    if sel is None:
        chunk_read_offset = tuple(0 for i in range(len(shape)))
    else:
        chunk_read_offset = tuple(s.start for s in sel)

    for group_type, read_chunks, write_chunks, group_start in _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel):
        if group_type == 'bulk':
            for read_chunk in read_chunks:
                read_chunk1 = tuple(slice(rc.start + cro, min(rc.stop + cro, s)) for rc, cro, s in zip(read_chunk, chunk_read_offset, shape))
                offset_slices = tuple(slice(rc1.start - gs - cro, rc1.stop - gs - cro) for gs, cro, rc1 in zip(group_start, chunk_read_offset, read_chunk1))
                mem_arr1[offset_slices] = source(read_chunk1)

            for write_chunk in write_chunks:
                offset_slices = tuple(slice(wc.start - gs, wc.stop - gs) for gs, wc in zip(group_start, write_chunk))
                yield write_chunk, mem_arr1[offset_slices]

        else:  # single
            write_chunk = write_chunks[0]
            mem_read_chunk_slice = tuple(slice(0, wc.stop - wc.start) for wc in write_chunk)
            for read_chunk in read_chunks:
                read_chunk1 = tuple(slice(rc.start + cro, min(rc.stop + cro, s)) for rc, cro, s in zip(read_chunk, chunk_read_offset, shape))
                clip_read_chunk = get_slice_min_max(read_chunk, write_chunk)
                read_slice = tuple(slice(cc.start - rc.start, cc.stop - rc.start) for cc, rc in zip(clip_read_chunk, read_chunk))
                write_slice = tuple(slice(cc.start - rc.start, cc.stop - rc.start) for cc, rc in zip(clip_read_chunk, write_chunk))
                mem_arr1[write_slice] = source(read_chunk1)[read_slice]

            yield write_chunk, mem_arr1[mem_read_chunk_slice]


























































