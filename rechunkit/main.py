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


def _count_reads(shape, source_chunk_shape, target_chunk_shape, source_read_chunk_shape):
    """
    Count total source chunk reads for a given buffer shape without building the full plan.
    Mirrors the _rechunk_plan logic exactly for accuracy.
    """
    ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)
    chunk_start = tuple(0 for _ in range(len(shape)))
    n_reads = 0

    if source_read_chunk_shape == ideal_read_chunk_shape:
        for read_chunk_grp in chunk_range(chunk_start, shape, source_read_chunk_shape):
            grp_start = tuple(s.start for s in read_chunk_grp)
            grp_stop = tuple(s.stop for s in read_chunk_grp)
            for _ in chunk_range(grp_start, grp_stop, source_chunk_shape):
                n_reads += 1
    else:
        written_chunks = set()
        for write_chunk in chunk_range(chunk_start, shape, target_chunk_shape):
            write_chunk_start = tuple(s.start for s in write_chunk)
            if write_chunk_start not in written_chunks:
                write_chunk_stop = tuple(s.stop for s in write_chunk)
                read_chunk_start = tuple(rc * (wc // rc) for wc, rc in zip(write_chunk_start, source_chunk_shape))
                read_chunk_stop = tuple(
                    min(max(rcs + rc, wc), sh)
                    for rcs, rc, wc, sh in zip(read_chunk_start, source_read_chunk_shape, write_chunk_stop, shape)
                )

                if all(stop - start <= rcs for start, stop, rcs in zip(read_chunk_start, read_chunk_stop, source_read_chunk_shape)):
                    is_end_chunk = any(wc.stop == ts for wc, ts in zip(write_chunk, shape))
                    for write_chunk1 in chunk_range(write_chunk_start, read_chunk_stop, target_chunk_shape, include_partial_chunks=is_end_chunk, clip_ends=False):
                        write_chunk2 = tuple(slice(wc.start, min(wc.stop, s)) for wc, s in zip(write_chunk1, shape))
                        if all(all((wc.stop - wcs <= src, wc.start < wc.stop)) for wcs, wc, src in zip(read_chunk_start, write_chunk2, source_read_chunk_shape)):
                            write_chunk1_start = tuple(s.start for s in write_chunk2)
                            if write_chunk1_start not in written_chunks:
                                written_chunks.add(write_chunk1_start)

                    for _ in chunk_range(read_chunk_start, read_chunk_stop, source_chunk_shape, True, False):
                        n_reads += 1
                else:
                    written_chunks.add(write_chunk_start)
                    for rc in chunk_range(read_chunk_start, read_chunk_stop, source_chunk_shape, True, False):
                        if all(max(rc_s.start, wc_s.start) < min(rc_s.stop, wc_s.stop) for rc_s, wc_s in zip(rc, write_chunk)):
                            n_reads += 1

    return n_reads


def _greedy_read_chunk_factors(source_chunk_shape, k_factors, capacity):
    """
    Greedy heuristic to find multiplier factors that fit in capacity and approximate
    the aspect ratio of k_factors. Returns list of ints.
    """
    source_len = len(source_chunk_shape)
    total_k = prod(k_factors)
    scale = (capacity / total_k) ** (1.0 / source_len)

    new_factors = [max(1, int(k * scale)) for k in k_factors]

    while prod(new_factors) > capacity:
        idx = max(range(source_len), key=lambda i: new_factors[i])
        new_factors[idx] = max(1, new_factors[idx] - 1)

    while True:
        candidates = [i for i in range(source_len) if new_factors[i] < k_factors[i]]
        if not candidates:
            break
        candidates.sort(key=lambda i: k_factors[i] / new_factors[i], reverse=True)

        grew = False
        curr_prod = prod(new_factors)
        for idx in candidates:
            if curr_prod * (new_factors[idx] + 1) // new_factors[idx] <= capacity:
                new_factors[idx] += 1
                grew = True
                break
        if not grew:
            break

    return new_factors


def _trim_factors(new_factors, source_chunk_shape, target_chunk_shape):
    """
    Trim factors so they don't extend past the last target chunk they cover.
    Returns a tuple of the final read chunk shape.
    """
    final_factors = []
    for n, s, t in zip(new_factors, source_chunk_shape, target_chunk_shape):
        m = (n * s) // t
        if m == 0:
            req_n = (t + s - 1) // s
            final_factors.append(min(n, req_n))
        else:
            req_n = (m * t + s - 1) // s
            final_factors.append(req_n)
    return tuple(f * s for f, s in zip(final_factors, source_chunk_shape))


def calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem, shape=None):
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
    shape: tuple of int or None
        The full array shape. When provided, enables candidate search to find
        the buffer shape with fewest source reads. When None, uses greedy heuristic only.

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

    # Constrained: ideal doesn't fit. Find best multiple of source_chunk_shape.
    capacity = max_cells // tot_source
    if capacity < 1:
        return source_chunk_shape

    k_factors = [i // s for i, s in zip(ideal_chunks, source_chunk_shape)]

    # Greedy baseline
    greedy_factors = _greedy_read_chunk_factors(source_chunk_shape, k_factors, capacity)
    greedy_shape = _trim_factors(greedy_factors, source_chunk_shape, target_chunk_shape)

    # If no shape provided, return greedy result (backward compatible)
    if shape is None:
        return greedy_shape

    # Generate candidate shapes and pick the one with fewest reads.
    # For small search spaces, exhaustively try all valid factor combinations.
    # For large ones, use targeted heuristic candidates.
    total_k = prod(k_factors)
    candidates = {greedy_shape}

    if total_k <= 500:
        # Exhaustive search over all valid factor combinations
        ranges = [range(1, k + 1) for k in k_factors]
        for factors in itertools.product(*ranges):
            if prod(factors) <= capacity:
                candidates.add(_trim_factors(list(factors), source_chunk_shape, target_chunk_shape))
    else:
        # Heuristic candidates for large search spaces

        # Round each dim down to nearest multiple of target chunk
        for dim in range(source_len):
            factors = list(greedy_factors)
            target_multiple = target_chunk_shape[dim] // source_chunk_shape[dim]
            if target_multiple > 1 and factors[dim] > target_multiple:
                factors[dim] = (factors[dim] // target_multiple) * target_multiple
                if prod(factors) <= capacity:
                    candidates.add(_trim_factors(factors, source_chunk_shape, target_chunk_shape))

        # Round each dim down to nearest multiple of LCM factor
        for dim in range(source_len):
            factors = list(greedy_factors)
            lcm_k = k_factors[dim]
            if lcm_k > 1 and factors[dim] > 1:
                rounded = (factors[dim] // lcm_k) * lcm_k
                if rounded >= 1:
                    factors[dim] = rounded
                    if prod(factors) <= capacity:
                        candidates.add(_trim_factors(factors, source_chunk_shape, target_chunk_shape))

        # Try ±1-2 around greedy factors per dimension
        for dim in range(source_len):
            for delta in (-2, -1, 1, 2):
                factors = list(greedy_factors)
                new_val = factors[dim] + delta
                if 1 <= new_val <= k_factors[dim]:
                    factors[dim] = new_val
                    if prod(factors) <= capacity:
                        candidates.add(_trim_factors(factors, source_chunk_shape, target_chunk_shape))

        # For each pair of dims, try shifting capacity between them
        if source_len >= 2:
            for d1 in range(source_len):
                for d2 in range(d1 + 1, source_len):
                    for shrink in range(1, min(greedy_factors[d1], 4)):
                        factors = list(greedy_factors)
                        if factors[d1] - shrink >= 1:
                            factors[d1] -= shrink
                            curr = prod(factors)
                            while curr * (factors[d2] + 1) // factors[d2] <= capacity and factors[d2] < k_factors[d2]:
                                factors[d2] += 1
                                curr = prod(factors)
                            candidates.add(_trim_factors(factors, source_chunk_shape, target_chunk_shape))
                    for shrink in range(1, min(greedy_factors[d2], 4)):
                        factors = list(greedy_factors)
                        if factors[d2] - shrink >= 1:
                            factors[d2] -= shrink
                            curr = prod(factors)
                            while curr * (factors[d1] + 1) // factors[d1] <= capacity and factors[d1] < k_factors[d1]:
                                factors[d1] += 1
                                curr = prod(factors)
                            candidates.add(_trim_factors(factors, source_chunk_shape, target_chunk_shape))

    # Filter: all candidates must fit in memory and be >= source_chunk_shape per dim
    valid = []
    for c in candidates:
        if prod(c) <= max_cells and all(ci >= si for ci, si in zip(c, source_chunk_shape)):
            valid.append(c)

    if not valid:
        return greedy_shape

    if len(valid) == 1:
        return valid[0]

    # Pre-filter by fast estimate to limit expensive _count_reads calls.
    # Always include greedy_shape to guarantee no regression.
    max_score = 8

    if len(valid) <= max_score:
        scored = valid
    else:
        def _fast_estimate(c):
            n_groups = prod(ceil(s / r) for s, r in zip(shape, c))
            chunks_per_group = prod(r // s for r, s in zip(c, source_chunk_shape))
            return n_groups * chunks_per_group

        valid.sort(key=_fast_estimate)
        scored = list(valid[:max_score])
        if greedy_shape not in scored:
            scored.append(greedy_shape)

    best_shape = min(scored, key=lambda c: _count_reads(shape, source_chunk_shape, target_chunk_shape, c))

    return best_shape


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


def _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel=None, source_read_chunk_shape=None):
    """
    Internal generator that yields rechunking plan entries. Each entry is a tuple:
        (group_type, read_chunks, write_chunks, group_start)

    - group_type: 'bulk' (reads fill a buffer, writes extract from it) or 'single' (one write chunk, reads clipped to it)
    - read_chunks: list of tuple-of-slices for source reads (in target coordinate space)
    - write_chunks: list of tuple-of-slices for target writes (in target coordinate space)
    - group_start: tuple of ints, reference point for buffer offset calculations
    """
    if source_read_chunk_shape is None:
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
    if sel is None:
        target_shape = shape
    else:
        target_shape = tuple(s.stop - s.start for s in sel)

    source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem, shape=target_shape)

    n_reads = 0
    n_writes = 0
    for group_type, read_chunks, write_chunks, group_start in _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel, source_read_chunk_shape=source_read_chunk_shape):
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

    if sel is None:
        chunk_read_offset = tuple(0 for i in range(len(shape)))
        target_shape = shape
    else:
        chunk_read_offset = tuple(s.start for s in sel)
        target_shape = tuple(s.stop - s.start for s in sel)

    source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem, shape=target_shape)
    buffer_shape = tuple(max(s, t) for s, t in zip(source_read_chunk_shape, target_chunk_shape))
    mem_arr1 = np.zeros(buffer_shape, dtype=dtype)

    # For canonical yield ordering: compute strides for C-order target chunk index
    n_chunks_per_dim = tuple(ceil(s / c) for s, c in zip(target_shape, target_chunk_shape))

    def _canon_idx(chunk_slices):
        idx = 0
        for s, c, nc in zip(chunk_slices, target_chunk_shape, n_chunks_per_dim):
            idx = idx * nc + (s.start // c)
        return idx

    pending = {}
    next_idx = 0

    for group_type, read_chunks, write_chunks, group_start in _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel, source_read_chunk_shape=source_read_chunk_shape):
        if group_type == 'bulk':
            for read_chunk in read_chunks:
                read_chunk1 = tuple(slice(rc.start + cro, min(rc.stop + cro, s)) for rc, cro, s in zip(read_chunk, chunk_read_offset, shape))
                offset_slices = tuple(slice(rc1.start - gs - cro, rc1.stop - gs - cro) for gs, cro, rc1 in zip(group_start, chunk_read_offset, read_chunk1))
                mem_arr1[offset_slices] = source(read_chunk1)

            for write_chunk in write_chunks:
                offset_slices = tuple(slice(wc.start - gs, wc.stop - gs) for gs, wc in zip(group_start, write_chunk))
                idx = _canon_idx(write_chunk)
                if idx == next_idx:
                    yield write_chunk, mem_arr1[offset_slices]
                    next_idx += 1
                    while next_idx in pending:
                        yield pending.pop(next_idx)
                        next_idx += 1
                else:
                    pending[idx] = (write_chunk, mem_arr1[offset_slices].copy())

        else:  # single
            write_chunk = write_chunks[0]
            mem_read_chunk_slice = tuple(slice(0, wc.stop - wc.start) for wc in write_chunk)
            for read_chunk in read_chunks:
                read_chunk1 = tuple(slice(rc.start + cro, min(rc.stop + cro, s)) for rc, cro, s in zip(read_chunk, chunk_read_offset, shape))
                clip_read_chunk = get_slice_min_max(read_chunk, write_chunk)
                read_slice = tuple(slice(cc.start - rc.start, cc.stop - rc.start) for cc, rc in zip(clip_read_chunk, read_chunk))
                write_slice = tuple(slice(cc.start - rc.start, cc.stop - rc.start) for cc, rc in zip(clip_read_chunk, write_chunk))
                mem_arr1[write_slice] = source(read_chunk1)[read_slice]

            idx = _canon_idx(write_chunk)
            if idx == next_idx:
                yield write_chunk, mem_arr1[mem_read_chunk_slice]
                next_idx += 1
                while next_idx in pending:
                    yield pending.pop(next_idx)
                    next_idx += 1
            else:
                pending[idx] = (write_chunk, mem_arr1[mem_read_chunk_slice].copy())

    while next_idx in pending:
        yield pending.pop(next_idx)
        next_idx += 1


























































