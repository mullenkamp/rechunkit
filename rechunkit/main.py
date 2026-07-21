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

# Above this many target chunks, the read-shape candidate check is skipped
# (planning cost stays bounded; the greedy shape is used unchanged).
_SCORE_LIMIT = 20_000

########################################################
### Functions


def guess_chunk_shape(shape: Tuple[int, ...], itemsize: int, target_chunk_size: int = 2**21) -> Tuple[int, ...]:
    """
    Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes. The returned chunk's pre-compressed
    size targets target_chunk_size and may exceed it by up to 1.5x (the
    shrink loop stops once within that band). Chunk dims are snapped to
    composite numbers: the least common multiple of two composite numbers
    is very likely far smaller than their product, which benefits later
    rechunking between two guessed layouts.

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

        if not all(isinstance(v, (int, np.integer)) for v in shape):
            raise TypeError('All values in the shape must be ints.')

        if any(v <= 0 for v in shape):
            raise ValueError('All values in the shape must be > 0. A chunk shape cannot have a zero-length dim.')

        chunks = [int(v) for v in shape]

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


def _buffer_bytes(read_chunk_shape, target_chunk_shape, phase, itemsize):
    """
    The TRUE bytes the rechunker buffer allocates for a given read chunk shape:
    per-dim max(read + phase, target), multiplied out. This is the single
    source of truth shared by the planner (calc_source_read_chunk_shape) and
    the generator's allocation (rechunker) — they must never diverge.
    """
    return prod(max(r + p, t) for r, p, t in zip(read_chunk_shape, phase, target_chunk_shape)) * itemsize


def calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape, shape=None):
    """
    Calculates the minimum ideal read chunk shape between a source and target.

    Parameters
    ----------
    source_chunk_shape: tuple of int
        The source chunk shape.
    target_chunk_shape: tuple of int
        The target chunk shape.
    shape: tuple of int or None
        The extent the read groups tile (the array shape, or the selection
        shape when rechunking a selection). When given, each dim's LCM is
        clipped to the smallest source-aligned extent covering the dim:
        cross-group alignment only matters where a dim has two or more
        groups, and the clipped extent produces exactly one group in that
        dim. Without it, non-divisor targets can inflate the "ideal" by
        orders of magnitude past anything the array can use.
    """
    if shape is None:
        return tuple(lcm(s, t) for s, t in zip(source_chunk_shape, target_chunk_shape))

    ideal = []
    for s, t, sh in zip(source_chunk_shape, target_chunk_shape, shape):
        l = lcm(s, t)
        if sh > 0:
            cover = ceil(sh / s) * s   # smallest source-aligned cover of the dim
            if l > cover:
                l = cover
        ideal.append(l)
    return tuple(ideal)


def calc_ideal_read_chunk_mem(ideal_read_chunk_shape, itemsize):
    """
    Calculates the minimum ideal read chunk memory between a source and target.
    """
    return int(prod(ideal_read_chunk_shape) * itemsize)


def calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem, shape=None, phase=None):
    """
    Calculates the optimum read chunk shape given a maximum amount of available memory.

    The memory budget is applied to the TRUE buffer allocation — per-dim
    max(read + phase, target) — not just the read region, so the returned
    shape keeps the rechunker's actual allocation within max_mem whenever
    that is feasible.

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
        The extent the read groups tile (array shape, or selection shape when
        rechunking a selection) — clips the ideal read shape per dim; see
        calc_ideal_read_chunk_shape.
    phase: tuple of int or None
        The per-dim misalignment of the selection start against the source
        chunk grid (``sel.start % source_chunk_shape``). Defaults to zeros.
        When calling this standalone for a misaligned selection without
        passing phase, the result is a lower bound on the real allocation
        (phase adds up to source_chunk_shape − 1 per dim to the buffer).

    Returns
    -------
    optimal chunk shape: tuple of ints

    Notes
    -----
    The irreducible floor is one source chunk: when even
    ``_buffer_bytes(source_chunk_shape)`` (which includes the full target
    chunk in every dim where the target is larger) exceeds max_mem, the
    source chunk shape is returned and the actual allocation will exceed
    the budget. The rechunker's batched single path handles those cases
    with per-target-chunk buffers instead of this bulk buffer.
    """
    source_len = len(source_chunk_shape)
    target_len = len(target_chunk_shape)

    if source_len != target_len:
        raise ValueError('The source_chunk_shape and target_chunk_shape do not have the same number of dims.')

    if phase is None:
        phase = tuple(0 for _ in range(source_len))

    def buf(read_shape):
        return _buffer_bytes(read_shape, target_chunk_shape, phase, itemsize)

    def enforce_pending(read_shape, budget):
        # Bulk groups spanning multiple target-chunk rows park later-row
        # copies in the canonical-order pending dict; count that band
        # toward the budget and shrink the read shape (outermost dim
        # first, in source-chunk steps) until buffer + pending fit.
        # Skipped when the extent is unknown, or when the shape is already
        # over budget (the batched single path takes over there and has no
        # bulk pending). Note read shapes never exceed the per-dim LCM
        # (the clip takes min(lcm, cover)), so there is no lcm-multiple
        # shrink stage — source-chunk steps are the only granularity.
        # When every dim is at one source chunk the residual band is
        # physically irreducible without re-reading (documented).
        if shape is None or buf(read_shape) > budget:
            return read_shape

        def total(rs):
            return buf(rs) + _pending_bytes(rs, shape, target_chunk_shape, itemsize)

        rs = list(read_shape)
        while total(rs) > budget:
            # Shrink the outermost dim whose shrink STRICTLY reduces
            # buffer + pending — a shrink that doesn't (e.g. a faster dim
            # under a constant outer-band term) only degrades reads.
            for d in range(source_len):
                if rs[d] > source_chunk_shape[d]:
                    trial = list(rs)
                    trial[d] -= source_chunk_shape[d]
                    if total(trial) < total(rs):
                        rs = trial
                        break
            else:
                break  # no shrink helps: irreducible residual (documented)
        return tuple(rs)

    # Calculate ideal (LCM) shape, clipped to the tiled extent when known
    ideal_chunks = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape, shape)

    # Floor: even a single source chunk (with target-sized buffer dims)
    # busts the budget — the allocation will exceed max_mem no matter what
    # (documented above). Growing the read in dims where the target already
    # forces the buffer size is memory-FREE, so take that free growth
    # (bounded by the ideal) instead of pinning to one source chunk.
    if buf(source_chunk_shape) >= max_mem:
        floor_shape = []
        for s, t, p, i in zip(source_chunk_shape, target_chunk_shape, phase, ideal_chunks):
            cap = max(s + p, t)          # the dim extent the buffer has anyway
            m = (cap - p) // s           # largest multiple of s with m*s + p <= cap
            floor_shape.append(min(i, max(s, m * s)))
        return tuple(floor_shape)

    enforced_ideal = None
    if buf(ideal_chunks) <= max_mem:
        enforced_ideal = enforce_pending(ideal_chunks, max_mem)
        if enforced_ideal == ideal_chunks or shape is None:
            # True ideal (or no extent to score against): done. A
            # pending-degraded ideal falls through to the candidate check
            # below instead of being returned unexamined.
            return enforced_ideal

    # If ideal doesn't fit, we need to find a multiple of source_chunk_shape
    # that fits the budget and approximates the aspect ratio of ideal_chunks.

    k_factors = [i // s for i, s in zip(ideal_chunks, source_chunk_shape)]

    def read_shape(factors):
        return tuple(f * s for f, s in zip(factors, source_chunk_shape))

    def greedy_shape(budget):
        # Seed with a cell-count estimate; the loops below enforce the true
        # buffer budget. ideal_chunks[i] = k_i * source_chunk_shape[i]; we
        # want factors n_i <= k_i, proportional to k_i, whose true buffer
        # fits the budget.
        max_cells = budget // itemsize
        tot_source = prod(source_chunk_shape)
        capacity = max(1, max_cells // tot_source)

        total_k = prod(k_factors)
        scale = (capacity / total_k) ** (1.0 / source_len)

        new_factors = [max(1, int(k * scale)) for k in k_factors]

        # Refine to ensure the true buffer fits (floor: all-1 factors)
        while buf(read_shape(new_factors)) > budget and any(f > 1 for f in new_factors):
            # Shrink the largest factor > 1
            idx = max(range(source_len), key=lambda i: new_factors[i])
            new_factors[idx] = max(1, new_factors[idx] - 1)

        # Grow to fill remaining budget
        while True:
            grow_dims = [i for i in range(source_len) if new_factors[i] < k_factors[i]]
            if not grow_dims:
                break

            # Heuristic: Grow the one with largest (k/n) ratio (most compressed)
            grow_dims.sort(key=lambda i: k_factors[i] / new_factors[i], reverse=True)

            grew = False
            for idx in grow_dims:
                grown = list(new_factors)
                grown[idx] += 1
                if buf(read_shape(grown)) <= budget:
                    new_factors[idx] += 1
                    grew = True
                    break  # Re-evaluate from top

            if not grew:
                break

        # Trim waste: Reduce factors that exceed what's needed for the
        # target chunks covered
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

        return enforce_pending(read_shape(final_factors), budget)

    greedy = greedy_shape(max_mem)

    # Candidate check (constrained regime only): the greedy heuristic can
    # pick target-grid-misaligned shapes, and the honest budget accounting
    # can flip bulk/batch regimes non-monotonically in max_mem. Score a
    # small candidate set by running the ACTUAL planner in counting mode
    # (no mirror to drift) and keep the shape with the fewest reads; ties
    # prefer the target-aligned snap. Skipped for large plans (bounded
    # planning cost) — the greedy shape is always in the set, so behavior
    # above the limit is unchanged.
    if shape is None:
        return greedy
    n_target_chunks = prod(ceil(ts / t) for ts, t in zip(shape, target_chunk_shape))
    if n_target_chunks == 0 or n_target_chunks > _SCORE_LIMIT:
        return enforced_ideal if enforced_ideal is not None else greedy

    def aligned_snap():
        # Snap greedy's factors down to multiples of the ideal's per-dim
        # factor (group boundaries then land on both grids in that dim),
        # then regrow in aligned steps while the true buffer fits.
        g_factors = [g // s for g, s in zip(greedy, source_chunk_shape)]
        snapped = []
        for nd, qd in zip(g_factors, k_factors):
            if qd > 1 and nd >= qd:
                snapped.append((nd // qd) * qd)
            else:
                snapped.append(max(1, nd))
        while True:
            grew = False
            for d in range(source_len):
                step = k_factors[d] if (k_factors[d] > 0 and snapped[d] >= k_factors[d]) else 1
                cand = list(snapped)
                cand[d] += step
                if cand[d] <= max(k_factors[d], 1) and buf(read_shape(cand)) <= max_mem:
                    snapped = cand
                    grew = True
            if not grew:
                break
        return enforce_pending(read_shape(snapped), max_mem)

    candidates = []
    cand_pool = [aligned_snap()]
    if enforced_ideal is not None:
        cand_pool.append(enforced_ideal)
    cand_pool.extend((greedy, greedy_shape(max_mem // 2), source_chunk_shape))
    for cand in cand_pool:
        if cand not in candidates:
            candidates.append(cand)

    if len(candidates) == 1:
        return candidates[0]

    # Score by running _rechunk_plan itself with the candidate threaded in
    # (a synthetic sel reproduces the phase geometry exactly).
    shape_syn = tuple(p + ts for p, ts in zip(phase, shape))
    sel_syn = tuple(slice(p, p + ts) for p, ts in zip(phase, shape))

    def count_reads(cand):
        n = 0
        for _gt, reads, _w, _gs in _rechunk_plan(shape_syn, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel_syn, _read_chunk_shape=cand):
            n += len(reads)
        return n

    return min(candidates, key=count_reads)


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


def _exact_chunk_range(start, stop, step, clip_ends=True):
    """
    Like chunk_range but starts exactly at `start` without floor-aligning.
    This is needed when the start position is phase-shifted to align reads
    with source chunk boundaries in source coordinate space.
    """
    dim_ranges = [range(s, e, c) for s, e, c in zip(start, stop, step)]
    for indices in itertools.product(*dim_ranges):
        yield tuple(
            slice(i, (min(i + c, e) if clip_ends else i + c))
            for i, c, e in zip(indices, step, stop)
        )


def _pending_bytes(read_chunk_shape, target_shape, target_chunk_shape, itemsize):
    """
    Conservative upper bound on the bytes the canonical-order ``pending``
    dict holds for bulk groups: chunks materialized by a read group but not
    yet due in C-order are copied and retained until the group's row band
    completes. Dominant term per dim d: (rows_d − 1) × (target chunks in
    the faster dims) × target-chunk bytes, active only when some faster dim
    has more than one read group.
    """
    ndims = len(read_chunk_shape)
    tgt_bytes = prod(target_chunk_shape) * itemsize
    total = 0
    for d in range(ndims):
        rows_d = ceil(read_chunk_shape[d] / target_chunk_shape[d])
        if rows_d <= 1:
            continue
        groups_faster = prod(ceil(target_shape[i] / read_chunk_shape[i]) for i in range(d + 1, ndims))
        if groups_faster <= 1:
            continue
        chunks_faster = prod(ceil(target_shape[i] / target_chunk_shape[i]) for i in range(d + 1, ndims))
        total += (rows_d - 1) * chunks_faster * tgt_bytes
    return total


def _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel=None, _read_chunk_shape=None):
    """
    Internal generator that yields rechunking plan entries. Each entry is a tuple:
        (group_type, read_chunks, write_chunks, group_start)

    - group_type: 'bulk' (reads fill a buffer, writes extract from it) or
      'multi' (a batch of write chunks whose read regions exceed the bulk
      buffer; reads are deduped across the batch and scattered into
      per-write-chunk buffers)
    - read_chunks: list of tuple-of-slices for source reads (in target coordinate space)
    - write_chunks: list of tuple-of-slices for target writes (in target coordinate space)
    - group_start: tuple of ints, reference point for buffer offset
      calculations ('bulk' only; None for 'multi')
    """
    # Parse sel FIRST so the planner sees the true phase — the read shape
    # chosen here and the buffer allocated in rechunker() must be computed
    # from identical arguments (see _buffer_bytes).
    if sel is None:
        target_shape = shape
        phase = tuple(0 for _ in range(len(shape)))
    else:
        for s, sh in zip(sel, shape):
            if s.start < 0 or s.stop > sh:
                raise ValueError('The selection must be a subset of the source.')
            if s.step is not None and s.step != 1:
                raise ValueError('The selection slices must have a step of 1 or None.')
        target_shape = tuple(s.stop - s.start for s in sel)
        phase = tuple(s.start % sc for s, sc in zip(sel, source_chunk_shape))

    # _read_chunk_shape is a scoring-mode override threaded as a PARAMETER
    # (never a module-global patch — plans may run concurrently): the
    # candidate check in calc_source_read_chunk_shape scores candidate read
    # shapes by running this very planner in counting mode.
    if _read_chunk_shape is None:
        source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem, shape=target_shape, phase=phase)
    else:
        source_read_chunk_shape = _read_chunk_shape
    ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape, target_shape)

    chunk_start = tuple(0 for i in range(len(shape)))

    if source_read_chunk_shape == ideal_read_chunk_shape:
        ## Ideal case: read chunks fill the buffer exactly, each source chunk is read once
        # Keep groups at original positions but extend reads backward by phase
        # to align with source chunk boundaries in source space.
        # group_start is shifted back by phase so buffer offsets work correctly.
        for read_chunk_grp in chunk_range(chunk_start, target_shape, source_read_chunk_shape):
            grp_start_orig = tuple(s.start for s in read_chunk_grp)
            grp_stop = tuple(s.stop for s in read_chunk_grp)

            # Shift group_start backward by phase for source-aligned reads
            grp_start = tuple(gs - p for gs, p in zip(grp_start_orig, phase))

            # Reads from phase-shifted start, aligned to source chunks
            # Filter out reads entirely outside [0, target_shape)
            read_chunks = [rc for rc in _exact_chunk_range(grp_start, grp_stop, source_chunk_shape)
                           if all(s.stop > 0 for s in rc)]
            # Write chunks at original positions (unchanged)
            write_chunks = list(chunk_range(grp_start_orig, grp_stop, target_chunk_shape))

            yield ('bulk', read_chunks, write_chunks, grp_start)

    else:
        ## Constrained case: buffer is smaller than ideal, some source chunks may be read multiple times
        written_chunks = set()

        # Batched single path: write chunks whose read region exceeds the
        # bulk buffer are collected (they arrive in C-order) and their reads
        # deduped, so each source chunk is read once per batch instead of
        # once per write chunk. The batch quota subtracts one transient
        # source chunk, the bulk buffer (if a bulk read shape is feasible),
        # and the bulk pending band — the terms that can be live at the same
        # time. When n_batch clamps to 1 the peak can exceed max_mem by the
        # irreducible floor (one target chunk + one source chunk).
        src_bytes = prod(source_chunk_shape) * itemsize
        tgt_bytes = prod(target_chunk_shape) * itemsize

        def _bulk_possible(rs):
            # A bulk group exists iff SOME write chunk's aligned read region
            # fits the buffer in every dim (the grid is a product space, so
            # dims are independent). When none can, the bulk buffer never
            # allocates and no pending forms — the batch quota must not pay
            # for them.
            for d in range(len(rs)):
                t = target_chunk_shape[d]
                s = source_chunk_shape[d]
                p = phase[d]
                ts = target_shape[d]
                found = False
                for k in range(ceil(ts / t) if ts else 0):
                    start = k * t
                    stop = min(start + t, ts)
                    aligned = s * ((start + p) // s) - p
                    if stop - aligned <= rs[d]:
                        found = True
                        break
                if not found:
                    return False
            return True

        use_bulk = (_buffer_bytes(source_read_chunk_shape, target_chunk_shape, phase, itemsize) <= max_mem
                    and _bulk_possible(source_read_chunk_shape))
        if use_bulk:
            reserve = _buffer_bytes(source_read_chunk_shape, target_chunk_shape, phase, itemsize)
            pending_bound = _pending_bytes(source_read_chunk_shape, target_shape, target_chunk_shape, itemsize)
            # Bulk mode must be AFFORDABLE: if the bulk buffer + its pending
            # band leave room for fewer than TWO batch buffers, the mixed
            # plan's batches can't dedup reads (n_batch=1 == one-chunk-at-a-
            # time) — measured strictly worse than the all-multi plan at the
            # same budget, and the regime flip is outside the candidate
            # check's reach (it varies bulk on/off at a FIXED shape).
            if (max_mem - src_bytes - reserve - pending_bound) // tgt_bytes < 2:
                use_bulk = False
                reserve = 0
                pending_bound = 0
        else:
            reserve = 0
            pending_bound = 0
        n_batch = max(1, (max_mem - src_bytes - reserve - pending_bound) // tgt_bytes)

        batch_writes = []
        batch_reads = {}

        def _flush_batch():
            group = ('multi', list(batch_reads.values()), list(batch_writes), None)
            batch_reads.clear()
            batch_writes.clear()
            return group

        for write_chunk in chunk_range(chunk_start, target_shape, target_chunk_shape):
            write_chunk_start = tuple(s.start for s in write_chunk)
            if write_chunk_start not in written_chunks:
                write_chunk_stop = tuple(s.stop for s in write_chunk)

                # Align read_chunk_start to source chunks in source space
                read_chunk_start = tuple(sc * ((wc + p) // sc) - p for wc, sc, p in zip(write_chunk_start, source_chunk_shape, phase))
                read_chunk_stop = tuple(min(max(rcs + rc, wc), sh) for rcs, rc, wc, sh in zip(read_chunk_start, source_read_chunk_shape, write_chunk_stop, target_shape))

                read_chunks = [rc for rc in _exact_chunk_range(read_chunk_start, read_chunk_stop, source_chunk_shape, clip_ends=False)
                               if all(s.stop > 0 for s in rc)]

                if use_bulk and all(stop - start <= rcs for start, stop, rcs in zip(read_chunk_start, read_chunk_stop, source_read_chunk_shape)):
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

                    # Flush any open batch BEFORE the bulk group so batch
                    # buffers don't sit parked across bulk processing.
                    if batch_writes:
                        yield _flush_batch()
                    yield ('bulk', read_chunks, write_chunks, read_chunk_start)

                else:
                    ## Batched single: read region exceeds the bulk buffer;
                    ## accumulate the write chunk and its (deduped) reads.
                    written_chunks.add(write_chunk_start)

                    batch_writes.append(write_chunk)
                    for rc in read_chunks:
                        if all(cc.start < cc.stop for cc in get_slice_min_max(rc, write_chunk)):
                            batch_reads.setdefault(tuple(s.start for s in rc), rc)

                    if len(batch_writes) >= n_batch:
                        yield _flush_batch()

        if batch_writes:
            yield _flush_batch()


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
        The memory budget for the rechunking operation in bytes. This bounds
        the sum of the read buffer, the canonical-order pending copies, and
        the batched-single-path target buffers — with two documented
        exceptions: the irreducible floors (bulk path: one source chunk plus
        one target chunk, i.e. per-dim max(source, target); batched path:
        one target chunk plus one source chunk), and the wide-array residual
        when a single source chunk spans multiple target-chunk rows
        (pending can then exceed the budget by up to
        (ceil(src0/tgt0) - 1) * row_width * itemsize).
    sel: tuple of slices or None
        A subset selection of the source in the form of a tuple of slices. The starts and stops must be within the shape of the source.
    itemsize: int or None
        The byte length of the data type. Only necessary to explicitly assign when using numpy StringDTypes.

    Returns
    -------
    Generator
        tuple of the target slices to the np.ndarray of data

    Notes
    -----
    Yielded arrays may be views into the generator's internal buffer, which is
    reused as iteration advances. Consume (or ``.copy()``) each yielded array
    BEFORE advancing the generator, and treat yielded arrays as read-only —
    holding references across iterations or mutating them in place produces
    silently wrong data.
    """
    if not isinstance(itemsize, int):
        itemsize = dtype.itemsize

    # Parse sel FIRST, compute the read shape ONCE, and thread it into
    # _rechunk_plan via the override — the buffer allocated below and the
    # plan's read shape are then identical BY CONSTRUCTION (bulk-path buffer
    # indexing depends on it), and the candidate scoring inside
    # calc_source_read_chunk_shape runs once instead of twice.
    if sel is None:
        chunk_read_offset = tuple(0 for i in range(len(shape)))
        target_shape = shape
        phase = tuple(0 for _ in range(len(shape)))
    else:
        chunk_read_offset = tuple(s.start for s in sel)
        target_shape = tuple(s.stop - s.start for s in sel)
        phase = tuple(s.start % sc for s, sc in zip(sel, source_chunk_shape))

    source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem, shape=target_shape, phase=phase)

    # Buffer must accommodate source_read_chunk_shape + phase to hold
    # phase-shifted reads that extend before the group's original start.
    # Allocated LAZILY on the first bulk group: pure-multi plans (every write
    # chunk on the batched single path) never need it.
    buffer_shape = tuple(max(s + p, t) for s, t, p in zip(source_read_chunk_shape, target_chunk_shape, phase))
    mem_arr1 = None

    # For canonical yield ordering: compute strides for C-order target chunk index
    n_chunks_per_dim = tuple(ceil(s / c) for s, c in zip(target_shape, target_chunk_shape))

    def _canon_idx(chunk_slices):
        idx = 0
        for s, c, nc in zip(chunk_slices, target_chunk_shape, n_chunks_per_dim):
            idx = idx * nc + (s.start // c)
        return idx

    pending = {}
    next_idx = 0

    for group_type, read_chunks, write_chunks, group_start in _rechunk_plan(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel, _read_chunk_shape=source_read_chunk_shape):
        if group_type == 'bulk':
            if mem_arr1 is None:
                mem_arr1 = np.zeros(buffer_shape, dtype=dtype)
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

        else:  # multi: batched single path
            # One dedicated buffer per batched write chunk; each (deduped)
            # read calls source() ONCE and scatters into every overlapping
            # buffer using the single-path intersection arithmetic.
            buffers = [np.empty(tuple(w.stop - w.start for w in wc), dtype=dtype)
                       for wc in write_chunks]
            for read_chunk in read_chunks:
                read_chunk1 = tuple(slice(rc.start + cro, min(rc.stop + cro, s)) for rc, cro, s in zip(read_chunk, chunk_read_offset, shape))
                data = source(read_chunk1)
                for write_chunk, buffer_arr in zip(write_chunks, buffers):
                    clip_read_chunk = get_slice_min_max(read_chunk, write_chunk)
                    if all(cc.start < cc.stop for cc in clip_read_chunk):
                        read_slice = tuple(slice(cc.start - rc.start, cc.stop - rc.start) for cc, rc in zip(clip_read_chunk, read_chunk))
                        write_slice = tuple(slice(cc.start - wc.start, cc.stop - wc.start) for cc, wc in zip(clip_read_chunk, write_chunk))
                        buffer_arr[write_slice] = data[read_slice]

            for i, write_chunk in enumerate(write_chunks):
                buffer_arr = buffers[i]
                buffers[i] = None  # release our reference once handed off
                idx = _canon_idx(write_chunk)
                if idx == next_idx:
                    yield write_chunk, buffer_arr
                    next_idx += 1
                    while next_idx in pending:
                        yield pending.pop(next_idx)
                        next_idx += 1
                else:
                    # dedicated buffer — parking it needs no copy
                    pending[idx] = (write_chunk, buffer_arr)
            buffers = None

    while next_idx in pending:
        yield pending.pop(next_idx)
        next_idx += 1
