#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 19:38:43 2025

@author: mike
"""
import numpy as np
import pytest
from math import prod, ceil
from rechunkit import guess_chunk_shape, chunk_range, calc_ideal_read_chunk_shape, calc_ideal_read_chunk_mem, calc_source_read_chunk_shape, calc_n_chunks, calc_n_reads_simple, calc_n_reads_rechunker, rechunker

###################################################
### Chunker testing

# source_shape = (31, 31)
source_shape = (31, 31, 31)
shape = source_shape

# sel = (slice(3, 21), slice(11, 27))
sel = (slice(3, 21), slice(11, 27), slice(7, 17))

# source_chunk_shape = (5, 2)
# target_chunk_shape = (2, 5)
source_chunk_shape = (5, 2, 4)
target_chunk_shape = (4, 5, 3)
# max_mem = 120 # smaller than the ideal chunk size
max_mem = 2000 # smaller than the ideal chunk size

dtype = np.dtype('int32')

source = np.arange(1, prod(source_shape) + 1, dtype=dtype).reshape(source_shape)
source = source.__getitem__


def test_calcs():
    ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)

    ideal_read_chunk_mem = calc_ideal_read_chunk_mem(ideal_read_chunk_shape, dtype.itemsize)

    assert ideal_read_chunk_mem > max_mem

    source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, dtype.itemsize, max_mem)

    n_chunks_source = calc_n_chunks(source_shape, source_chunk_shape)
    n_chunks_target = calc_n_chunks(source_shape, target_chunk_shape)

    n_reads_simple = calc_n_reads_simple(source_shape, source_chunk_shape, target_chunk_shape)

    n_reads, n_writes = calc_n_reads_rechunker(source_shape, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem)

    assert n_reads < n_reads_simple and n_writes == n_chunks_target

    n_reads_sel, n_writes_sel = calc_n_reads_rechunker(source_shape, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel)

    assert n_reads_sel < n_reads and n_writes_sel < n_chunks_target


def test_rechunking_same_shape():
    target = np.zeros(source_shape, dtype=dtype)
    for write_chunk, data in rechunker(source, source_shape, dtype, source_chunk_shape, target_chunk_shape, max_mem):
        # print(write_chunk)
        target[write_chunk] = data

    assert np.all(source(()) == target)


def test_rechunking_selection():
    target = np.zeros(source_shape, dtype=dtype)[sel]
    for write_chunk, data in rechunker(source, source_shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel):
        # print(write_chunk)
        target[write_chunk] = data

    assert np.all(source(sel) == target)


###################################################
### Helper to verify rechunking correctness

def _verify_rechunk(src_shape, src_chunks, tgt_chunks, mem, dt=np.dtype('int32'), selection=None):
    """
    Build a source array, rechunk it, and assert the target matches.
    Returns (n_reads, n_writes) from calc_n_reads_rechunker.
    """
    src_arr = np.arange(1, prod(src_shape) + 1, dtype=dt).reshape(src_shape)
    src_func = src_arr.__getitem__

    if selection is None:
        tgt_shape = src_shape
    else:
        tgt_shape = tuple(s.stop - s.start for s in selection)

    target = np.zeros(tgt_shape, dtype=dt)
    for write_chunk, data in rechunker(src_func, src_shape, dt, src_chunks, tgt_chunks, mem, selection):
        target[write_chunk] = data

    if selection is None:
        expected = src_arr
    else:
        expected = src_arr[selection]

    assert np.all(expected == target), "Rechunked data does not match expected"
    return calc_n_reads_rechunker(src_shape, dt.itemsize, src_chunks, tgt_chunks, mem, selection)


###################################################
### 1D tests

def test_rechunk_1d():
    _verify_rechunk((100,), (7,), (11,), 500)


def test_rechunk_1d_selection():
    _verify_rechunk((100,), (7,), (11,), 500, selection=(slice(10, 80),))


def test_rechunk_1d_tiny_mem():
    """Force constrained single-chunk path with very small memory."""
    _verify_rechunk((50,), (7,), (11,), 50)


###################################################
### 2D tests

def test_rechunk_2d():
    _verify_rechunk((31, 31), (5, 2), (2, 5), 200)


def test_rechunk_2d_selection():
    _verify_rechunk((31, 31), (5, 2), (2, 5), 200, selection=(slice(3, 21), slice(11, 27)))


def test_rechunk_2d_large_mem():
    """Ideal path: memory is large enough for the LCM read chunk."""
    shape = (31, 31)
    src_c = (5, 2)
    tgt_c = (2, 5)
    ideal_mem = calc_ideal_read_chunk_mem(calc_ideal_read_chunk_shape(src_c, tgt_c), 4)
    _verify_rechunk(shape, src_c, tgt_c, ideal_mem * 2)


###################################################
### Identical chunk shapes (no actual rechunking)

def test_rechunk_same_chunks():
    _verify_rechunk((20, 20), (5, 5), (5, 5), 500)


def test_rechunk_same_chunks_selection():
    _verify_rechunk((20, 20), (5, 5), (5, 5), 500, selection=(slice(2, 18), slice(3, 15)))


###################################################
### Target chunks are multiples of source (or vice versa)

def test_rechunk_target_multiple_of_source():
    """Target chunks evenly divide into source — LCM equals target."""
    _verify_rechunk((24, 24), (4, 4), (12, 12), 2000)


def test_rechunk_source_multiple_of_target():
    """Source chunks evenly divide into target — LCM equals source."""
    _verify_rechunk((24, 24), (12, 12), (4, 4), 2000)


###################################################
### Shapes not evenly divisible by chunk shapes (partial edge chunks)

def test_rechunk_partial_chunks_2d():
    _verify_rechunk((17, 13), (5, 4), (3, 7), 500)


def test_rechunk_partial_chunks_3d():
    _verify_rechunk((11, 13, 7), (3, 5, 2), (4, 3, 5), 1000)


def test_rechunk_partial_chunks_selection():
    _verify_rechunk((17, 13), (5, 4), (3, 7), 500, selection=(slice(2, 15), slice(1, 11)))


###################################################
### Different dtypes

def test_rechunk_float64():
    _verify_rechunk((20, 20), (6, 4), (4, 6), 1000, dt=np.dtype('float64'))


def test_rechunk_int8():
    _verify_rechunk((50, 30), (7, 11), (10, 6), 500, dt=np.dtype('int8'))


###################################################
### Read count optimality tests

def test_ideal_mem_reads_equal_source_chunks():
    """With ideal memory, each source chunk is read exactly once."""
    shape = (24, 24)
    src_c = (6, 4)
    tgt_c = (4, 6)
    itemsize = 4
    ideal_mem = calc_ideal_read_chunk_mem(calc_ideal_read_chunk_shape(src_c, tgt_c), itemsize)

    n_reads, n_writes = calc_n_reads_rechunker(shape, itemsize, src_c, tgt_c, ideal_mem)
    n_source_chunks = calc_n_chunks(shape, src_c)
    n_target_chunks = calc_n_chunks(shape, tgt_c)

    assert n_reads == n_source_chunks, f"Ideal reads {n_reads} != source chunks {n_source_chunks}"
    assert n_writes == n_target_chunks


def test_ideal_mem_reads_equal_source_chunks_3d():
    """With ideal memory in 3D, each source chunk is read exactly once."""
    shape = (24, 30, 20)
    src_c = (6, 5, 4)
    tgt_c = (4, 6, 5)
    itemsize = 4
    ideal_mem = calc_ideal_read_chunk_mem(calc_ideal_read_chunk_shape(src_c, tgt_c), itemsize)

    n_reads, n_writes = calc_n_reads_rechunker(shape, itemsize, src_c, tgt_c, ideal_mem)
    n_source_chunks = calc_n_chunks(shape, src_c)
    n_target_chunks = calc_n_chunks(shape, tgt_c)

    assert n_reads == n_source_chunks
    assert n_writes == n_target_chunks


def test_rechunker_always_fewer_reads_than_simple():
    """Optimized reads should never exceed the brute-force count."""
    configs = [
        ((31, 31, 31), (5, 2, 4), (4, 5, 3), 2000),
        ((50,), (7,), (11,), 100),
        ((17, 13), (5, 4), (3, 7), 200),
        ((24, 24), (6, 4), (4, 6), 300),
        # ((11, 13, 7), (3, 5, 2), (4, 3, 5), 500),
    ]
    itemsize = 4
    for shp, src_c, tgt_c, mem in configs:
        n_reads, n_writes = calc_n_reads_rechunker(shp, itemsize, src_c, tgt_c, mem)
        n_simple = calc_n_reads_simple(shp, src_c, tgt_c)
        assert n_reads <= n_simple, f"shape={shp}: optimized {n_reads} > simple {n_simple}"


def test_writes_always_equal_target_chunks():
    """Number of writes should always equal the total target chunks."""
    configs = [
        ((31, 31, 31), (5, 2, 4), (4, 5, 3), 2000),
        ((50,), (7,), (11,), 100),
        ((17, 13), (5, 4), (3, 7), 200),
        ((24, 24), (6, 4), (4, 6), 300),
        ((24, 24), (12, 12), (4, 4), 2000),
    ]
    itemsize = 4
    for shp, src_c, tgt_c, mem in configs:
        n_reads, n_writes = calc_n_reads_rechunker(shp, itemsize, src_c, tgt_c, mem)
        n_target_chunks = calc_n_chunks(shp, tgt_c)
        assert n_writes == n_target_chunks, f"shape={shp}: writes {n_writes} != target chunks {n_target_chunks}"


def test_more_mem_means_fewer_or_equal_reads():
    """Increasing max_mem should never increase the read count."""
    shape = (31, 31, 31)
    src_c = (5, 2, 4)
    tgt_c = (4, 5, 3)
    itemsize = 4

    prev_reads = None
    for mem in [200, 500, 2000, 6000, 10000]:
        n_reads, _ = calc_n_reads_rechunker(shape, itemsize, src_c, tgt_c, mem)
        if prev_reads is not None:
            assert n_reads <= prev_reads, f"mem={mem}: reads {n_reads} > prev {prev_reads}"
        prev_reads = n_reads


###################################################
### Selection edge cases

def test_selection_full_extent():
    """Selection covering the full array should match no-selection results."""
    shape = (20, 20)
    src_c = (6, 4)
    tgt_c = (4, 6)
    itemsize = 4
    full_sel = (slice(0, 20), slice(0, 20))

    n_reads, n_writes = calc_n_reads_rechunker(shape, itemsize, src_c, tgt_c, 500)
    n_reads_sel, n_writes_sel = calc_n_reads_rechunker(shape, itemsize, src_c, tgt_c, 500, full_sel)

    assert n_reads == n_reads_sel
    assert n_writes == n_writes_sel

    _verify_rechunk(shape, src_c, tgt_c, 500, selection=full_sel)


def test_selection_chunk_aligned():
    """Selection aligned to both source and target chunk boundaries."""
    _verify_rechunk((24, 24), (6, 4), (4, 6), 1000, selection=(slice(0, 12), slice(4, 24)))


def test_selection_single_element_wide():
    """Selection that is 1 element wide in one dimension."""
    _verify_rechunk((20, 20), (6, 4), (4, 6), 500, selection=(slice(5, 6), slice(0, 20)))


def test_selection_step_validation():
    """Slices with step != 1 should raise ValueError."""
    src_arr = np.arange(100, dtype='int32').reshape(10, 10)
    with pytest.raises(ValueError, match='step'):
        list(rechunker(src_arr.__getitem__, (10, 10), np.dtype('int32'), (3, 3), (5, 5), 500, sel=(slice(0, 10, 2), slice(0, 10))))

    with pytest.raises(ValueError, match='step'):
        calc_n_reads_rechunker((10, 10), 4, (3, 3), (5, 5), 500, sel=(slice(0, 10, 2), slice(0, 10)))


###################################################
### guess_chunk_shape edge cases

def test_guess_chunk_shape_small_target():
    """When target size is very small, chunks should shrink."""
    cs = guess_chunk_shape((100, 100, 100), 4, 100)
    assert prod(cs) * 4 < 200  # within 50% tolerance


def test_guess_chunk_shape_large_target():
    """When target is larger than the array, chunk shape equals array shape."""
    shape = (10, 10, 10)
    cs = guess_chunk_shape(shape, 4, 100000)
    assert cs == shape


def test_guess_chunk_shape_0dim():
    """0-dimensional shape should return empty tuple."""
    assert guess_chunk_shape((), 4) == ()


###################################################
### chunk_range edge cases

def test_chunk_range_none_start():
    """Passing None as chunk_start should default to zeros."""
    chunks = list(chunk_range(None, (10, 10), (5, 5)))
    assert len(chunks) == 4
    assert chunks[0] == (slice(0, 5), slice(0, 5))


###################################################
### calc_n_chunks consistency

def test_calc_n_chunks_exact():
    assert calc_n_chunks((20, 20), (5, 5)) == 16
    assert calc_n_chunks((20, 20), (7, 7)) == 9  # ceil(20/7) * ceil(20/7) = 3 * 3


def test_calc_n_chunks_1d():
    assert calc_n_chunks((100,), (10,)) == 10
    assert calc_n_chunks((101,), (10,)) == 11

