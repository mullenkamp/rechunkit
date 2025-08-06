#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 19:38:43 2025

@author: mike
"""
import numpy as np
from math import prod
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
    
    n_reads, n_writes = calc_n_reads_rechunker(source_shape, dtype, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem)

    assert n_reads < n_reads_simple and n_writes == n_chunks_target
    
    n_reads_sel, n_writes_sel = calc_n_reads_rechunker(source_shape, dtype, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel)

    assert n_reads_sel < n_reads and n_writes_sel < n_chunks_target


def test_rechunking_same_shape():
    target = np.zeros(source_shape, dtype=dtype)
    for write_chunk, data in rechunker(source, source_shape, dtype, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem):
        # print(write_chunk)
        target[write_chunk] = data
    
    assert np.all(source(()) == target)


def test_rechunking_selection():
    target = np.zeros(source_shape, dtype=dtype)[sel]
    for write_chunk, data in rechunker(source, source_shape, dtype, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel):
        # print(write_chunk)
        target[write_chunk] = data
    
    assert np.all(source(sel) == target)


