#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 19:38:43 2025

@author: mike
"""
import numpy as np
from math import prod
from rechunkit.main import guess_chunk_shape, chunk_range, calc_ideal_read_chunk_shape, calc_ideal_read_chunk_mem, calc_source_read_chunk_shape, calc_n_chunks, calc_n_reads_simple, calc_n_reads_rechunker, rechunker

###################################################
### Chunker testing

source_shape = (31, 31)
shape = source_shape
# new_shape = (35, 31)

sel = (slice(3, 21), slice(11, 25))
sel = (slice(5, 20), slice(10, 24))

source_chunk_shape = (5, 2)
target_chunk_shape = (2, 5)
# source_chunk_shape = (5, 2, 4)
# target_chunk_shape = (2, 5, 3)
itemsize = 4
max_mem = 40 * itemsize
max_mem = 160 * itemsize

dtype = np.dtype('int32')

source = np.arange(1, prod(source_shape) + 1, dtype=dtype).reshape(source_shape)
source = source.__getitem__

# chunk_read_offset = tuple(s.start for s in sel)

# chunk_start = tuple(cs * (ss.start//cs) for cs, ss in zip(source_chunk_shape, sel))
# # chunk_start = tuple(ss.start for ss in self._sel)
# chunk_end = tuple(cs * ((ss.stop - 1)//cs + 1) for cs, ss in zip(source_chunk_shape, sel))

# shape = tuple(cs * ((ss.stop - ss.start - 1)//cs + 1) for cs, ss in zip(target_chunk_shape, sel))

# shape = tuple(ss.stop - ss.start for ss in sel)



# source_read_chunk_shape = out_chunks[0]
# inter_chunks = out_chunks[1]
# target_read_chunks = out_chunks[2]

source_read_chunk_shape = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)


n_reads_simple = calc_n_reads_simple(source_shape, source_chunk_shape, target_chunk_shape)

n_reads, n_writes = calc_n_reads_rechunker(source_shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel)


target = np.zeros(source_shape, dtype=dtype)[sel]
for write_chunk, data in rechunker(source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel):
    # print(write_chunk)
    target[write_chunk] = data

np.all(source(sel) == target)


target = np.zeros(source_shape, dtype=dtype)
for write_chunk, data in rechunker(source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem):
    # print(write_chunk)
    target[write_chunk] = data

np.all(source(()) == target)