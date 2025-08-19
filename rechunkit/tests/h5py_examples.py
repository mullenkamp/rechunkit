#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:10:04 2025

@author: mike
"""
import numpy as np
import h5py
import rechunkit
import pathlib
import math

#######################################################
### Parameters

data_path = pathlib.Path('/home/mike/data/cache')
file_path1 = data_path.joinpath('test1.h5')
file_path2 = data_path.joinpath('test2.h5')

lat_data = np.linspace(0, 9.9, 100, dtype='float32')
lon_data = np.linspace(-5, 4.9, 100, dtype='float32')
time_data = np.linspace(0, 10, 100000, dtype='datetime64[D]')

itemsize = 4

shape = (len(lat_data), len(lon_data), len(time_data))

source_chunk_shape = rechunkit.guess_chunk_shape(shape, itemsize, 2**20)
target_chunk_shape = (1, 1, 100000)

#######################################################
### Tests

f = h5py.File(file_path1, 'w')

ds = f.create_dataset('temperature', shape,  'f4', chunks=source_chunk_shape, compression='lzf')

mem_arr = np.tile(np.linspace(0, 59.5, 120, dtype='f4'), int(math.prod(source_chunk_shape)/120)).reshape(source_chunk_shape)

chunk_len = int(math.prod(source_chunk_shape))

# rng = np.random.default_rng()

for chunk_slices in ds.iter_chunks():
    mem_slices = tuple(slice(0, cs.stop - cs.start) for cs in chunk_slices)
    ds[chunk_slices] = mem_arr[mem_slices]
    # ds[chunk_slices] = rng.normal(13, 5, chunk_len).astype('f4').reshape(source_chunk_shape)[mem_slices]


n_chunks_source = rechunkit.calc_n_chunks(shape, source_chunk_shape)
n_chunks_target = rechunkit.calc_n_chunks(shape, target_chunk_shape)

ideal_read_chunk_shape = rechunkit.calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)
ideal_read_chunk_mem = rechunkit.calc_ideal_read_chunk_mem(ideal_read_chunk_shape, itemsize)

n_reads_simple = rechunkit.calc_n_reads_simple(shape, source_chunk_shape, target_chunk_shape)

n_reads, n_writes = rechunkit.calc_n_reads_rechunker(shape, itemsize, source_chunk_shape, target_chunk_shape, 2**27)

## New dataset
ds2 = f.create_dataset('temperature2', shape, 'f4', chunks=target_chunk_shape, compression='lzf')

rechunker = rechunkit.rechunker(ds.__getitem__, shape, 'f4', itemsize, source_chunk_shape, target_chunk_shape, 2**24)

for write_chunk, data in rechunker:
    ds2[write_chunk] = data

## Check - takes ~10 mins
for chunk_slices in ds.iter_chunks():
    close = np.allclose(ds[chunk_slices], ds2[chunk_slices])
    if not close:
        raise ValueError()


f.close()

































































