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

source_chunk_shape = (12, 12, 5040)
target_chunk_shape = (1, 1, 100000)

#######################################################
### Tests

f = h5py.File(file_path1, 'w')

ds = f.create_dataset('temperature', (len(lat_data), len(lon_data), len(time_data)),  'f4', chunks=source_chunk_shape, compression='lzf')

mem_arr = np.tile(np.linspace(0, 59.5, 120, dtype='f4'), int(math.prod(source_chunk_shape)/120)).reshape(source_chunk_shape)

chunk_len = int(math.prod(source_chunk_shape))

# rng = np.random.default_rng()

for chunk_slices in ds.iter_chunks():
    mem_slices = tuple(slice(0, cs.stop - cs.start) for cs in chunk_slices)
    ds[chunk_slices] = mem_arr[mem_slices]
    # ds[chunk_slices] = rng.normal(13, 5, chunk_len).astype('f4').reshape(source_chunk_shape)[mem_slices]
















































































