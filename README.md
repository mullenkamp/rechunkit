# rechunkit

<p align="center">
    <em>Functions to efficiently rechunk multidimensional arrays</em>
</p>

[![build](https://github.com/mullenkamp/rechunkit/workflows/Build/badge.svg)](https://github.com/mullenkamp/rechunkit/actions)
[![codecov](https://codecov.io/gh/mullenkamp/rechunkit/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/rechunkit)
[![PyPI version](https://badge.fury.io/py/rechunkit.svg)](https://badge.fury.io/py/rechunkit)

---

**Source Code**: <a href="https://github.com/mullenkamp/rechunkit" target="_blank">https://github.com/mullenkamp/rechunkit</a>

---
## Introduction
Rechunkit is a set of functions to allow efficient rechunking of multidimensional arrays that have been stored as chunks of numpy ndarrays. It allows for rechunking on-the-fly via python generators instead of requiring the user to save the full target array. It also contains several other handy tools for assisting the user as part of the rechunking process (e.g. estimating an optimal or ideal chunking size, iterating over chunks with a range-type function, etc).  


## Installation
```
pip install rechunkit
```
I can add it to conda-forge if there is demand.

## Usage
Import the necessary modules and assign some parameters for the examples:

```python
from rechunkit import guess_chunk_shape, chunk_range, calc_ideal_read_chunk_shape, calc_ideal_read_chunk_mem, calc_source_read_chunk_shape, calc_n_chunks, calc_n_reads_simple, calc_n_reads_rechunker, rechunker

source_shape = (31, 31, 31)
shape = source_shape

sel = (slice(3, 21), slice(11, 27), slice(7, 17))

source_chunk_shape = (5, 2, 4)
target_chunk_shape = (4, 5, 3)
max_mem = 2000 # smaller than the ideal chunk size

dtype = np.dtype('int32')
```

### Preprocessing tools
We have defined our target_chunk_shape above, but rechunkit has a function to guess a good chunk shape given a user-defined amount of memory per chunk:

```python
new_chunk_shape = guess_chunk_shape(source_shape, dtype.itemsize, 400)
```

Chunks will be assigned to the highest composite number within the target_chunk_size. Using composite numbers will benefit the rehunking process as there is a very high likelihood that the least common multiple (LCM) of two composite numbers will be significantly lower than the product of those two numbers. The LCM is used to determine the ideal chunk size for the rechunking process.

Speaking of the ideal chunk size, we can determine the ideal chunk shape and size via a couple functions:

```python
ideal_read_chunk_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape) # (20, 10, 12)

ideal_read_chunk_size = calc_ideal_read_chunk_mem(ideal_read_chunk_shape, dtype.itemsize) # 9600 bytes
```

If the ideal_read_chunk_size can comfortably fit in your memory, then you should use this value. Using the ideal chunk size will mean that you will only need to read all chunks in the source once. If the chunk size (called max_mem in the functions) is less than the ideal, then some chunks will need to be read multiple times. 

To see how many reads are required if no optimization is performed during rechunking (i.e. every target chunk must iterate over every associated source chunk), you can use the calc_n_reads_simple function and compare it to the total number of chunks in the source:

```python
n_chunks_source = calc_n_chunks(source_shape, source_chunk_shape) # 896
n_chunks_target = calc_n_chunks(source_shape, target_chunk_shape) # 616

n_reads_simple = calc_n_reads_simple(source_shape, source_chunk_shape, target_chunk_shape) # 3952
```

Using the simple brute force method requires one chunk to be read 4.4 times on average.

There's also a function to check the number of reads (and writes) using the optimized algorithm:

```python
n_reads, n_writes = calc_n_reads_rechunker(source_shape, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem) # 2044, 616
```

In this case, we only require one chunk to be read 2.28 times on average. The more max_mem you give to the rechunker, the less reads per chunk is required (to a minium of 1 in the ideal case).


### Rechunking
We need a source dataset to get data from. Rechunkit requires that the source input is a function/method that has a single parameter input of a tuple of slices. The slices contain the start and stop of the chunk to be read in the source. 

For example, we can simply use a numpy array and it's `__getitem__` method as the source:

```python
source_data = np.arange(1, prod(source_shape) + 1, dtype=dtype).reshape(source_shape)
source = source_data.__getitem__
```

And again as a simple example, we can use a numpy array as the target:

```python
target = np.zeros(source_shape, dtype=dtype)
```

We don't necessarily need the target as an array to be filled, because the rechunker function returns a generator that can be iterated over. The generator returns a tuple of slices (representing the target chunk) and the associated numpy array data:

```python
for write_chunk, data in rechunker(source, source_shape, dtype, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem):
        target[write_chunk] = data
    
assert np.all(source(()) == target) # Should pass!
```

#### Subsets of the source
There are many use-cases where you don't want the entire dataset. Rather you want a subset of the dataset, but you also want the subset rechunked. The rechunker function has a `sel` parameter which needs to be a tuple of slices of the number of dimensions.

```python
n_reads, n_writes = calc_n_reads_rechunker(source_shape, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel) # 288, 80

target = np.zeros(source_shape, dtype=dtype)[sel]

for write_chunk, data in rechunker(source, source_shape, dtype, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel):
    target[write_chunk] = data

assert np.all(source(sel) == target) # Should pass!
```


## License

This project is licensed under the terms of the Apache Software License 2.0.
