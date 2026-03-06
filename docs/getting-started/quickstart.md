# Quick Start

This walkthrough covers the core rechunkit workflow: estimating chunk shapes, predicting I/O cost, and rechunking an array.

## Setup

```python
import numpy as np
from math import prod
from rechunkit import (
    guess_chunk_shape,
    calc_ideal_read_chunk_shape,
    calc_ideal_read_chunk_mem,
    calc_n_reads_rechunker,
    rechunker,
)

shape = (31, 31, 31)
dtype = np.dtype('int32')

source_chunk_shape = (5, 2, 4)
target_chunk_shape = (4, 5, 3)
max_mem = 2000  # bytes
```

## 1. Estimate a Chunk Shape

If you don't have a target chunk shape in mind, `guess_chunk_shape` picks one using composite numbers:

```python
target_chunk_shape = guess_chunk_shape(shape, dtype.itemsize, target_chunk_size=400)
```

## 2. Check Memory Requirements

The ideal read chunk shape is the LCM of source and target. If it fits in memory, every source chunk is read exactly once:

```python
ideal_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)
ideal_mem = calc_ideal_read_chunk_mem(ideal_shape, dtype.itemsize)

print(f"Ideal read shape: {ideal_shape}")
print(f"Ideal memory: {ideal_mem} bytes")
```

## 3. Predict Read Counts

Compare the optimized read count against your memory budget:

```python
n_reads, n_writes = calc_n_reads_rechunker(
    shape, dtype.itemsize, source_chunk_shape, target_chunk_shape, max_mem
)
print(f"Reads: {n_reads}, Writes: {n_writes}")
```

## 4. Rechunk

The `rechunker` function takes a callable source and yields `(slices, data)` tuples:

```python
source_data = np.arange(1, prod(shape) + 1, dtype=dtype).reshape(shape)
source = source_data.__getitem__

target = np.zeros(shape, dtype=dtype)
for write_chunk, data in rechunker(
    source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem
):
    target[write_chunk] = data

assert np.all(source_data == target)
```

Target chunks are always yielded in canonical order (C-order iteration of the target chunk grid), regardless of the source chunk layout.

## Next Steps

- [Preprocessing Tools](../guide/preprocessing.md) — all the planning and estimation functions
- [Rechunking Guide](../guide/rechunking.md) — selections, memory tuning, yield order
- [Integration](../guide/integration.md) — using rechunkit with h5py, zarr, and other storage backends
