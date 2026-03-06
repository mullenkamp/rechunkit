# Rechunking

The `rechunker` function is the core of the library. It takes a source callable and yields target chunks as `(slices, data)` tuples via a Python generator.

## Basic Usage

The source must be a callable that accepts a tuple of slices and returns a numpy array — for example, a numpy array's `__getitem__` method:

```python
import numpy as np
from math import prod
from rechunkit import rechunker

shape = (31, 31, 31)
dtype = np.dtype('int32')
source_chunk_shape = (5, 2, 4)
target_chunk_shape = (4, 5, 3)
max_mem = 2000

source_data = np.arange(1, prod(shape) + 1, dtype=dtype).reshape(shape)
source = source_data.__getitem__

target = np.zeros(shape, dtype=dtype)
for write_chunk, data in rechunker(
    source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem
):
    target[write_chunk] = data

assert np.all(source_data == target)
```

## Subset Selection

The `sel` parameter lets you rechunk a subset of the source without reading the entire array:

```python
sel = (slice(3, 21), slice(11, 27), slice(7, 17))
target_shape = tuple(s.stop - s.start for s in sel)

target = np.zeros(target_shape, dtype=dtype)
for write_chunk, data in rechunker(
    source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel=sel
):
    target[write_chunk] = data

assert np.all(source_data[sel] == target)
```

The yielded slices are relative to the selection, not the original array. Selection slices must have a step of 1 (or `None`).

## Yield Order

Target chunks are always yielded in **canonical order** — the same order as iterating `chunk_range((0, ...), target_shape, target_chunk_shape)`. This is C-order (row-major) iteration of the target chunk grid.

The yield order depends only on `shape` (or the selection shape) and `target_chunk_shape`. It does not depend on `source_chunk_shape` or `max_mem`. This makes the output predictable regardless of the source layout or memory budget.

## Tuning max_mem

The `max_mem` parameter controls how much memory (in bytes) the internal buffer can use:

| Scenario | Effect |
|----------|--------|
| `max_mem` ≥ ideal read chunk memory | Each source chunk is read exactly once (optimal) |
| `max_mem` < ideal but > source chunk | Some source chunks are read multiple times |
| `max_mem` ≈ source chunk size | Falls back to per-target-chunk reads (most redundant) |

Use `calc_ideal_read_chunk_mem` to find the ideal value, then decide based on your available memory. More memory always means fewer or equal reads — never more.

!!! tip
    You can use `calc_n_reads_rechunker` to compare read counts at different `max_mem` values before committing to a budget.

## StringDType Support

For numpy `StringDType` arrays, the `itemsize` cannot be inferred from the dtype. Pass it explicitly:

```python
for write_chunk, data in rechunker(
    source, shape, dtype, src_chunks, tgt_chunks, max_mem, itemsize=64
):
    target[write_chunk] = data
```
