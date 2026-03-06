# Preprocessing Tools

rechunkit provides several functions to help plan a rechunking operation before running it. These let you estimate chunk shapes, calculate memory requirements, and predict I/O costs.

## Guessing a Chunk Shape

If you don't already have a target chunk shape, `guess_chunk_shape` picks one that fits within a byte budget:

```python
from rechunkit import guess_chunk_shape

shape = (1000, 2000, 500)
itemsize = 4  # float32

chunk_shape = guess_chunk_shape(shape, itemsize, target_chunk_size=2**21)
```

The function assigns each dimension to the largest [highly composite number](../concepts/composite-numbers.md) that keeps the total chunk size within the target. It iterates over dimensions, halving the largest one each round until the budget is met.

!!! tip
    Using composite numbers for chunk dimensions produces smaller LCMs with other chunk shapes, which directly reduces the number of redundant source reads during rechunking.

## Ideal Read Chunk Shape

The ideal intermediate read shape is the element-wise LCM of source and target chunk shapes:

```python
from rechunkit import calc_ideal_read_chunk_shape, calc_ideal_read_chunk_mem

source_chunks = (6, 4)
target_chunks = (4, 6)

ideal_shape = calc_ideal_read_chunk_shape(source_chunks, target_chunks)  # (12, 12)
ideal_mem = calc_ideal_read_chunk_mem(ideal_shape, itemsize=4)           # 576 bytes
```

If you can afford `ideal_mem` bytes of buffer, every source chunk will be read exactly once.

## Memory-Constrained Read Shape

When the ideal shape doesn't fit in memory, `calc_source_read_chunk_shape` computes a reduced read shape that fits within `max_mem`:

```python
from rechunkit import calc_source_read_chunk_shape

read_shape = calc_source_read_chunk_shape(
    source_chunks, target_chunks, itemsize=4, max_mem=200
)
```

The algorithm finds a multiple of the source chunk shape that fits in memory while preserving the aspect ratio of the ideal shape as closely as possible.

## Counting Chunks

`calc_n_chunks` returns the total number of chunks for a given shape and chunk shape:

```python
from rechunkit import calc_n_chunks

n_source = calc_n_chunks((100, 100), (6, 4))   # 425
n_target = calc_n_chunks((100, 100), (4, 6))   # 425
```

## Predicting Read Counts

Two functions let you compare I/O strategies:

### Brute-force reads

`calc_n_reads_simple` counts the reads if every target chunk independently reads all its overlapping source chunks — no optimization:

```python
from rechunkit import calc_n_reads_simple

n_simple = calc_n_reads_simple((31, 31, 31), (5, 2, 4), (4, 5, 3))  # 3952
```

### Optimized reads

`calc_n_reads_rechunker` predicts the reads and writes using the optimized algorithm at a given memory budget:

```python
from rechunkit import calc_n_reads_rechunker

n_reads, n_writes = calc_n_reads_rechunker(
    (31, 31, 31), dtype.itemsize, (5, 2, 4), (4, 5, 3), max_mem=2000
)
# n_reads=2044, n_writes=616
```

More memory means fewer reads — see [How It Works](../concepts/how-it-works.md) for details.

## chunk_range Utility

`chunk_range` is a multi-dimensional equivalent of Python's `range`, yielding tuples of slices:

```python
from rechunkit import chunk_range

for chunk in chunk_range((0, 0), (10, 10), (4, 6)):
    print(chunk)
# (slice(0, 4), slice(0, 6))
# (slice(0, 4), slice(6, 10))
# (slice(4, 8), slice(0, 6))
# (slice(4, 8), slice(6, 10))
# (slice(8, 10), slice(0, 6))
# (slice(8, 10), slice(6, 10))
```

This is useful for iterating over chunk positions when writing to storage backends.
