# API Reference

All public functions are exported from the top-level `rechunkit` package.

```python
import rechunkit
```

## Chunk Shape Estimation

| Function | Description |
|----------|-------------|
| [`guess_chunk_shape`](rechunkit.md#rechunkit.guess_chunk_shape) | Estimate an appropriate chunk shape using composite numbers |

## Read Planning

| Function | Description |
|----------|-------------|
| [`calc_ideal_read_chunk_shape`](rechunkit.md#rechunkit.calc_ideal_read_chunk_shape) | LCM of source and target chunk shapes (ideal buffer shape) |
| [`calc_ideal_read_chunk_mem`](rechunkit.md#rechunkit.calc_ideal_read_chunk_mem) | Memory required for the ideal read buffer |
| [`calc_source_read_chunk_shape`](rechunkit.md#rechunkit.calc_source_read_chunk_shape) | Reduced read shape that fits within a memory budget |

## I/O Cost Estimation

| Function | Description |
|----------|-------------|
| [`calc_n_chunks`](rechunkit.md#rechunkit.calc_n_chunks) | Total number of chunks for a given shape and chunk shape |
| [`calc_n_reads_simple`](rechunkit.md#rechunkit.calc_n_reads_simple) | Brute-force read count (no optimization) |
| [`calc_n_reads_rechunker`](rechunkit.md#rechunkit.calc_n_reads_rechunker) | Optimized read and write count for a given memory budget |

## Core

| Function | Description |
|----------|-------------|
| [`rechunker`](rechunkit.md#rechunkit.rechunker) | Generator that rechunks from source to target chunk shape |
| [`chunk_range`](rechunkit.md#rechunkit.chunk_range) | Multi-dimensional range yielding tuples of slices |
