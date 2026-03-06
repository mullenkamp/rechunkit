# Composite Numbers

Rechunking efficiency depends on the Least Common Multiple (LCM) of source and target chunk dimensions. rechunkit uses **highly composite numbers** to keep LCMs small.

## Why LCM Matters

The ideal read buffer for rechunking is shaped by the element-wise LCM of the source and target chunk shapes. A smaller LCM means a smaller buffer, which means less memory needed for the optimal (one-read-per-chunk) path.

Consider two 1D examples:

| Source | Target | LCM | Buffer cells |
|--------|--------|-----|-------------|
| 12 | 8 | 24 | 24 |
| 17 | 19 | 323 | 323 |

With primes (17 and 19), the LCM equals their product — the worst case. With composite numbers (12 and 8), the LCM is much smaller because they share common factors.

## Highly Composite Numbers

A highly composite number has more divisors than any smaller positive integer. rechunkit uses a pre-computed table of these numbers:

```
1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840,
1260, 1680, 2520, 5040, 7560, 10080, ...
```

Because these numbers are divisible by many small factors, the LCM of any two highly composite numbers is very likely to be much smaller than their product.

## How guess_chunk_shape Uses Them

`guess_chunk_shape` iterates over dimensions, shrinking each to the largest highly composite number that fits within the byte budget. The algorithm:

1. Start with the full array shape
2. While the chunk exceeds the target size:
    - Take the current largest dimension
    - Replace it with the largest composite number ≤ half its value
3. Return the resulting shape

This produces chunk shapes that are efficient for rechunking against almost any other chunk shape — the LCMs will be small, and the ideal read buffer will fit in less memory.

## Worked Example

Suppose you have a `(1000, 2000)` array with `float32` (4 bytes) and a 4 KB target chunk size:

```python
from rechunkit import guess_chunk_shape, calc_ideal_read_chunk_shape

chunk_a = guess_chunk_shape((1000, 2000), 4, 4096)  # e.g. (24, 36)
chunk_b = guess_chunk_shape((1000, 2000), 4, 8192)  # e.g. (48, 36)

lcm_shape = calc_ideal_read_chunk_shape(chunk_a, chunk_b)  # (48, 36)
```

Because both chunk shapes use composite numbers, their LCM is just `(48, 36)` — only 6912 bytes. With arbitrary prime-based chunks, the LCM could easily be 10–100x larger.
