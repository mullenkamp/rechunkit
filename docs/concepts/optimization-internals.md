# Optimization Internals

This page documents the internal optimization strategies used by `calc_source_read_chunk_shape` and the trade-offs discovered during their development. It is intended as a reference for future contributors considering algorithmic improvements.

## The Buffer Shape Problem

When the ideal LCM buffer doesn't fit in `max_mem`, rechunkit must choose a smaller read buffer shape. This shape must be a multiple of the source chunk shape and fit within the memory budget. The goal is to minimize total source chunk reads.

The choice of buffer shape has a significant impact on read count because of how the constrained path works: it iterates over target chunks, reads a buffer-sized region of source chunks, and serves all target chunks that fit within that region. A poorly chosen buffer shape can cross target chunk boundaries in ways that leave partial coverage, forcing re-reads when adjacent target chunks are processed.

### Why the Greedy Heuristic is Suboptimal

The original approach uses a greedy heuristic (`_greedy_read_chunk_factors`) that:

1. Computes the per-dimension scaling factor from source to ideal: `k_i = lcm_dim / source_dim`
2. Scales all factors down uniformly to fit in memory: `n_i = floor(k_i * scale)`
3. Greedily grows the most compressed dimension until capacity is exhausted
4. Trims factors to snap to target chunk boundaries (`_trim_factors`)

This works well in many cases, but the greedy growth order doesn't account for how the buffer aligns with the target chunk grid. A buffer of shape `(7, 15)` might cross more target boundaries than `(7, 9)` in a particular dimension, causing more read groups and redundant source reads.

Benchmarking across ~60 constrained configurations showed the greedy heuristic was suboptimal in **13% of cases**, producing 1-4% more reads than the optimal buffer shape.

## Current Optimization Strategy

The current implementation uses a **candidate search** when the `shape` parameter is provided (which `rechunker()` and `calc_n_reads_rechunker()` do automatically):

### Candidate Generation

**Small search spaces** (`prod(k_factors) <= 500`): Exhaustively enumerate all valid factor combinations. For typical 2D arrays, this means 5-25 combinations. For 3D arrays with moderate LCM factors, up to a few hundred.

**Large search spaces** (`prod(k_factors) > 500`): Generate targeted heuristic candidates:

- The greedy result (always included as baseline)
- Per-dimension rounding to target chunk multiples
- Per-dimension rounding to LCM factor multiples
- Perturbations of +/-1-2 per dimension around the greedy factors
- Pairwise capacity shifts between dimensions (shrink one, grow another)

This typically produces 10-30 unique candidate shapes after deduplication via `_trim_factors`.

### Scoring

Each candidate is scored by `_count_reads()`, which mirrors the `_rechunk_plan` constrained-path logic exactly — iterating over all target chunks, simulating the bulk/single grouping, and counting source chunk reads. This is the most expensive part of the optimization.

When there are more than 8 candidates, a fast analytical pre-filter (`n_read_groups * source_chunks_per_group`) narrows the set before expensive scoring. The greedy shape is always included in the scored set to guarantee no regression.

### Overhead

The optimization adds zero overhead for ideal-path cases (LCM fits in memory) and early-exit cases (source chunk >= max_mem). For constrained cases, the overhead is dominated by `_count_reads` calls:

| Case | Typical overhead | Read improvement |
|------|-----------------|-----------------|
| Ideal path or early exit | 0 | N/A |
| 2D constrained (3-7 candidates) | 2-7 ms | 1-2% fewer reads |
| 3D constrained (8-31 candidates) | 50-62 ms | 0-3% fewer reads |

For real-world rechunking operations (reading chunks from disk or network at ~1 ms+ each, with hundreds or thousands of chunks), this one-time planning overhead is negligible compared to I/O time.

## Alternative Approaches Evaluated

Several approaches were tried and rejected during development. They are documented here to avoid repeating the same explorations.

### Fast Analytical Estimate Only

A simple O(ndim) analytical score was tested as a replacement for `_count_reads`:

```python
score = n_read_groups * source_chunks_per_group
```

where `n_read_groups = prod(ceil(shape[d] / buffer[d]))` and `source_chunks_per_group = prod(buffer[d] // source[d])`.

**Result:** The estimate does not correlate well with actual read counts. It picked worse shapes than the greedy heuristic in 2 out of 7 test cases. The fundamental problem is that it doesn't account for the write-chunk deduplication logic in the constrained path — the `written_chunks` set that avoids redundant reads when a buffer covers multiple target chunks. Two buffer shapes with the same analytical score can have very different actual read counts depending on how they align with the target chunk grid.

**Verdict:** Useful as a pre-filter to narrow candidates, but not reliable enough as the sole scoring metric.

### Hybrid: Fast Estimate + Verify Against Greedy

Tried using the fast estimate to pick the best candidate, then comparing it against the greedy shape using `_count_reads` (at most 2 expensive calls).

**Result:** When the fast estimate picks the wrong winner, the verification just confirms greedy is better, and the optimization produces zero improvement. The approach correctly avoids regressions but also avoids improvements.

**Verdict:** Safe but ineffective — the fast estimate's top-1 pick is often not the actual best candidate.

### Top-N Pre-filter Without Greedy Guarantee

Used the fast estimate to pick the top-5 candidates, then scored them with `_count_reads`.

**Result:** In some 3D cases, the greedy shape ranked outside the top-5 by fast estimate. Since none of the top-5 were actually better than greedy, the optimization picked a shape with significantly more reads (e.g., 2044 -> 3664 in one case — a 79% regression).

**Verdict:** Dangerous without always including the greedy shape in the scored set. The current implementation fixes this by ensuring greedy_shape is always scored.

### Rewriting _count_reads with Pure Arithmetic

Attempted to replace the generator-based `_count_reads` with direct index arithmetic to avoid `chunk_range` overhead (tuple/slice creation, `itertools.product` generators).

**Result:** The rewritten version had subtle bugs — it didn't exactly mirror the `_rechunk_plan` bulk-grouping logic (off-by-one in which write chunks are considered "covered"). One test case showed 367 reads vs the correct 374. The constrained path's write-chunk coverage logic has several interacting conditions (`include_partial_chunks`, `clip_ends`, the `written_chunks` set), making it fragile to reimplement.

**Verdict:** Not worth the risk. The original `_count_reads` that mirrors `_rechunk_plan` exactly is the only safe implementation. Performance improvements should focus on reducing the number of calls, not the per-call cost.

## Future Improvement Opportunities

### Monotonicity Guarantee

The current implementation maintains monotonicity (more memory = fewer or equal reads) empirically across all tested configurations. However, there is no formal proof. The exhaustive search for small factor spaces (total_k <= 500) guarantees optimality within that space. For large factor spaces, the heuristic candidates may miss the true optimum.

A formal monotonicity proof or a counterexample would be valuable.

### Reducing _count_reads Cost

`_count_reads` is O(n_write_chunks) per call, where `n_write_chunks = prod(ceil(shape[d] / target[d]))`. For large 3D arrays, this can be hundreds or thousands of iterations per candidate. Possible approaches:

- **Analytical formula for bulk grouping:** If the number of write chunks covered per read group could be computed analytically (without iterating), scoring would be O(n_read_groups) instead of O(n_write_chunks). The challenge is that coverage depends on alignment between the source grid, target grid, and buffer boundaries — which varies per read group.
- **Caching across candidates:** Many candidates share the same structure in most dimensions. Partial results from one candidate could inform scoring of another.
- **NumPy vectorization:** The inner loop is pure Python with set operations. A NumPy-based implementation using array indexing could be significantly faster, but the set-based deduplication logic is hard to vectorize.

### Better Candidate Generation for Large Factor Spaces

The heuristic candidates for `total_k > 500` are based on perturbations of the greedy shape. If the optimal shape is structurally very different from greedy (e.g., [2, 5, 3] vs greedy [4, 4, 3]), it may not be in the candidate set. More sophisticated generation strategies could help:

- **Target-aligned search:** For each dimension, try all multiples of the target chunk size that fit, rather than perturbations of greedy.
- **LCM-sub-multiple search:** Try shapes that are divisors of the LCM shape, as these align cleanly with both source and target grids.
- **Adaptive expansion:** If the best candidate is at the edge of the search neighborhood, expand the search in that direction.

### The Underlying Assumption

The entire optimization assumes that minimizing source reads is the primary objective. In practice, other factors matter:

- **Read size:** Reading 10 large source chunks may be faster than reading 8 small ones if I/O has high per-request overhead.
- **Spatial locality:** Buffer shapes that read contiguous memory regions may benefit from OS-level caching and prefetching.
- **Write ordering:** The canonical yield order may interact with downstream write patterns in ways that affect performance.

These factors are currently not modeled. A more complete optimization would consider total I/O cost rather than raw read count.
