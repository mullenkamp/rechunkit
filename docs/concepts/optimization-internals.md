# Optimization Internals

This page documents optimization strategies that were explored for `calc_source_read_chunk_shape` and ultimately rejected. It is intended as a reference for future contributors considering algorithmic improvements, so the same explorations are not repeated.

## The Buffer Shape Problem

When the ideal LCM buffer doesn't fit in `max_mem`, rechunkit must choose a smaller read buffer shape. This shape must be a multiple of the source chunk shape and fit within the memory budget. The goal is to minimize total source chunk reads.

The choice of buffer shape has a significant impact on read count because of how the constrained path works: it iterates over target chunks, reads a buffer-sized region of source chunks, and serves all target chunks that fit within that region. A poorly chosen buffer shape can cross target chunk boundaries in ways that leave partial coverage, forcing re-reads when adjacent target chunks are processed.

## Current Approach: Greedy Heuristic

The current implementation uses a greedy heuristic that:

1. Computes the per-dimension scaling factor from source to ideal: `k_i = lcm_dim / source_dim`
2. Scales all factors down uniformly to fit in memory: `n_i = floor(k_i * scale)`
3. Greedily grows the most compressed dimension until capacity is exhausted

This works well in most cases. Benchmarking across ~60 constrained configurations showed the greedy heuristic was suboptimal in **13% of cases**, producing 1-4% more reads than the optimal buffer shape. These marginal losses were judged acceptable given the complexity required to do better (see below).

## Candidate Search (Explored and Rejected)

A candidate search optimization was fully implemented and benchmarked. It was reverted because the added complexity (~150 lines of code, 50-62 ms overhead for 3D cases) was not justified by the marginal gains (1-4% fewer reads in 13% of constrained cases).

### How It Worked

When the array `shape` was provided, `calc_source_read_chunk_shape` would:

**Small search spaces** (`prod(k_factors) <= 500`): Exhaustively enumerate all valid factor combinations. For typical 2D arrays, this means 5-25 combinations. For 3D arrays with moderate LCM factors, up to a few hundred.

**Large search spaces** (`prod(k_factors) > 500`): Generate targeted heuristic candidates:

- The greedy result (always included as baseline)
- Per-dimension rounding to target chunk multiples
- Per-dimension rounding to LCM factor multiples
- Perturbations of +/-1-2 per dimension around the greedy factors
- Pairwise capacity shifts between dimensions (shrink one, grow another)

Each candidate was scored by `_count_reads()`, which mirrored the `_rechunk_plan` constrained-path logic exactly — iterating over all target chunks, simulating the bulk/single grouping, and counting source chunk reads. When there were more than 8 candidates, a fast analytical pre-filter narrowed the set before expensive scoring. The greedy shape was always included in the scored set to guarantee no regression.

### Why It Was Rejected

1. **Marginal gains:** Only 13% of constrained configurations benefited, and only by 1-4% fewer reads.
2. **Non-trivial overhead:** Scoring required `_count_reads` calls at O(n_write_chunks) each. For 3D arrays, this added 50-62 ms of planning time.
3. **Fragile mirror code:** `_count_reads` had to exactly mirror `_rechunk_plan`'s constrained-path logic (including `include_partial_chunks`, `clip_ends`, and the `written_chunks` set). Any divergence caused incorrect scores. Maintaining this mirror across future changes to `_rechunk_plan` would be error-prone.
4. **~150 lines of added complexity** for `_count_reads`, `_trim_factors`, `_greedy_read_chunk_factors`, candidate generation, pre-filtering, and scoring logic.

For real-world rechunking (reading chunks from disk or network at ~1 ms+ each), a 1-4% reduction in reads is typically a few saved I/O operations — dwarfed by the inherent variability of I/O latency.

### Overhead Measurements

| Case | Typical overhead | Read improvement |
|------|-----------------|-----------------|
| Ideal path or early exit | 0 | N/A |
| 2D constrained (3-7 candidates) | 2-7 ms | 1-2% fewer reads |
| 3D constrained (8-31 candidates) | 50-62 ms | 0-3% fewer reads |

## Other Approaches Evaluated

Several alternative approaches were also tried and rejected. They are documented here to avoid repeating the same explorations.

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

**Verdict:** Dangerous without always including the greedy shape in the scored set.

### Rewriting _count_reads with Pure Arithmetic

Attempted to replace the generator-based `_count_reads` with direct index arithmetic to avoid `chunk_range` overhead (tuple/slice creation, `itertools.product` generators).

**Result:** The rewritten version had subtle bugs — it didn't exactly mirror the `_rechunk_plan` bulk-grouping logic (off-by-one in which write chunks are considered "covered"). One test case showed 367 reads vs the correct 374. The constrained path's write-chunk coverage logic has several interacting conditions (`include_partial_chunks`, `clip_ends`, the `written_chunks` set), making it fragile to reimplement.

**Verdict:** Not worth the risk. A `_count_reads` that mirrors `_rechunk_plan` exactly is the only safe implementation. Performance improvements should focus on reducing the number of calls, not the per-call cost.

## Future Improvement Opportunities

If revisiting this optimization in the future, the key challenges to address are:

### Reducing Scoring Cost

`_count_reads` is O(n_write_chunks) per call. For large 3D arrays, this can be hundreds or thousands of iterations per candidate. Possible approaches:

- **Analytical formula for bulk grouping:** If the number of write chunks covered per read group could be computed analytically (without iterating), scoring would be O(n_read_groups) instead of O(n_write_chunks). The challenge is that coverage depends on alignment between the source grid, target grid, and buffer boundaries — which varies per read group.
- **Caching across candidates:** Many candidates share the same structure in most dimensions. Partial results from one candidate could inform scoring of another.
- **NumPy vectorization:** The inner loop is pure Python with set operations. A NumPy-based implementation using array indexing could be significantly faster, but the set-based deduplication logic is hard to vectorize.

### Better Candidate Generation for Large Factor Spaces

The heuristic candidates for large search spaces are based on perturbations of the greedy shape. If the optimal shape is structurally very different from greedy (e.g., [2, 5, 3] vs greedy [4, 4, 3]), it may not be in the candidate set. More sophisticated generation strategies could help:

- **Target-aligned search:** For each dimension, try all multiples of the target chunk size that fit, rather than perturbations of greedy.
- **LCM-sub-multiple search:** Try shapes that are divisors of the LCM shape, as these align cleanly with both source and target grids.
- **Adaptive expansion:** If the best candidate is at the edge of the search neighborhood, expand the search in that direction.

### Monotonicity

Any optimization must guarantee monotonicity (more memory = fewer or equal reads). The greedy heuristic currently maintains this empirically. The candidate search maintained it by always including the greedy shape as a fallback. A formal monotonicity proof would be valuable regardless of whether an optimization is added.

### The Underlying Assumption

The entire optimization assumes that minimizing source reads is the primary objective. In practice, other factors matter:

- **Read size:** Reading 10 large source chunks may be faster than reading 8 small ones if I/O has high per-request overhead.
- **Spatial locality:** Buffer shapes that read contiguous memory regions may benefit from OS-level caching and prefetching.
- **Write ordering:** The canonical yield order may interact with downstream write patterns in ways that affect performance.

These factors are currently not modeled. A more complete optimization would consider total I/O cost rather than raw read count.
