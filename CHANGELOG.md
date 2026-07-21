# Changelog

## 0.6.0 (2026-07-21)

### Performance
- **Honest memory budgeting** (`calc_source_read_chunk_shape`): the planner now budgets the TRUE buffer allocation — per-dim `max(read + phase, target)` plus the canonical-order pending band — instead of just the read region's cell count. New optional `shape` and `phase` parameters (backward compatible). When even one source chunk busts the budget, memory-free growth up to the target-forced buffer dims is taken (measured: the constrained benchmark case became 79% faster at identical peak memory).
- **Ideal read shapes are clipped to the array extent** (`calc_ideal_read_chunk_shape`, new optional `shape` parameter): per dim, the LCM never exceeds the smallest source-aligned cover of the extent. Non-divisor full-field targets (the groupby pattern) previously inflated the "ideal" past any usable size (a 193 GiB ideal on a 3 GB array), forcing the worst constrained path. Measured: 9x -> 1x read amplification at a 1 MB budget on the benchmark analog.
- **Batched single path**: write chunks whose read region exceeds the bulk buffer are batched in C-order with deduplicated reads (new `'multi'` plan group replacing `'single'`), bounded by the memory budget. Measured: 9x -> 2.1x read amplification at a tight budget; the bulk buffer is now allocated lazily so batched-only plans never create it.
- **Pending-aware planning**: the canonical-order reorder buffer (previously unbounded by `max_mem` — measured up to 271x over budget on wide arrays) is now counted toward the budget, with the read shape shrunk when that strictly reduces the total. The irreducible wide-array residual (a single source chunk spanning multiple target-chunk rows) is documented with its formula instead of being silently ignored.
- **Read-shape candidate check**: for constrained plans under 20,000 target chunks, up to 5 candidate read shapes are scored by running the planner itself and the fewest-reads shape wins (never worse than the greedy choice, which stays in the set). Fixes non-monotone read counts (measured worst case pre-fix: doubling the budget increased reads 11.5x). Planning cost is a few extra passes in that regime only. See `docs/concepts/optimization-internals.md` for the revisit rationale.

### Fixed
- `guess_chunk_shape` now raises `ValueError` when any shape dim is <= 0 (previously a zero-length dim passed straight through, producing an invalid chunk shape with a zero dim that breaks any downstream chunk arithmetic). The empty shape `()` still returns `()`.
- `guess_chunk_shape` now accepts numpy integer shape values (`np.int64` etc.) instead of raising `TypeError`; returned chunk dims are always plain Python ints.

### Documentation
- `rechunker()` docstring and the how-it-works concept page now document the yield-lifetime contract: yielded arrays may be views into the internal buffer, must be consumed or copied before advancing the generator, and must be treated as read-only.
- Corrected two previously false claims: `max_mem` now genuinely bounds the allocation (with the documented floors/residual), and the memory-vs-reads relationship is documented as *generally* decreasing rather than guaranteed monotone.

## 0.5.1

Prior releases have no changelog; see the git history.
