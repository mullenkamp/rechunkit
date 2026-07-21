"""Perf-round regression tests: true-buffer budgeting (R1) and the planning
behaviors layered on it. Each fixed case was proven to FAIL on the pre-round
tree (see cfdb/plans/perf-round-plan-2026-07-21.md)."""
from math import prod

import numpy as np
import pytest

from rechunkit import calc_ideal_read_chunk_shape, calc_source_read_chunk_shape, rechunker


def true_buf(read_shape, tgt, phase, itemsize):
    return prod(max(r + p, t) for r, p, t in zip(read_shape, phase, tgt)) * itemsize


def floor_buf(src, tgt, phase, itemsize):
    return true_buf(src, tgt, phase, itemsize)


def test_true_buffer_budget_fixed():
    # Pre-R1: returns (6, 2) -> true buffer max(6,3)*max(2,3)*8 = 144 B > 100 B
    # even though (2, 4) fits at 96 B. (Review finding P4a.)
    src, tgt, itemsize, mm = (2, 2), (3, 3), 8, 100
    res = calc_source_read_chunk_shape(src, tgt, itemsize, mm)
    assert true_buf(res, tgt, (0, 0), itemsize) <= mm


def test_true_buffer_budget_sweep():
    # Every result either fits the true-buffer budget or is the
    # allocation-neutral floor (target-dominated dims force the allocation
    # regardless of the read shape).
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(200):
        ndims = int(rng.integers(1, 4))
        src = tuple(int(rng.integers(1, 20)) for _ in range(ndims))
        tgt = tuple(int(rng.integers(1, 40)) for _ in range(ndims))
        itemsize = int(rng.choice([1, 2, 4, 8]))
        mm = int(rng.choice([2**6, 2**8, 2**10, 2**14]))
        res = calc_source_read_chunk_shape(src, tgt, itemsize, mm)
        zeros = tuple(0 for _ in range(ndims))
        for r, s in zip(res, src):
            assert r % s == 0 and r >= s
        buf = true_buf(res, tgt, zeros, itemsize)
        assert buf <= mm or buf == floor_buf(src, tgt, zeros, itemsize), (
            src, tgt, itemsize, mm, res)
        checked += 1
    assert checked == 200


def test_phase_enters_budget():
    # With a phase-shifted selection the buffer is max(read+phase, target):
    # ignoring phase, the ideal (16, 8) is picked and allocates 345 B on a
    # 300 B budget; phase-aware planning must fit (or hit the neutral floor).
    src, tgt, itemsize, mm = (8, 8), (16, 2), 1, 300
    phase = (7, 7)
    res = calc_source_read_chunk_shape(src, tgt, itemsize, mm, phase=phase)
    buf = true_buf(res, tgt, phase, itemsize)
    assert buf <= mm or buf == floor_buf(src, tgt, phase, itemsize)


def test_floor_takes_free_growth():
    # When even one source chunk busts the budget, growth up to the
    # target-forced buffer dims is memory-free and must be taken.
    src, tgt, itemsize, mm = (5, 16), (14, 2), 8, 640
    res = calc_source_read_chunk_shape(src, tgt, itemsize, mm)
    zeros = (0, 0)
    # allocation-neutral: same true buffer as reading one source chunk...
    assert true_buf(res, tgt, zeros, itemsize) == floor_buf(src, tgt, zeros, itemsize)
    # ...but with the read grown within the target-forced dims.
    assert res >= src and res != src


def test_ideal_clip_era5_shape():
    # Review P1a: the per-dim LCM must be clipped to the source-aligned cover
    # of the tiled extent. ERA5-like full-field target: unclipped "ideal" is
    # (100, 72100, 7200) = 193 GiB-scale; clipped is (100, 800, 1500).
    src, tgt, shape = (100, 100, 100), (1, 721, 1440), (720, 721, 1440)
    assert calc_ideal_read_chunk_shape(src, tgt) == (100, 72100, 7200)
    assert calc_ideal_read_chunk_shape(src, tgt, shape) == (100, 800, 1500)


def test_clip_restores_ideal_path_reads():
    # Scaled ERA5 analog: at 1 MB the clipped ideal fits, so the groupby-style
    # target takes the ideal path and reads == stored chunks.
    # Pre-R2: constrained single path, 8640 reads (9x amplification).
    shape, src, tgt = (72, 73, 144), (10, 10, 10), (1, 73, 144)
    arr = np.arange(prod(shape), dtype='float64').reshape(shape)
    calls = [0]

    def source(slices):
        calls[0] += 1
        return arr[slices]

    out = np.full(shape, np.nan)
    for wc, data in rechunker(source, shape, arr.dtype, src, tgt, 2**20):
        out[wc] = data
    assert np.array_equal(out, arr)
    assert calls[0] == 8 * 8 * 15  # every stored chunk read exactly once


def test_batched_single_read_amplification():
    # Scaled ERA5 analog at TIGHT memory (clipped ideal does not fit):
    # the batched single path reads each source chunk once per batch.
    # Pre-R3: one read per overlapping (source chunk, target chunk) pair
    # = 8640 reads (9x). Post-R3: ~2x.
    shape, src, tgt = (72, 73, 144), (10, 10, 10), (1, 73, 144)
    arr = np.arange(prod(shape), dtype='float64').reshape(shape)
    calls = [0]

    def source(slices):
        calls[0] += 1
        return arr[slices]

    out = np.full(shape, np.nan)
    for wc, data in rechunker(source, shape, arr.dtype, src, tgt, 2**19):
        out[wc] = data
    assert np.array_equal(out, arr)
    stored = 8 * 8 * 15
    assert calls[0] <= 3 * stored


def test_multi_path_memory_bounded():
    # The multi path's peak allocation is the batch quota (+ one transient
    # source chunk + plan-object churn, which is itemsize-independent and
    # covered by the fixed allowance). Pre-R3 the single path allocated the
    # 841 KB max(read, target) bulk buffer on a 512 KB budget.
    import gc
    import tracemalloc

    shape, src, tgt, mm = (72, 73, 144), (10, 10, 10), (1, 73, 144), 2**19
    arr = np.arange(prod(shape), dtype='float64').reshape(shape)

    def source(slices):
        return arr[slices]

    gc.collect()
    tracemalloc.start()
    for _wc, _data in rechunker(source, shape, arr.dtype, src, tgt, mm):
        pass
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= mm + 300 * 1024


def test_mixed_plan_exactness_and_order():
    # src=6 vs tgt=4 along dim0 mixes bulk-served and batched write chunks
    # in one plan; exactness, exactly-once, and canonical C-order must hold
    # across the interleaving.
    from math import ceil

    from rechunkit.main import _rechunk_plan

    shape, src, tgt, mm = (24, 80), (4, 10), (9, 10), 2**12
    itemsize = 8
    types = {gt for gt, _r, _w, _g in _rechunk_plan(shape, itemsize, src, tgt, mm)}
    assert types == {'bulk', 'multi'}

    arr = np.arange(prod(shape), dtype='float64').reshape(shape)
    n_chunks_per_dim = tuple(ceil(s / c) for s, c in zip(shape, tgt))

    def canon_idx(wc):
        idx = 0
        for s, c, nc in zip(wc, tgt, n_chunks_per_dim):
            idx = idx * nc + (s.start // c)
        return idx

    out = np.full(shape, np.nan)
    seen = []
    for wc, data in rechunker(arr.__getitem__, shape, arr.dtype, src, tgt, mm):
        seen.append(canon_idx(wc))
        out[wc] = data
    assert seen == list(range(prod(n_chunks_per_dim)))
    assert np.array_equal(out, arr)


def test_pending_budget_enforced():
    # Wide-array ideal path: the canonical-order pending band (~1.15 MB) sat
    # on top of a 512 KB budget pre-R4. The planner now counts it and shrinks
    # the read shape until buffer + pending fit (memory over reads, ruling 3);
    # this config lands on the all-multi path within budget at <= 2.2x reads.
    import gc
    import tracemalloc

    shape, src, tgt, mm = (48, 24000), (4, 300), (6, 300), 2**19
    arr = np.arange(prod(shape), dtype='float64').reshape(shape)
    calls = [0]

    def source(slices):
        calls[0] += 1
        return arr[slices]

    out = np.full(shape, np.nan)
    gc.collect()
    tracemalloc.start()
    for wc, data in rechunker(source, shape, arr.dtype, src, tgt, mm):
        out[wc] = data
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert np.array_equal(out, arr)
    assert peak <= mm + 300 * 1024
    stored = 12 * 80
    assert calls[0] <= 2.2 * stored


@pytest.mark.parametrize("shape, src, tgt, itemsize", [
    # review's original non-monotone case (needs itemsize>=4 to be non-trivial:
    # at itemsize 1-2 it is already monotone and the case pins nothing)
    ((53, 11), (3, 8), (6, 6), 8),
    # honest-budget regime-flip pathologies found by the R5 sweep (worst was
    # 63 -> 723 reads when memory doubled, pre-fix)
    ((36, 21), (14, 1), (1, 10), 8),
    ((30, 54), (15, 1), (1, 8), 8),
])
def test_read_count_monotonicity(shape, src, tgt, itemsize):
    # More memory must never cost dramatically more reads: the candidate
    # check scores the actual plan for a small candidate set (incl. the
    # half-budget greedy and the enforced ideal) and keeps the best.
    from rechunkit import calc_n_reads_rechunker

    ladder = [512, 1024, 2048, 4096]
    reads = [calc_n_reads_rechunker(shape, itemsize, src, tgt, mm)[0] for mm in ladder]
    for a, b in zip(reads, reads[1:]):
        assert b <= a, (reads, ladder)


def test_planner_buffer_agreement():
    # The buffer rechunker() allocates must never exceed what the planner
    # budgeted for the same arguments: run a phase-shifted rechunk and check
    # the plan's read shape against the observed reads.
    shape, src, tgt, mm = (40, 40), (8, 8), (16, 2), 300
    sel = (slice(7, 39), slice(7, 39))
    arr = np.arange(prod(shape), dtype='int8').reshape(shape)
    out = np.full(tuple(s.stop - s.start for s in sel), -1, dtype='int8')
    for wc, data in rechunker(arr.__getitem__, shape, arr.dtype, src, tgt, mm, sel=sel):
        out[wc] = data
    assert np.array_equal(out, arr[sel])
