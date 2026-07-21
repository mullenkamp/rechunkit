"""Rechunkit performance benchmark suite.

Measures rechunker() throughput and peak memory on fixed, seeded configurations
covering the three planning regimes (ideal, constrained, phase-shifted sel),
plus plan-only and guess_chunk_shape timings.

Deterministic gate cases (counts and allocation ratios, not wall time — these
are the primary perf-round gates because they are noise-free):
- groupby_amplification: non-divisor (1, full, full) target; source-call count
  vs stored-chunk count (read/decompression amplification).
- pending_wide: wide-array canonical-order reorder buffer; tracemalloc peak vs
  max_mem ratio.
- mixed_plan_peak: a plan containing BOTH bulk and single groups; tracemalloc
  peak vs max_mem ratio (where bulk pending and single/batch memory can stack).
- plan_only_small_constrained: planning wall time on a small constrained case
  (the regime where planner additions actually run).

Usage:
    uv run python benchmarks/bench_rechunk.py --output benchmarks/results/baseline-YYYY-MM-DD.json
    uv run python benchmarks/bench_rechunk.py --compare benchmarks/results/baseline.json benchmarks/results/after.json

Protocol: median of N_RUNS (default 9) wall-clock runs per case, fixed seeds,
tracemalloc peak captured on a separate (non-timed) run so tracing overhead
never pollutes the timing numbers.
"""
import argparse
import gc
import json
import time
import tracemalloc
from math import prod, ceil

import numpy as np

import rechunkit
from rechunkit.main import _rechunk_plan

N_RUNS = 9


def _median_time(func):
    times = []
    for _ in range(N_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _peak_mem(func):
    gc.collect()
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def _run_rechunk_case(shape, src_cs, tgt_cs, max_mem, sel=None, dtype='int64', reps=1):
    """reps: consecutive full rechunks per timed run — lifts fast cases out of
    the noisy few-ms regime without inflating the array size."""
    arr = np.arange(prod(shape), dtype=dtype).reshape(shape)

    def source(slices):
        return arr[slices]

    def run():
        for _ in range(reps):
            for _wc, _data in rechunkit.rechunker(source, shape, arr.dtype, src_cs, tgt_cs, max_mem, sel=sel):
                pass

    median_s = _median_time(run)
    peak = _peak_mem(run)
    n_cells = prod(tuple(s.stop - s.start for s in sel) if sel is not None else shape) * reps
    return {
        'median_s': median_s,
        'cells_per_s': n_cells / median_s,
        'peak_mem_mb': peak / 2**20,
        'max_mem_mb': max_mem / 2**20,
    }


def _count_source_calls(shape, src_cs, tgt_cs, max_mem, dtype='float64'):
    """Run rechunker() to completion counting actual source-function calls."""
    arr = np.arange(prod(shape), dtype=dtype).reshape(shape)
    calls = [0]

    def source(slices):
        calls[0] += 1
        return arr[slices]

    for _wc, _data in rechunkit.rechunker(source, shape, arr.dtype, src_cs, tgt_cs, max_mem):
        pass
    return calls[0]


def _plan_group_types(shape, src_cs, tgt_cs, max_mem, itemsize=8):
    """Count plan group types (uses the private planner; benchmark-only)."""
    counts = {}
    for group_type, _r, _w, _g in _rechunk_plan(shape, itemsize, src_cs, tgt_cs, max_mem):
        counts[group_type] = counts.get(group_type, 0) + 1
    return counts


def run_benchmarks():
    results = {}

    # 1. Ideal path: LCM read shape fits comfortably.  reps lift each timed
    # run to ~50 ms so same-tree variance can gate core changes.
    results['rechunk_ideal'] = _run_rechunk_case(
        (4000, 3000), (100, 300), (300, 100), 2**30, reps=8)

    # 2. Constrained path: 2**19 sits BELOW the 720 KB LCM buffer, so this
    # genuinely exercises the constrained branch (2**20 did not).
    results['rechunk_constrained'] = _run_rechunk_case(
        (4000, 3000), (100, 300), (300, 100), 2**19, reps=8)

    # 3. Phase-shifted selection (sel starts mid-chunk in both dims).
    results['rechunk_sel_phase'] = _run_rechunk_case(
        (4000, 3000), (100, 300), (300, 100), 2**30,
        sel=(slice(37, 3937), slice(41, 2941)), reps=8)

    # 4. Identity-ish rechunk (same chunks, offset sel) - overhead-dominated case.
    results['rechunk_identity_sel'] = _run_rechunk_case(
        (4000, 3000), (100, 300), (100, 300), 2**30,
        sel=(slice(50, 3950), slice(150, 2850)), reps=8)

    # 5. Plan-only cost on a large virtual array (no data movement).
    def plan_only():
        for _ in range(5):
            rechunkit.calc_n_reads_rechunker((10000, 10000), 8, (100, 100), (250, 250), 2**27)
    results['plan_only_large'] = {'median_s': _median_time(plan_only)}

    # 5b. Plan-only cost on a SMALL constrained case — the regime where planner
    # additions (candidate scoring, shrink cascades) actually execute.
    def plan_only_small():
        for _ in range(5):
            rechunkit.calc_n_reads_rechunker((530, 110), 8, (3, 8), (6, 6), 2**10)
    results['plan_only_small_constrained'] = {'median_s': _median_time(plan_only_small)}

    # 6. guess_chunk_shape bulk timing.
    shapes = [(int(a), int(b), int(c)) for a, b, c in
              np.random.default_rng(7).integers(1, 10**6, size=(2000, 3))]

    def guess_bulk():
        for _ in range(4):
            for s in shapes:
                rechunkit.guess_chunk_shape(s, 4)
    results['guess_chunk_shape_2000'] = {'median_s': _median_time(guess_bulk)}

    # --- Deterministic gate cases (counts / allocation ratios, noise-free) ---

    # 7. Groupby-style amplification: non-divisor (1, full, full) target.
    # Scaled ERA5 analog; stored chunks = 8*8*15 = 960.  The read count vs
    # stored-chunk count is the P1 read/decompression amplification metric.
    gshape, gsrc, gtgt = (72, 73, 144), (10, 10, 10), (1, 73, 144)
    stored = prod(ceil(s / c) for s, c in zip(gshape, gsrc))
    for mm_name, mm in (('tight', 2**19), ('mid', 2**20)):
        reads = _count_source_calls(gshape, gsrc, gtgt, mm)
        results[f'groupby_amplification_{mm_name}'] = {
            'source_calls': reads,
            'stored_chunks': stored,
            'amplification': reads / stored,
            'max_mem_mb': mm / 2**20,
        }

    # 8. Pending reorder buffer on a wide array: ideal path, read group spans
    # 10 target rows, 800 groups across the width -> later-row chunks pend.
    pshape, psrc, ptgt, pmm = (50, 40000), (50, 50), (5, 50), 2**16
    pres = _run_rechunk_case(pshape, psrc, ptgt, pmm)
    pres['peak_over_budget'] = pres['peak_mem_mb'] * 2**20 / pmm
    results['pending_wide'] = pres

    # 8b. REDUCIBLE pending case: the read shape spans 2 target rows over an
    # 80-group-wide band; the R4 cascade can shrink it row-free.  (The
    # pending_wide case above is the IRREDUCIBLE regime — src0 > tgt0 at a
    # one-chunk read — kept as the documented-residual record.)
    rshape, rsrc, rtgt, rmm = (48, 24000), (4, 300), (6, 300), 2**19
    rres = _run_rechunk_case(rshape, rsrc, rtgt, rmm, dtype='float64')
    rres['peak_over_budget'] = rres['peak_mem_mb'] * 2**20 / rmm
    rres['source_calls'] = _count_source_calls(rshape, rsrc, rtgt, rmm)
    rres['stored_chunks'] = prod(ceil(s / c) for s, c in zip(rshape, rsrc))
    results['pending_reducible'] = rres

    # 9. Mixed bulk+single plan: src=6 vs tgt=4 along dim0 makes interior
    # write chunks alternate between bulk-served and single-path; bulk pending
    # and single-path memory can coexist.  Group-type counts are recorded so
    # the case self-documents if a future planner change makes it pure.
    mshape, msrc, mtgt, mmm = (48, 4000), (6, 50), (4, 50), 2**12
    mres = _run_rechunk_case(mshape, msrc, mtgt, mmm)
    mres['peak_over_budget'] = mres['peak_mem_mb'] * 2**20 / mmm
    mres.update({f'n_{k}': v for k, v in
                 _plan_group_types(mshape, msrc, mtgt, mmm).items()})
    results['mixed_plan_peak'] = mres

    return results


def compare(base_path, new_path):
    with open(base_path) as f:
        base = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    print(f"{'case':<28} {'metric':<14} {'base':>12} {'new':>12} {'delta':>8}")
    for case in base:
        if case not in new:
            continue
        for metric in base[case]:
            b, n = base[case][metric], new[case].get(metric)
            if n is None or not isinstance(b, (int, float)) or b == 0:
                continue
            delta = (n - b) / b * 100
            flag = '  <<<' if (metric in ('median_s', 'peak_mem_mb') and delta > 10) else ''
            print(f"{case:<28} {metric:<14} {b:>12.4g} {n:>12.4g} {delta:>+7.1f}%{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', help='write results JSON to this path')
    p.add_argument('--compare', nargs=2, metavar=('BASE', 'NEW'), help='compare two results files')
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    results = run_benchmarks()
    print(f"rechunkit {rechunkit.__version__}")
    for case, metrics in results.items():
        line = '  '.join(f"{k}={v:.4g}" for k, v in metrics.items())
        print(f"{case:<28} {line}")
    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"written: {args.output}")


if __name__ == '__main__':
    main()
