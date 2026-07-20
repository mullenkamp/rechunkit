"""Rechunkit performance benchmark suite.

Measures rechunker() throughput and peak memory on fixed, seeded configurations
covering the three planning regimes (ideal, constrained, phase-shifted sel),
plus plan-only and guess_chunk_shape timings.

Usage:
    uv run python benchmarks/bench_rechunk.py --output benchmarks/results/baseline-YYYY-MM-DD.json
    uv run python benchmarks/bench_rechunk.py --compare benchmarks/results/baseline.json benchmarks/results/after.json

Protocol: median of N_RUNS (default 5) wall-clock runs per case, fixed seeds,
tracemalloc peak captured on a separate (non-timed) run so tracing overhead
never pollutes the timing numbers.
"""
import argparse
import gc
import json
import time
import tracemalloc
from math import prod

import numpy as np

import rechunkit

N_RUNS = 5


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


def _run_rechunk_case(shape, src_cs, tgt_cs, max_mem, sel=None, dtype='int64'):
    arr = np.arange(prod(shape), dtype=dtype).reshape(shape)

    def source(slices):
        return arr[slices]

    def run():
        for _wc, _data in rechunkit.rechunker(source, shape, arr.dtype, src_cs, tgt_cs, max_mem, sel=sel):
            pass

    median_s = _median_time(run)
    peak = _peak_mem(run)
    n_cells = prod(tuple(s.stop - s.start for s in sel) if sel is not None else shape)
    return {
        'median_s': median_s,
        'cells_per_s': n_cells / median_s,
        'peak_mem_mb': peak / 2**20,
        'max_mem_mb': max_mem / 2**20,
    }


def run_benchmarks():
    results = {}

    # 1. Ideal path: LCM read shape fits comfortably.
    results['rechunk_ideal'] = _run_rechunk_case(
        (2000, 3000), (100, 300), (300, 100), 2**30)

    # 2. Constrained path: buffer smaller than the LCM shape.
    results['rechunk_constrained'] = _run_rechunk_case(
        (2000, 3000), (100, 300), (300, 100), 2**20)

    # 3. Phase-shifted selection (sel starts mid-chunk in both dims).
    results['rechunk_sel_phase'] = _run_rechunk_case(
        (2000, 3000), (100, 300), (300, 100), 2**30,
        sel=(slice(37, 1937), slice(41, 2941)))

    # 4. Identity-ish rechunk (same chunks, offset sel) - overhead-dominated case.
    results['rechunk_identity_sel'] = _run_rechunk_case(
        (2000, 3000), (100, 300), (100, 300), 2**30,
        sel=(slice(50, 1950), slice(150, 2850)))

    # 5. Plan-only cost on a large virtual array (no data movement).
    def plan_only():
        rechunkit.calc_n_reads_rechunker((10000, 10000), 8, (100, 100), (250, 250), 2**27)
    results['plan_only_large'] = {'median_s': _median_time(plan_only)}

    # 6. guess_chunk_shape bulk timing.
    shapes = [(int(a), int(b), int(c)) for a, b, c in
              np.random.default_rng(7).integers(1, 10**6, size=(500, 3))]

    def guess_bulk():
        for s in shapes:
            rechunkit.guess_chunk_shape(s, 4)
    results['guess_chunk_shape_500'] = {'median_s': _median_time(guess_bulk)}

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
