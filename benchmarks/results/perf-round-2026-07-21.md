# Perf-round benchmark results — 2026-07-21

Comparison of `perf-baseline-2026-07-21.json` (pre-round, commit db5ca6c) vs
`after-perf-2026-07-21.json` (all perf-round changes: true-buffer budgeting,
extent-clipped ideal, batched single path, pending-aware planning, candidate
check). Same harness, same machine/venv; median of 9 runs per timing case;
same-tree timing variance <= ~3% (verified with back-to-back runs), so timing
deltas inside that band are noise. Count and memory-ratio metrics are
deterministic.

Regenerate with:

    uv run python benchmarks/bench_rechunk.py --compare \
        benchmarks/results/perf-baseline-2026-07-21.json \
        benchmarks/results/after-perf-2026-07-21.json

## Headline (deterministic gates)

| Case | Metric | Before | After |
|---|---|---|---|
| groupby_amplification_mid (1 MB) | read amplification | 9x | **1x** (ideal path restored by the extent clip) |
| groupby_amplification_tight (512 KB) | read amplification | 9x | **2.12x** (batched single path) |
| rechunk_constrained | throughput | 852 M cells/s | **1.62 G cells/s (+91%)** at identical peak memory (free-growth floor) |
| pending_reducible | peak over budget | 2.54x (pre-round, measured separately) | **1.41x** at 1.33x reads (pending-aware planning; case added mid-round) |
| mixed_plan_peak | peak over budget | 56.6x | 53.1x (documented irreducible residual; see below) |
| pending_wide | peak over budget | 271x | 271x — **unchanged BY DESIGN** (irreducible tall-source/flat-target residual; the alternative is ~10x read amplification; formula documented in how-it-works.md) |

## Known regression (accepted at the R5 checkpoint)

`plan_only_small_constrained` +523% (45 ms -> 282 ms): the read-shape candidate
check runs up to 4 extra planning passes on constrained plans under 20,000
target chunks. This is planning time only (~35 us per target chunk per
candidate) — invisible against actual data movement; it buys never-worse plans
(monotonicity violations 203 -> 104 on the pinned 1,500-config sweep, worst
jump 1.67x). This case is the permanent canary for that cost.

## Full comparison

```
case                         metric                 base          new    delta
rechunk_ideal                median_s            0.06084      0.05956    -2.1%
rechunk_ideal                cells_per_s       1.578e+09    1.612e+09    +2.1%
rechunk_ideal                peak_mem_mb           1.492        1.492    +0.0%
rechunk_ideal                max_mem_mb             1024         1024    +0.0%
rechunk_constrained          median_s             0.1127      0.05917   -47.5%
rechunk_constrained          cells_per_s       8.519e+08    1.622e+09   +90.5%
rechunk_constrained          peak_mem_mb           1.489        1.492    +0.2%
rechunk_constrained          max_mem_mb              0.5          0.5    +0.0%
rechunk_sel_phase            median_s            0.07915      0.07782    -1.7%
rechunk_sel_phase            cells_per_s       1.143e+09    1.163e+09    +1.7%
rechunk_sel_phase            peak_mem_mb           1.874        1.874    +0.0%
rechunk_sel_phase            max_mem_mb             1024         1024    +0.0%
rechunk_identity_sel         median_s             0.1001       0.1013    +1.2%
rechunk_identity_sel         cells_per_s       8.414e+08    8.318e+08    -1.1%
rechunk_identity_sel         peak_mem_mb            1.15         1.15    +0.0%
rechunk_identity_sel         max_mem_mb             1024         1024    +0.0%
plan_only_large              median_s            0.04325      0.04283    -1.0%
plan_only_small_constrained  median_s            0.04524       0.2821  +523.5%  <<<
guess_chunk_shape_2000       median_s            0.05693      0.05663    -0.5%
groupby_amplification_tight  source_calls           8640         2040   -76.4%
groupby_amplification_tight  stored_chunks           960          960    +0.0%
groupby_amplification_tight  amplification             9        2.125   -76.4%
groupby_amplification_tight  max_mem_mb              0.5          0.5    +0.0%
groupby_amplification_mid    source_calls           8640          960   -88.9%
groupby_amplification_mid    stored_chunks           960          960    +0.0%
groupby_amplification_mid    amplification             9            1   -88.9%
groupby_amplification_mid    max_mem_mb                1            1    +0.0%
pending_wide                 median_s             0.0197      0.01953    -0.9%
pending_wide                 cells_per_s       1.015e+08    1.024e+08    +0.9%
pending_wide                 peak_mem_mb           16.93        16.94    +0.0%
pending_wide                 max_mem_mb           0.0625       0.0625    +0.0%
pending_wide                 peak_over_budget          271          271    +0.0%
mixed_plan_peak              median_s           0.009135     0.009491    +3.9%
mixed_plan_peak              cells_per_s       2.102e+07    2.023e+07    -3.7%
mixed_plan_peak              peak_mem_mb          0.2213       0.2076    -6.2%
mixed_plan_peak              max_mem_mb         0.003906     0.003906    +0.0%
mixed_plan_peak              peak_over_budget        56.64        53.14    -6.2%
```
