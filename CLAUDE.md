# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rechunkit is a Python library for efficiently rechunking multidimensional numpy arrays stored as chunks. It uses a generator-based approach for on-the-fly rechunking without requiring the full target array in memory. The core optimization uses composite numbers for chunk sizes to minimize LCM values between source and target chunks, reducing redundant reads.

## Build & Development Commands

All commands use [uv](https://docs.astral.sh/uv/) as the build/environment manager:

```bash
uv build                  # Build distribution packages
uv run test               # Run pytest
uv run cov                # Run tests with coverage report
uv run docs-serve         # Local docs server (mkdocs)
uv run docs-build         # Build docs to site/
uv sync --group docs      # Install docs dependencies (mkdocs-material, mkdocstrings)
```

To run a single test: `uv run pytest rechunkit/tests/test_rechunkit.py::test_name`

## Code Style

- Line length: 120 characters
- Formatter: black (string normalization disabled)
- Linter: ruff (target Python 3.11)
- Relative imports are banned; use absolute imports (`from rechunkit.main import ...`)

## Architecture

The entire library lives in a single module: `rechunkit/main.py`. The public API is re-exported from `rechunkit/__init__.py`.

**Two-tier rechunking algorithm in `rechunker()`:**
- **Ideal path:** When the (extent-clipped) LCM of source/target chunk shapes fits in `max_mem`, reads each source chunk exactly once. Uses `chunk_range` to iterate over read groups, then yields target chunks from in-memory buffer. The ideal shape is clipped per dim to the smallest source-aligned cover of the tiled extent (`calc_ideal_read_chunk_shape(shape=...)`) — do not remove the clip; unclipped LCMs on non-divisor full-field targets inflate to absurd sizes and force the worst path.
- **Constrained path:** When memory is insufficient for ideal chunks, `calc_source_read_chunk_shape()` computes a reduced read chunk. Bulk groups serve write chunks whose read region fits the buffer; the rest go to the **batched single path** (`'multi'` groups): consecutive write chunks batched in C-order with deduplicated reads, one dedicated buffer per write chunk, bounded by `n_batch`. Bulk mode must be AFFORDABLE (room for >= 2 batch buffers besides the bulk buffer + pending band) or the plan goes all-multi — an n_batch=1 mixed plan cannot dedup and is strictly worse.

**Memory accounting invariants (do not "simplify"):**
- `_buffer_bytes` (per-dim `max(read + phase, target)` x itemsize) is the single source of truth for the bulk buffer, used by BOTH the planner and `rechunker()`'s allocation. `rechunker()` threads its computed read shape into `_rechunk_plan` via `_read_chunk_shape` so plan and buffer agree BY CONSTRUCTION.
- `_pending_bytes` bounds the canonical-order reorder band; the planner enforces buffer + pending <= max_mem, shrinking only dims whose shrink STRICTLY reduces the total (blind shrinks degrade reads without saving memory). The wide-array residual (one source chunk spanning multiple target rows) is irreducible and documented.
- The candidate check (constrained plans under `_SCORE_LIMIT` target chunks) scores up to 5 read-shape candidates by running `_rechunk_plan` in counting mode. The override is a threaded PARAMETER, never a module-global (concurrent plans). The greedy shape is always in the set (no plan can be worse than pre-check). A pinned sweep (`test_read_count_monotonicity` + the one-off 1,500-config gate) guards read-count monotonicity empirically.

**Selection alignment via phase shifting:** When `sel` is provided, `_rechunk_plan()` computes `phase = sel.start % source_chunk_shape` per dimension. Read groups are extended backward by this phase so that reads, after being shifted by the selection offset in `rechunker()`, land on source chunk boundaries. This ensures source functions backed by chunk-based storage always receive aligned reads. The buffer is increased by `phase` to accommodate the extra pre-selection data. `_exact_chunk_range()` is used instead of `chunk_range()` to avoid floor-alignment issues with negative start positions. When `sel` is `None` or chunk-aligned, phase is zero and behavior is identical to the un-shifted case.

**Canonical yield order:** `rechunker()` always yields target chunks in C-order (row-major) based only on `target_chunk_shape` and the target shape, independent of source chunk layout or `max_mem`. A reordering buffer with direct-yield optimization ensures this without increasing read counts.

**Yield lifetime contract:** direct-yield chunks are VIEWS into the internal buffer (reused as iteration advances); reorder-path chunks are copies. Consumers must consume or `.copy()` each yielded array before advancing the generator and treat yields as read-only — documented in the `rechunker()` docstring and `docs/concepts/how-it-works.md`; keep those in sync with any buffer-handling change.

**Input validation:** `guess_chunk_shape` accepts numpy integers, rejects dims <= 0 (a zero chunk dim is never valid output), and still returns `()` for the empty shape — pinned by tests.

**Key data flow:** `source` is a callable that accepts a tuple of slices and returns an ndarray. `rechunker()` is a generator yielding `(target_slices, data)` tuples.

**Composite numbers table** (`composite_numbers` at module top): Pre-computed highly composite numbers used by `guess_chunk_shape()` to pick chunk dimensions that produce small LCMs.

## Documentation

Docs use mkdocs-material with mkdocstrings (numpy docstring style). Structure:

- `docs/index.md` — standalone homepage (not README inline)
- `docs/getting-started/` — installation, quickstart
- `docs/guide/` — preprocessing, rechunking, integration
- `docs/concepts/` — algorithm explanation, composite numbers
- `docs/reference/` — API overview + per-function mkdocstrings directives

Docs deps are in the `docs` dependency group. CI deploys via `.github/workflows/documentation.yml` on push to main.

## Dependencies

- Runtime: `numpy>=1.26`
- Python: `>=3.9` (CI tests 3.10, 3.11, 3.12)
