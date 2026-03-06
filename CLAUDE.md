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
- **Ideal path:** When the LCM of source/target chunk shapes fits in `max_mem`, reads each source chunk exactly once. Uses `chunk_range` to iterate over read groups, then yields target chunks from in-memory buffer.
- **Constrained path:** When memory is insufficient for ideal chunks, `calc_source_read_chunk_shape()` computes a reduced read chunk. The algorithm then performs smart partial reads, tracking already-written chunks via a set to avoid duplicates. Some source chunks may be read multiple times.

**Canonical yield order:** `rechunker()` always yields target chunks in C-order (row-major) based only on `target_chunk_shape` and the target shape, independent of source chunk layout or `max_mem`. A reordering buffer with direct-yield optimization ensures this without increasing read counts.

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
