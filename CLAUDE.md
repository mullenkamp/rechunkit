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
uv run lint:style         # Check style (ruff + black)
uv run lint:typing        # Run mypy type checking
uv run lint:fmt           # Auto-format code
uv run lint:all           # All lint checks
uv run docs-serve         # Local docs server (mkdocs)
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

**Key data flow:** `source` is a callable that accepts a tuple of slices and returns an ndarray. `rechunker()` is a generator yielding `(target_slices, data)` tuples.

**Composite numbers table** (`composite_numbers` at module top): Pre-computed highly composite numbers used by `guess_chunk_shape()` to pick chunk dimensions that produce small LCMs.

## Dependencies

- Runtime: `numpy>=1.26`
- Python: `>=3.9` (CI tests 3.10, 3.11, 3.12)
