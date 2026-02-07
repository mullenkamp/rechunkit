# GEMINI.md

## Project Overview
**Rechunkit** is a specialized Python library designed for the efficient rechunking of multidimensional arrays stored as chunks (e.g., NumPy ndarrays). It provides a generator-based approach that allows for on-the-fly rechunking, avoiding the need to load or save the entire target array into memory.

### Key Features
- **Generator-based API:** Returns `(slice, data)` tuples for target chunks.
- **LCM Optimization:** Uses composite numbers for chunk sizes to minimize redundant reads by reducing the Least Common Multiple (LCM) between source and target chunks.
- **Memory Management:** Allows users to specify `max_mem` to balance between speed (fewer reads) and memory usage.
- **Preprocessing Tools:** Functions to estimate optimal chunk shapes and calculate the number of reads/writes required.

### Architecture
- **`rechunkit/main.py`:** Contains the entire core logic, including the `rechunker` generator and helper calculation functions.
- **`rechunkit/__init__.py`:** Re-exports the public API for easy access.
- **Algorithm:** Uses a two-tier approach:
  - **Ideal path:** When the LCM of source and target chunk shapes fits in memory, each source chunk is read exactly once.
  - **Constrained path:** When memory is limited, it performs partial reads and tracks progress to minimize redundant operations.

---

## Building and Running
The project uses [uv](https://docs.astral.sh/uv/) as its primary environment and build manager.

### Development Commands
| Task | Command |
| :--- | :--- |
| **Build** | `uv build` |
| **Test** | `uv run test` |
| **Coverage** | `uv run cov` |
| **Lint (All)** | `uv run lint:all` |
| **Lint (Style)** | `uv run lint:style` |
| **Lint (Type)** | `uv run lint:typing` |
| **Format** | `uv run lint:fmt` |
| **Docs (Serve)** | `uv run docs-serve` |
| **Docs (Build)** | `uv run docs-build` |

---

## Development Conventions

### Code Style
- **Line Length:** 120 characters.
- **Formatting:** Handled by `black` (with `skip-string-normalization = true`).
- **Linting:** Handled by `ruff` (targeting Python 3.11).
- **Imports:** **Absolute imports are required** (e.g., `from rechunkit.main import ...`). Relative imports are banned.
- **Typing:** Type hints are used throughout; `mypy` is used for static analysis.

### Testing Practices
- **Framework:** `pytest`.
- **Location:** Tests are located in `rechunkit/tests/`.
- **Patterns:** Tests typically involve creating a source NumPy array, wrapping its `__getitem__` method, and verifying that the rechunked output matches the original array or a selection.
- **Coverage:** Aim for high coverage, especially for different combinations of source/target chunk shapes and memory constraints.

### Documentation
- Built with **MkDocs** and **Material for MkDocs**.
- Documentation source is in the `docs/` directory.
- `mkdocstrings` is used to generate reference documentation from docstrings in `rechunkit/main.py`.
