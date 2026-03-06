# rechunkit

**Functions to efficiently rechunk multidimensional arrays**

[![build](https://github.com/mullenkamp/rechunkit/workflows/Build/badge.svg)](https://github.com/mullenkamp/rechunkit/actions)
[![codecov](https://codecov.io/gh/mullenkamp/rechunkit/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/rechunkit)
[![PyPI version](https://badge.fury.io/py/rechunkit.svg)](https://badge.fury.io/py/rechunkit)

---

rechunkit is a Python library for efficiently rechunking multidimensional numpy arrays stored as chunks. It uses a generator-based approach for on-the-fly rechunking without requiring the full target array in memory.

## Key Features

- **Efficient On-the-Fly Rechunking** — Python generators yield rechunked data without storing the full target array in memory
- **Memory-Aware Optimization** — smart scaling algorithm maximizes performance within a user-defined memory limit
- **LCM Minimization** — highly composite numbers for chunk guessing minimize the Least Common Multiple between source and target, reducing redundant reads
- **Flexible Data Access** — subset selection and compatibility with any numpy `__getitem__`-style callable
- **Preprocessing Utilities** — tools for estimating chunk shapes, calculating memory requirements, and predicting read counts

## Quick Example

```python
import numpy as np
from rechunkit import guess_chunk_shape, rechunker

shape = (100, 100, 100)
dtype = np.dtype('float32')

source_data = np.random.rand(*shape).astype(dtype)
source = source_data.__getitem__

target_chunk_shape = guess_chunk_shape(shape, dtype.itemsize, target_chunk_size=4000)

target = np.zeros(shape, dtype=dtype)
for write_chunk, data in rechunker(source, shape, dtype, (10, 10, 10), target_chunk_shape, max_mem=50000):
    target[write_chunk] = data
```

## Next Steps

- [Installation](getting-started/installation.md) — install rechunkit
- [Quick Start](getting-started/quickstart.md) — complete walkthrough of a typical workflow
- [User Guide](guide/preprocessing.md) — detailed guides for every function
- [API Reference](reference/index.md) — full function reference
