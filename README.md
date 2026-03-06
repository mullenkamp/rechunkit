# rechunkit

<p align="center">
    <em>Functions to efficiently rechunk multidimensional arrays</em>
</p>

[![codecov](https://codecov.io/gh/mullenkamp/rechunkit/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/rechunkit)
[![PyPI version](https://badge.fury.io/py/rechunkit.svg)](https://badge.fury.io/py/rechunkit)

---

**Documentation**: <a href="https://mullenkamp.github.io/rechunkit/" target="_blank">https://mullenkamp.github.io/rechunkit/</a>

**Source Code**: <a href="https://github.com/mullenkamp/rechunkit" target="_blank">https://github.com/mullenkamp/rechunkit</a>

---

## Introduction

Rechunkit is a Python library for efficiently rechunking multidimensional numpy arrays stored as chunks. It uses a generator-based approach for on-the-fly rechunking without requiring the full target array in memory.

## Key Features

- **Efficient On-the-Fly Rechunking:** Uses Python generators to yield rechunked data without requiring the full target array to be stored in memory.
- **Memory-Aware Optimization:** Employs a smart scaling algorithm to maximize performance within a user-defined memory limit (`max_mem`).
- **LCM Minimization:** Utilizes highly composite numbers for chunk guessing to minimize the Least Common Multiple (LCM) between source and target, significantly reducing redundant reads.
- **Flexible Data Access:** Supports subset selection (`sel`) and works with any source that implements a numpy `__getitem__` style callable (method or function).
- **Preprocessing Utilities:** Includes tools for estimating ideal chunk shapes, calculating memory requirements, and predicting the number of required read operations.

## Installation

```
pip install rechunkit
```

## Quick Example

```python
import numpy as np
from math import prod
from rechunkit import rechunker

shape = (31, 31, 31)
dtype = np.dtype('int32')
source_data = np.arange(1, prod(shape) + 1, dtype=dtype).reshape(shape)
source = source_data.__getitem__

target = np.zeros(shape, dtype=dtype)
for write_chunk, data in rechunker(source, shape, dtype, (5, 2, 4), (4, 5, 3), max_mem=2000):
    target[write_chunk] = data

assert np.all(source_data == target)
```

See the [documentation](https://mullenkamp.github.io/rechunkit/) for detailed guides, integration examples (h5py, zarr), and the full API reference.

## License

This project is licensed under the terms of the Apache Software License 2.0.
