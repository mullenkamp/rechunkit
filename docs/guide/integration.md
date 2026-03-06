# Integration

rechunkit works with any storage backend that provides a numpy `__getitem__`-style callable. This page shows common integration patterns.

## h5py

```python
import h5py
from rechunkit import rechunker

with h5py.File('source.h5', 'r') as f_src, h5py.File('target.h5', 'w') as f_tgt:
    dset_src = f_src['data']

    dset_tgt = f_tgt.create_dataset(
        'data',
        shape=dset_src.shape,
        dtype=dset_src.dtype,
        chunks=(4, 5, 3),
    )

    for write_slices, data in rechunker(
        source=dset_src.__getitem__,
        shape=dset_src.shape,
        dtype=dset_src.dtype,
        source_chunk_shape=dset_src.chunks,
        target_chunk_shape=dset_tgt.chunks,
        max_mem=100 * 1024 * 1024,  # 100 MB
    ):
        dset_tgt[write_slices] = data
```

## Zarr

```python
import zarr
from rechunkit import rechunker

source = zarr.open('source.zarr', mode='r')['data']
target = zarr.open('target.zarr', mode='w')

target_arr = target.create_dataset(
    'data',
    shape=source.shape,
    dtype=source.dtype,
    chunks=(4, 5, 3),
)

for write_slices, data in rechunker(
    source=source.__getitem__,
    shape=source.shape,
    dtype=source.dtype,
    source_chunk_shape=source.chunks,
    target_chunk_shape=target_arr.chunks,
    max_mem=100 * 1024 * 1024,
):
    target_arr[write_slices] = data
```

## Custom Source Functions

Any callable that accepts a tuple of slices and returns a numpy array works as a source. For example, a function that reads from a remote API:

```python
import numpy as np
from rechunkit import rechunker

def read_from_api(slices):
    # Fetch data for the given slice range from your API
    # Must return a numpy array
    ...

for write_slices, data in rechunker(
    source=read_from_api,
    shape=(1000, 2000),
    dtype=np.float32,
    source_chunk_shape=(100, 200),
    target_chunk_shape=(50, 100),
    max_mem=500_000,
):
    save_chunk(write_slices, data)
```

The only requirement is that `source(tuple_of_slices)` returns a numpy ndarray with the expected shape and dtype.
