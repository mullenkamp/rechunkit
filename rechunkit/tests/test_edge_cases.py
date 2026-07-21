import numpy as np
import pytest
from math import prod
from rechunkit import rechunker, calc_source_read_chunk_shape, calc_n_reads_rechunker

@pytest.mark.parametrize("shape, src_chunks, tgt_chunks, selection", [
    ((100,), (7,), (11,), None),
    ((100,), (7,), (11,), (slice(10, 20),)),
    ((100,), (10,), (10,), (slice(5, 6),)), # single element
    ((100,), (10,), (10,), (slice(5, 5),)), # empty selection
])
def test_rechunk_1d_parametrized(shape, src_chunks, tgt_chunks, selection):
    """Test 1D rechunking with various selections and chunking."""
    dt = np.dtype('int32')
    src_arr = np.arange(prod(shape), dtype=dt).reshape(shape)
    
    if selection is None:
        expected = src_arr
        tgt_shape = shape
    else:
        expected = src_arr[selection]
        tgt_shape = tuple(s.stop - s.start for s in selection)
        
    target = np.zeros(tgt_shape, dtype=dt)
    for write_slices, data in rechunker(src_arr.__getitem__, shape, dt, src_chunks, tgt_chunks, 1000, selection):
        target[write_slices] = data
        
    assert np.all(target == expected)

@pytest.mark.parametrize("source_chunk_shape, target_chunk_shape, max_mem", [
    ((1000, 1000), (1000, 1000), 500 * 4),   # mem < source chunk -> floor
    ((10, 10), (20, 20), 100 * 4),           # floor dominated by target dims
    ((1, 1), (1000000, 1000000), 100 * 4),   # huge LCM, small mem
])
def test_scaling_logic_large_arrays(source_chunk_shape, target_chunk_shape, max_mem):
    """calc_source_read_chunk_shape budgets the TRUE buffer (per-dim
    max(read+phase, target)); when even one source chunk busts the budget it
    takes memory-free growth up to the target-forced buffer dims."""
    itemsize = 4
    res = calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)

    # Check that result is a multiple of source_chunk_shape
    for r, s in zip(res, source_chunk_shape):
        assert r % s == 0

    # The true buffer either fits the budget, or every dim stays within the
    # extent the target chunk forces anyway (allocation-neutral floor).
    true_buf = prod(max(r, t) for r, t in zip(res, target_chunk_shape)) * itemsize
    floor_buf = prod(max(s, t) for s, t in zip(source_chunk_shape, target_chunk_shape)) * itemsize
    assert true_buf <= max_mem or true_buf == floor_buf

    # For the huge LCM case, ensure it didn't hang and returned something sensible
    if source_chunk_shape == (1, 1):
        assert prod(res) > 1

@pytest.mark.parametrize("sel", [
    (slice(0, 0), slice(0, 10)),
    (slice(5, 5), slice(5, 5)),
    (slice(10, 10), slice(0, 10)),
])
def test_empty_selection_2d(sel):
    """Verify that rechunking an empty selection works and returns no data."""
    shape = (10, 10)
    dt = np.dtype('int32')
    src_arr = np.arange(100, dtype=dt).reshape(shape)
    
    # This should yield nothing or work correctly with zero-sized array
    results = list(rechunker(src_arr.__getitem__, shape, dt, (2, 2), (3, 3), 1000, sel))
    
    # Target shape based on selection
    tgt_shape = tuple(s.stop - s.start for s in sel)
    assert prod(tgt_shape) == 0
    
    # If there are results, they should all be empty
    for write_slices, data in results:
        assert data.size == 0
        assert all(s.stop == s.start for s in write_slices)

def test_single_element_selection_high_dim():
    """Test selection of exactly one element in a 3D array."""
    shape = (10, 10, 10)
    dt = np.dtype('int32')
    src_arr = np.arange(1000, dtype=dt).reshape(shape)
    sel = (slice(5, 6), slice(5, 6), slice(5, 6))
    
    results = list(rechunker(src_arr.__getitem__, shape, dt, (2, 2, 2), (3, 3, 3), 1000, sel))
    assert len(results) == 1
    write_slices, data = results[0]
    assert data.shape == (1, 1, 1)
    assert data[0, 0, 0] == src_arr[5, 5, 5]
