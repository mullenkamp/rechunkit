"""Functions to efficiently rechunk multidimensional arrays"""
from rechunkit.main import guess_chunk_shape, chunk_range, calc_ideal_read_chunk_shape, calc_ideal_read_chunk_mem, calc_source_read_chunk_shape, calc_n_chunks, calc_n_reads_simple, calc_n_reads_rechunker, rechunker


__version__ = '0.4.0'
