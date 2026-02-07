import numpy as np
import matplotlib.pyplot as plt
from rechunkit import calc_n_reads_rechunker, calc_n_reads_simple, calc_n_chunks, calc_ideal_read_chunk_shape, calc_ideal_read_chunk_mem
import os

def generate_memory_plot(output_path):
    """
    Plot 1: Buffer ROI (Read Count vs. Memory)
    Shows how increasing memory reduces source reads compared to brute force.
    """
    source_shape = (64, 64, 64)
    source_chunk_shape = (8, 8, 8)
    target_chunk_shape = (10, 10, 10)
    itemsize = 4
    
    # Calculate Ideal Memory
    ideal_shape = calc_ideal_read_chunk_shape(source_chunk_shape, target_chunk_shape)
    ideal_mem = calc_ideal_read_chunk_mem(ideal_shape, itemsize)
    
    # Range of memory: From very constrained (1 source chunk) to Ideal
    min_mem = np.prod(source_chunk_shape) * itemsize
    mem_points = np.linspace(min_mem, ideal_mem * 1.1, 20).astype(int)
    
    rechunkit_reads = []
    simple_reads = []
    
    n_simple = calc_n_reads_simple(source_shape, source_chunk_shape, target_chunk_shape)
    
    for mem in mem_points:
        n_r, _ = calc_n_reads_rechunker(source_shape, itemsize, source_chunk_shape, target_chunk_shape, mem)
        rechunkit_reads.append(n_r)
        simple_reads.append(n_simple)
        
    plt.figure(figsize=(10, 6))
    plt.plot(mem_points / 1024, simple_reads, label='Brute Force (Simple)', linestyle='--', color='red')
    plt.plot(mem_points / 1024, rechunkit_reads, label='Rechunkit Optimized', marker='o', color='green')
    
    plt.axvline(x=ideal_mem / 1024, color='blue', linestyle=':', label='Ideal Memory Threshold')
    
    plt.title('I/O Efficiency: Buffer Size vs. Source Reads')
    plt.xlabel('Available Memory (KB)')
    plt.ylabel('Total Source Chunk Reads')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Generated {output_path}")
    plt.close()

def generate_scalability_plot(output_path):
    """
    Plot 2: Scalability (Read Count vs. Target Chunk Size)
    Shows how rechunkit handles small/misaligned target chunks efficiently compared to brute force.
    """
    source_shape = (40, 40, 40)
    source_chunk_shape = (10, 10, 10)
    itemsize = 4
    max_mem = 32000  # Enough for roughly 8 source chunks (8 * 1000 * 4)
    
    # Focus on SMALL target sizes where brute force explodes
    target_sizes = range(2, 16)
    rechunkit_reads = []
    simple_reads = []
    
    for t in target_sizes:
        tgt_chunks = (t, t, t)
        
        n_simple = calc_n_reads_simple(source_shape, source_chunk_shape, tgt_chunks)
        n_r, _ = calc_n_reads_rechunker(source_shape, itemsize, source_chunk_shape, tgt_chunks, max_mem)
        
        rechunkit_reads.append(n_r)
        simple_reads.append(n_simple)
        
    plt.figure(figsize=(10, 6))
    plt.plot(target_sizes, simple_reads, label='Brute Force (Simple)', linestyle='--', color='red')
    plt.plot(target_sizes, rechunkit_reads, label='Rechunkit Optimized', marker='o', color='green')
    
    plt.title('Performance: Target Chunk Size vs. Source Reads')
    plt.xlabel('Target Chunk Dimension (NxNxN)')
    plt.ylabel('Total Source Chunk Reads')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Generated {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_memory_plot('docs/assets/benchmark_memory.png')
    generate_scalability_plot('docs/assets/benchmark_scalability.png')