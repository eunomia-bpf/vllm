/*
 * CUDA UVM Allocator for vLLM
 *
 * This is a custom CUDA allocator that uses cudaMallocManaged
 * to enable Unified Virtual Memory (UVM) in vLLM.
 *
 * UVM allows memory oversubscription - allocating more GPU memory
 * than physically available by using CPU memory as backing store.
 *
 * Usage:
 *   Set environment variable VLLM_USE_UVM=1 before starting vLLM
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <atomic>

extern "C" {

// Global memory statistics (using C++ std::atomic)
static std::atomic<size_t> total_allocated{0};
static std::atomic<size_t> peak_allocated{0};
static std::atomic<size_t> num_allocs{0};
static std::atomic<size_t> num_frees{0};

// Configuration
static int enable_prefetch = 0;  // Whether to prefetch to device after allocation
static int verbose_logging = 0;  // Whether to log allocations

/**
 * Allocate CUDA managed (UVM) memory
 *
 * This function is called by PyTorch's pluggable allocator interface.
 *
 * @param size Size in bytes to allocate
 * @param device CUDA device ID
 * @param stream CUDA stream (unused for managed memory)
 * @return Pointer to allocated memory, or NULL on failure
 */
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    void* ptr = NULL;
    cudaError_t err;

    // Use cudaMallocManaged for UVM
    // cudaMemAttachGlobal makes memory accessible from any GPU and CPU
    err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMallocManaged failed for %zd bytes: %s\n",
                size, cudaGetErrorString(err));
        return NULL;
    }

    // Update statistics
    size_t current = total_allocated.fetch_add(size) + size;
    size_t alloc_count = num_allocs.fetch_add(1) + 1;

    // Update peak if needed (lock-free)
    size_t peak = peak_allocated.load();
    while (current > peak) {
        if (peak_allocated.compare_exchange_weak(peak, current)) {
            break;
        }
    }

    // Log large allocations (> 100MB)
    if (verbose_logging && size > 100 * 1024 * 1024) {
        fprintf(stderr, "[vLLM UVM] Alloc #%zu: %.2f MB (total: %.2f GB, peak: %.2f GB)\n",
                alloc_count, size / 1e6, current / 1e9,
                peak_allocated.load() / 1e9);
    }

    // Optionally prefetch to device
    // This can improve performance by proactively moving data to GPU
    if (enable_prefetch && device >= 0 && ptr != NULL) {
        cudaMemPrefetchAsync(ptr, size, device, stream);
    }

    return ptr;
}

/**
 * Free CUDA managed memory
 *
 * @param ptr Pointer to memory to free
 * @param size Size of allocation (for statistics)
 * @param device CUDA device ID (unused)
 * @param stream CUDA stream (unused)
 */
void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    if (ptr != NULL) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vLLM UVM] cudaFree failed: %s\n",
                    cudaGetErrorString(err));
        }

        // Update statistics
        total_allocated.fetch_sub(size);
        num_frees.fetch_add(1);
    }
}

// =============================================================================
// Statistics API (can be called from Python via ctypes)
// =============================================================================

/**
 * Get current allocated bytes
 */
size_t uvm_get_allocated_bytes(void) {
    return total_allocated.load();
}

/**
 * Get peak allocated bytes
 */
size_t uvm_get_peak_allocated_bytes(void) {
    return peak_allocated.load();
}

/**
 * Get total number of allocations
 */
size_t uvm_get_num_allocs(void) {
    return num_allocs.load();
}

/**
 * Get total number of frees
 */
size_t uvm_get_num_frees(void) {
    return num_frees.load();
}

/**
 * Reset peak statistics to current allocation
 */
void uvm_reset_peak_stats(void) {
    peak_allocated.store(total_allocated.load());
}

/**
 * Reset all statistics
 */
void uvm_reset_all_stats(void) {
    total_allocated.store(0);
    peak_allocated.store(0);
    num_allocs.store(0);
    num_frees.store(0);
}

/**
 * Enable/disable prefetching
 */
void uvm_set_prefetch(int enabled) {
    enable_prefetch = enabled;
}

/**
 * Enable/disable verbose logging
 */
void uvm_set_verbose(int enabled) {
    verbose_logging = enabled;
}

/**
 * Prefetch memory region to a specific device
 *
 * @param ptr Pointer to memory region
 * @param size Size of region in bytes
 * @param device Target device (-1 for CPU, >= 0 for GPU)
 * @param stream CUDA stream for async prefetch
 */
void uvm_prefetch(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr != NULL && size > 0) {
        cudaMemPrefetchAsync(ptr, size, device, stream);
    }
}

/**
 * Set memory advice for a region
 *
 * @param ptr Pointer to memory region
 * @param size Size of region in bytes
 * @param advice One of: 1=ReadMostly, 2=PreferredLocation, 3=AccessedBy
 * @param device Device ID for the advice
 */
void uvm_advise(void* ptr, size_t size, int advice, int device) {
    if (ptr == NULL || size == 0) return;

    cudaMemoryAdvise cuda_advice;
    switch (advice) {
        case 1:
            cuda_advice = cudaMemAdviseSetReadMostly;
            break;
        case 2:
            cuda_advice = cudaMemAdviseSetPreferredLocation;
            break;
        case 3:
            cuda_advice = cudaMemAdviseSetAccessedBy;
            break;
        default:
            fprintf(stderr, "[vLLM UVM] Unknown advice type: %d\n", advice);
            return;
    }

    cudaError_t err = cudaMemAdvise(ptr, size, cuda_advice, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMemAdvise failed: %s\n",
                cudaGetErrorString(err));
    }
}

}  // extern "C"
