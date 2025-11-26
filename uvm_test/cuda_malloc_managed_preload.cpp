/*
 * LD_PRELOAD library to intercept cudaMalloc and replace with cudaMallocManaged
 *
 * This library intercepts all CUDA memory allocation calls and redirects them
 * to use Unified Virtual Memory (UVM) via cudaMallocManaged. This enables
 * memory oversubscription - allocating more GPU memory than physically available.
 *
 * Usage:
 *   LD_PRELOAD=./libcudamalloc_managed.so python your_script.py
 *
 * Environment variables:
 *   CUDA_MANAGED_VERBOSE=1    - Log all intercepted allocations
 *   CUDA_MANAGED_DISABLE=1    - Disable interception (use normal cudaMalloc)
 *
 * Note: This intercepts cudaMalloc, cudaMallocAsync, cudaMallocPitch, etc.
 * but NOT cudaMallocHost (which is for pinned CPU memory).
 *
 * IMPORTANT: cuBLAS uses internal allocators that may bypass cudaMalloc.
 * We also intercept cuMemAlloc (driver API) to catch those allocations.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <cuda_runtime.h>

// Undefine the macros before including cuda.h to avoid redefinition issues
// CUDA headers define cuMemAlloc -> cuMemAlloc_v2, etc.
#include <cuda.h>
#undef cuMemAlloc
#undef cuMemFree

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <unordered_map>

// Statistics
static std::atomic<size_t> total_allocated{0};
static std::atomic<size_t> peak_allocated{0};
static std::atomic<size_t> num_allocs{0};
static std::atomic<size_t> num_managed_allocs{0};

// Track allocations for proper freeing
static std::unordered_map<void*, size_t> allocation_sizes;
static std::mutex alloc_map_mutex;

// Configuration (read from environment on first use)
static int verbose = -1;  // -1 = not initialized
static int disabled = -1;

// Original function pointers - Runtime API
static cudaError_t (*real_cudaMalloc)(void**, size_t) = nullptr;
static cudaError_t (*real_cudaMallocAsync)(void**, size_t, cudaStream_t) = nullptr;
static cudaError_t (*real_cudaFree)(void*) = nullptr;
static cudaError_t (*real_cudaFreeAsync)(void*, cudaStream_t) = nullptr;

// Original function pointers - Driver API (used by cuBLAS internally)
static CUresult (*real_cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
static CUresult (*real_cuMemAlloc_v2)(CUdeviceptr*, size_t) = nullptr;
static CUresult (*real_cuMemFree)(CUdeviceptr) = nullptr;
static CUresult (*real_cuMemFree_v2)(CUdeviceptr) = nullptr;

// Mutex for initialization
static std::mutex init_mutex;

// Initialize configuration from environment
static void init_config() {
    if (verbose < 0) {
        const char* v = getenv("CUDA_MANAGED_VERBOSE");
        verbose = (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y')) ? 1 : 0;
    }
    if (disabled < 0) {
        const char* d = getenv("CUDA_MANAGED_DISABLE");
        disabled = (d && (d[0] == '1' || d[0] == 'y' || d[0] == 'Y')) ? 1 : 0;
    }
}

// Get the real CUDA functions
static void init_real_functions() {
    std::lock_guard<std::mutex> lock(init_mutex);

    // Runtime API
    if (real_cudaMalloc == nullptr) {
        real_cudaMalloc = (cudaError_t (*)(void**, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
        if (!real_cudaMalloc) {
            fprintf(stderr, "[CUDA_MANAGED] Failed to find real cudaMalloc: %s\n", dlerror());
        }
    }

    if (real_cudaMallocAsync == nullptr) {
        real_cudaMallocAsync = (cudaError_t (*)(void**, size_t, cudaStream_t))dlsym(RTLD_NEXT, "cudaMallocAsync");
    }

    if (real_cudaFree == nullptr) {
        real_cudaFree = (cudaError_t (*)(void*))dlsym(RTLD_NEXT, "cudaFree");
    }

    if (real_cudaFreeAsync == nullptr) {
        real_cudaFreeAsync = (cudaError_t (*)(void*, cudaStream_t))dlsym(RTLD_NEXT, "cudaFreeAsync");
    }

    // Driver API (used by cuBLAS and other libraries internally)
    if (real_cuMemAlloc == nullptr) {
        real_cuMemAlloc = (CUresult (*)(CUdeviceptr*, size_t))dlsym(RTLD_NEXT, "cuMemAlloc");
    }
    if (real_cuMemAlloc_v2 == nullptr) {
        real_cuMemAlloc_v2 = (CUresult (*)(CUdeviceptr*, size_t))dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    }
    if (real_cuMemFree == nullptr) {
        real_cuMemFree = (CUresult (*)(CUdeviceptr))dlsym(RTLD_NEXT, "cuMemFree");
    }
    if (real_cuMemFree_v2 == nullptr) {
        real_cuMemFree_v2 = (CUresult (*)(CUdeviceptr))dlsym(RTLD_NEXT, "cuMemFree_v2");
    }

    init_config();
}

// Update peak memory tracking
static void update_peak(size_t current) {
    size_t peak = peak_allocated.load();
    while (current > peak) {
        if (peak_allocated.compare_exchange_weak(peak, current)) {
            break;
        }
    }
}

extern "C" {

/*
 * Intercept cudaMalloc and replace with cudaMallocManaged
 */
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (real_cudaMalloc == nullptr) {
        init_real_functions();
    }

    // If disabled, use real cudaMalloc
    if (disabled) {
        return real_cudaMalloc(devPtr, size);
    }

    // Use cudaMallocManaged instead
    cudaError_t err = cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);

    if (err == cudaSuccess) {
        size_t current = total_allocated.fetch_add(size) + size;
        size_t count = num_allocs.fetch_add(1) + 1;
        num_managed_allocs.fetch_add(1);
        update_peak(current);

        if (verbose && size > 1024 * 1024) {  // Log allocations > 1MB
            fprintf(stderr, "[CUDA_MANAGED] cudaMalloc(%zu) -> cudaMallocManaged: %p "
                    "(total: %.2f GB, peak: %.2f GB, #%zu)\n",
                    size, *devPtr, current / 1e9, peak_allocated.load() / 1e9, count);
        }
    } else {
        if (verbose) {
            fprintf(stderr, "[CUDA_MANAGED] cudaMallocManaged(%zu) FAILED: %s\n",
                    size, cudaGetErrorString(err));
        }
    }

    return err;
}

/*
 * Intercept cudaMallocAsync and replace with cudaMallocManaged
 * (async allocation doesn't make sense for managed memory, but we handle it)
 */
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    if (real_cudaMallocAsync == nullptr) {
        init_real_functions();
    }

    // If disabled, use real cudaMallocAsync
    if (disabled && real_cudaMallocAsync) {
        return real_cudaMallocAsync(devPtr, size, stream);
    }

    // Use cudaMallocManaged instead (ignore stream, managed memory is sync)
    cudaError_t err = cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);

    if (err == cudaSuccess) {
        size_t current = total_allocated.fetch_add(size) + size;
        num_allocs.fetch_add(1);
        num_managed_allocs.fetch_add(1);
        update_peak(current);

        if (verbose && size > 1024 * 1024) {
            fprintf(stderr, "[CUDA_MANAGED] cudaMallocAsync(%zu, stream) -> cudaMallocManaged: %p\n",
                    size, *devPtr);
        }
    }

    return err;
}

/*
 * Intercept cudaFree - works the same for managed memory
 */
cudaError_t cudaFree(void* devPtr) {
    if (real_cudaFree == nullptr) {
        init_real_functions();
    }

    // cudaFree works for both regular and managed memory
    // We can't easily track the size being freed, so we don't update total_allocated
    // (This is a limitation - for accurate tracking, we'd need to maintain a map)

    return real_cudaFree(devPtr);
}

/*
 * Intercept cudaFreeAsync
 */
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    if (real_cudaFreeAsync == nullptr) {
        init_real_functions();
    }

    if (real_cudaFreeAsync) {
        return real_cudaFreeAsync(devPtr, stream);
    } else {
        // Fall back to sync free if async not available
        return real_cudaFree(devPtr);
    }
}

/*
 * Intercept cuMemAlloc (Driver API) - used by cuBLAS internally
 */
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    if (real_cuMemAlloc_v2 == nullptr) {
        init_real_functions();
    }

    if (disabled) {
        if (real_cuMemAlloc_v2) return real_cuMemAlloc_v2(dptr, bytesize);
        if (real_cuMemAlloc) return real_cuMemAlloc(dptr, bytesize);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    // Use cuMemAllocManaged instead
    CUresult err = cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);

    if (err == CUDA_SUCCESS) {
        size_t current = total_allocated.fetch_add(bytesize) + bytesize;
        size_t count = num_allocs.fetch_add(1) + 1;
        num_managed_allocs.fetch_add(1);
        update_peak(current);

        // Track allocation size for proper freeing
        {
            std::lock_guard<std::mutex> lock(alloc_map_mutex);
            allocation_sizes[(void*)*dptr] = bytesize;
        }

        if (verbose && bytesize > 1024 * 1024) {
            fprintf(stderr, "[CUDA_MANAGED] cuMemAlloc(%zu) -> cuMemAllocManaged: 0x%llx "
                    "(total: %.2f GB, peak: %.2f GB, #%zu)\n",
                    bytesize, (unsigned long long)*dptr, current / 1e9,
                    peak_allocated.load() / 1e9, count);
        }
    } else {
        if (verbose) {
            fprintf(stderr, "[CUDA_MANAGED] cuMemAllocManaged(%zu) FAILED: %d\n",
                    bytesize, (int)err);
        }
    }

    return err;
}

/*
 * Intercept cuMemAlloc_v2 (Driver API v2)
 */
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc(dptr, bytesize);  // Same implementation
}

/*
 * Intercept cuMemFree (Driver API)
 */
CUresult cuMemFree(CUdeviceptr dptr) {
    if (real_cuMemFree_v2 == nullptr) {
        init_real_functions();
    }

    // Update statistics if we tracked this allocation
    {
        std::lock_guard<std::mutex> lock(alloc_map_mutex);
        auto it = allocation_sizes.find((void*)dptr);
        if (it != allocation_sizes.end()) {
            total_allocated.fetch_sub(it->second);
            allocation_sizes.erase(it);
        }
    }

    if (real_cuMemFree_v2) return real_cuMemFree_v2(dptr);
    if (real_cuMemFree) return real_cuMemFree(dptr);
    return CUDA_ERROR_NOT_INITIALIZED;
}

/*
 * Intercept cuMemFree_v2 (Driver API v2)
 */
CUresult cuMemFree_v2(CUdeviceptr dptr) {
    return cuMemFree(dptr);  // Same implementation
}

/*
 * Get statistics (can be called from Python via ctypes)
 */
size_t cuda_managed_get_total_allocated() {
    return total_allocated.load();
}

size_t cuda_managed_get_peak_allocated() {
    return peak_allocated.load();
}

size_t cuda_managed_get_num_allocs() {
    return num_allocs.load();
}

size_t cuda_managed_get_num_managed_allocs() {
    return num_managed_allocs.load();
}

void cuda_managed_print_stats() {
    fprintf(stderr, "[CUDA_MANAGED] Stats: total=%.2f GB, peak=%.2f GB, "
            "allocs=%zu, managed=%zu\n",
            total_allocated.load() / 1e9,
            peak_allocated.load() / 1e9,
            num_allocs.load(),
            num_managed_allocs.load());
}

}  // extern "C"

// Print stats on library unload
__attribute__((destructor))
static void on_unload() {
    if (verbose && num_managed_allocs.load() > 0) {
        cuda_managed_print_stats();
    }
}
