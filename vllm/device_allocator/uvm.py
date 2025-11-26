# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA UVM (Unified Virtual Memory) allocator for vLLM.

This module provides a custom CUDA memory allocator that uses cudaMallocManaged
to enable memory oversubscription - allocating more GPU memory than physically
available by using CPU memory as backing store.

Usage:
    # Enable UVM before any CUDA operations
    from vllm.device_allocator.uvm import enable_uvm_allocator

    # Option 1: Call explicitly
    enable_uvm_allocator()

    # Option 2: Set environment variable VLLM_USE_UVM=1

Example:
    VLLM_USE_UVM=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf

Note:
    UVM allows running models larger than GPU memory, but with significant
    performance overhead due to page migration between CPU and GPU.
    This is useful for:
    - Testing large models on smaller GPUs
    - Development and debugging
    - Scenarios where memory > throughput

Warning:
    UVM mode is NOT recommended for production use due to:
    - Unpredictable latency from page faults
    - Lower throughput compared to native GPU memory
    - Potential compatibility issues with CUDA graphs
"""

import ctypes
import os
from contextlib import contextmanager
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global state
_uvm_enabled = False
_uvm_lib: Optional[ctypes.CDLL] = None


def find_uvm_allocator_library() -> Optional[str]:
    """
    Find the UVM allocator shared library.

    Returns:
        Path to the library if found, None otherwise.
    """
    # Try to find the library in common locations
    import vllm

    vllm_path = os.path.dirname(vllm.__file__)

    # Possible library names and locations
    lib_names = [
        "uvm_allocator.abi3.so",  # ABI3 build (most common)
        "uvm_allocator.so",
        "libuvm_allocator.so",
    ]
    search_paths = [
        vllm_path,  # Installed location
        os.path.join(vllm_path, ".."),  # Development location
        os.path.dirname(os.path.abspath(__file__)),  # Same directory
    ]

    for path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                return lib_path

    # Try to find using /proc/self/maps (similar to cumem allocator)
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "uvm_allocator" in line:
                    start = line.index("/")
                    return line[start:].strip()
    except (FileNotFoundError, ValueError):
        pass

    return None


def load_uvm_library() -> ctypes.CDLL:
    """
    Load the UVM allocator library and set up function signatures.

    Returns:
        CDLL handle to the library.

    Raises:
        RuntimeError: If library cannot be found or loaded.
    """
    lib_path = find_uvm_allocator_library()

    if lib_path is None:
        raise RuntimeError(
            "UVM allocator library not found. "
            "Please ensure vLLM is built with UVM support enabled. "
            "The library should be at: <vllm_path>/uvm_allocator.so"
        )

    logger.info("Loading UVM allocator library from: %s", lib_path)

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load UVM allocator library: {e}") from e

    # Define function signatures for statistics API
    lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
    lib.uvm_get_allocated_bytes.argtypes = []

    lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t
    lib.uvm_get_peak_allocated_bytes.argtypes = []

    lib.uvm_get_num_allocs.restype = ctypes.c_size_t
    lib.uvm_get_num_allocs.argtypes = []

    lib.uvm_get_num_frees.restype = ctypes.c_size_t
    lib.uvm_get_num_frees.argtypes = []

    lib.uvm_reset_peak_stats.restype = None
    lib.uvm_reset_peak_stats.argtypes = []

    lib.uvm_reset_all_stats.restype = None
    lib.uvm_reset_all_stats.argtypes = []

    lib.uvm_set_prefetch.restype = None
    lib.uvm_set_prefetch.argtypes = [ctypes.c_int]

    lib.uvm_set_verbose.restype = None
    lib.uvm_set_verbose.argtypes = [ctypes.c_int]

    return lib


def enable_uvm_allocator(
    enable_prefetch: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Enable the UVM allocator for PyTorch CUDA operations.

    This must be called before any CUDA memory allocations occur.

    Args:
        enable_prefetch: If True, prefetch allocations to GPU after allocation.
            This can improve performance but uses more GPU memory.
        verbose: If True, log large allocations to stderr.

    Returns:
        True if UVM was enabled successfully, False otherwise.

    Raises:
        RuntimeError: If UVM allocator library is not available.
    """
    global _uvm_enabled, _uvm_lib

    if _uvm_enabled:
        logger.warning("UVM allocator is already enabled")
        return True

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, UVM allocator cannot be enabled")
        return False

    # Load the library
    _uvm_lib = load_uvm_library()

    # Configure the allocator
    _uvm_lib.uvm_set_prefetch(1 if enable_prefetch else 0)
    _uvm_lib.uvm_set_verbose(1 if verbose else 0)

    # Find library path for PyTorch pluggable allocator
    lib_path = find_uvm_allocator_library()
    if lib_path is None:
        raise RuntimeError("UVM allocator library not found")

    try:
        # Create and enable the pluggable allocator
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            lib_path, 'uvm_malloc', 'uvm_free'
        )
        torch.cuda.memory.change_current_allocator(allocator)
        _uvm_enabled = True
        logger.info("UVM allocator enabled successfully")

        # Pre-initialize cuBLAS after UVM is enabled
        # This ensures cuBLAS workspace is allocated via UVM
        # and won't fail later due to native memory exhaustion
        try:
            logger.info("Pre-initializing cuBLAS with UVM...")
            a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
            b = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            del a, b
            torch.cuda.empty_cache()
            logger.info("cuBLAS pre-initialized successfully with UVM")
        except Exception as e:
            logger.warning("Failed to pre-initialize cuBLAS: %s", e)

        return True

    except Exception as e:
        logger.error("Failed to enable UVM allocator: %s", e)
        raise RuntimeError(f"Failed to enable UVM allocator: {e}") from e


def is_uvm_enabled() -> bool:
    """Check if UVM allocator is currently enabled."""
    return _uvm_enabled


def get_uvm_stats() -> dict:
    """
    Get UVM memory statistics.

    Returns:
        Dictionary with memory statistics:
        - allocated_bytes: Current allocated bytes
        - peak_bytes: Peak allocated bytes
        - num_allocs: Total number of allocations
        - num_frees: Total number of frees
    """
    if not _uvm_enabled or _uvm_lib is None:
        return {
            "allocated_bytes": 0,
            "peak_bytes": 0,
            "num_allocs": 0,
            "num_frees": 0,
            "enabled": False,
        }

    return {
        "allocated_bytes": _uvm_lib.uvm_get_allocated_bytes(),
        "peak_bytes": _uvm_lib.uvm_get_peak_allocated_bytes(),
        "num_allocs": _uvm_lib.uvm_get_num_allocs(),
        "num_frees": _uvm_lib.uvm_get_num_frees(),
        "enabled": True,
    }


def reset_uvm_peak_stats() -> None:
    """Reset peak memory statistics to current allocation."""
    if _uvm_lib is not None:
        _uvm_lib.uvm_reset_peak_stats()


def reset_uvm_all_stats() -> None:
    """Reset all memory statistics."""
    if _uvm_lib is not None:
        _uvm_lib.uvm_reset_all_stats()


def prefetch_to_device(tensor: torch.Tensor, device: int = 0) -> None:
    """
    Prefetch a UVM tensor to the specified GPU device.

    This can improve performance by proactively moving data to GPU
    before it's needed.

    Args:
        tensor: The tensor to prefetch (must be on CUDA)
        device: Target GPU device ID
    """
    if not _uvm_enabled:
        return

    if not tensor.is_cuda:
        logger.warning("Cannot prefetch non-CUDA tensor")
        return

    # Use cudaMemPrefetchAsync via the library
    if _uvm_lib is not None:
        import ctypes
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        _uvm_lib.uvm_prefetch(
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            ctypes.c_int(device),
            ctypes.c_void_p(stream)
        )


def set_preferred_location(tensor: torch.Tensor, device: int) -> None:
    """
    Set the preferred location for a UVM tensor.

    Args:
        tensor: The tensor to configure
        device: Preferred device (-1 for CPU, >= 0 for GPU)
    """
    if not _uvm_enabled or _uvm_lib is None:
        return

    if not tensor.is_cuda:
        return

    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    _uvm_lib.uvm_advise(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_int(2),  # cudaMemAdviseSetPreferredLocation
        ctypes.c_int(device)
    )


@contextmanager
def uvm_memory_tracking():
    """
    Context manager for tracking UVM memory usage within a block.

    Example:
        with uvm_memory_tracking() as tracker:
            # Do some operations
            pass
        print(f"Peak memory: {tracker['peak_bytes'] / 1e9:.2f} GB")
    """
    if not _uvm_enabled:
        yield {"enabled": False}
        return

    reset_uvm_peak_stats()
    stats_before = get_uvm_stats()

    result = {
        "enabled": True,
        "start_bytes": stats_before["allocated_bytes"],
    }

    try:
        yield result
    finally:
        stats_after = get_uvm_stats()
        result["end_bytes"] = stats_after["allocated_bytes"]
        result["peak_bytes"] = stats_after["peak_bytes"]
        result["delta_bytes"] = (
            stats_after["allocated_bytes"] - stats_before["allocated_bytes"]
        )
        result["allocations"] = (
            stats_after["num_allocs"] - stats_before["num_allocs"]
        )


def maybe_enable_uvm_from_env() -> bool:
    """
    Enable UVM allocator if VLLM_USE_UVM environment variable is set.

    This is called automatically during vLLM initialization.

    Returns:
        True if UVM was enabled, False otherwise.
    """
    if os.environ.get("VLLM_USE_UVM", "0").lower() in ("1", "true", "yes"):
        enable_prefetch = os.environ.get(
            "VLLM_UVM_PREFETCH", "0"
        ).lower() in ("1", "true", "yes")
        verbose = os.environ.get(
            "VLLM_UVM_VERBOSE", "0"
        ).lower() in ("1", "true", "yes")

        try:
            enable_uvm_allocator(enable_prefetch=enable_prefetch, verbose=verbose)
            return True
        except Exception as e:
            logger.error("Failed to enable UVM allocator from environment: %s", e)
            return False

    return False
