# vLLM UVM (Unified Virtual Memory) Support

This directory contains documentation and test scripts for vLLM's UVM memory oversubscription feature.

## Overview

UVM (Unified Virtual Memory) allows vLLM to allocate more GPU memory than physically available by using CPU memory as backing store. When GPU memory is exhausted, data is automatically migrated between CPU and GPU via page faults.

**Key Benefits:**
- Run models larger than GPU memory
- Test large models on smaller GPUs
- Development and debugging of large models

**Trade-offs:**
- Significant performance overhead due to page faults
- Unpredictable latency
- Not recommended for production use

## Quick Start

```bash
# Enable UVM and run vLLM
VLLM_USE_UVM=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-30B-A3B-FP8 --enforce-eager

# Or in Python
VLLM_USE_UVM=1 python -c "
from vllm import LLM
llm = LLM(model='Qwen/Qwen3-30B-A3B-FP8', enforce_eager=True)
"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_UVM` | `0` | Enable UVM allocator (`1` to enable) |
| `VLLM_UVM_PREFETCH` | `0` | Enable prefetching to GPU after allocation |
| `VLLM_UVM_VERBOSE` | `0` | Log large allocations (>100MB) |

## Requirements

1. **CUDA Toolkit** with UVM support
2. **enforce_eager=True** - CUDA Graphs are incompatible with UVM
3. **Quantized models recommended** - FP8/GGUF models perform much better than BF16

## Performance Results (RTX 5090, 32GB)

| Model | Size | Oversubscription | Speed | Status |
|-------|------|------------------|-------|--------|
| Qwen3-30B-A3B-FP8 | 29 GB | -0.67 GB | 25 tokens/s | ✓ |
| Qwen3-30B-A3B-FP8 (40K ctx) | 29 GB | -0.66 GB | 24 tokens/s | ✓ |
| GGUF Q6_K (42B MoE) | 33 GB | -4 GB | 25 tokens/s | ✓ |
| Qwen3-30B-A3B BF16 | 56 GB | -28 GB | N/A | ✗ OOM |

**Key Findings:**
- Quantized models (FP8, GGUF) work well with UVM
- BF16 models with severe oversubscription (>20GB) may fail
- MoE models require more intermediate memory than Dense models

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PyTorch                                 │
├─────────────────────────────────────────────────────────────┤
│              CUDAPluggableAllocator                          │
├─────────────────────────────────────────────────────────────┤
│              uvm_allocator.so                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ uvm_malloc  │  │  uvm_free   │  │  Statistics │          │
│  │             │  │             │  │   API       │          │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘          │
│         │                │                                   │
│         ▼                ▼                                   │
│  cudaMallocManaged    cudaFree                              │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDA UVM Driver                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Page Migration Engine                     │  │
│  │   GPU Memory ◄──────────────────────► CPU Memory      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `csrc/uvm_allocator.cpp` - C++ UVM allocator using cudaMallocManaged
- `vllm/device_allocator/uvm.py` - Python wrapper and PyTorch integration
- `vllm/envs.py` - Environment variable definitions
- `vllm/env_override.py` - Auto-enable UVM on startup

## Test Scripts

### test_uvm_vllm.py
Comprehensive benchmark script for testing UVM with various models.

```bash
# List available preset models
uv run python uvm_test/test_uvm_vllm.py --list-models

# Test with a specific preset
VLLM_USE_UVM=1 uv run python uvm_test/test_uvm_vllm.py --preset gguf-q6

# Run benchmarks
VLLM_USE_UVM=1 uv run python uvm_test/test_uvm_vllm.py --benchmark
```

### test_uvm_oversubscription.py
Basic oversubscription test.

```bash
VLLM_USE_UVM=1 uv run python uvm_test/test_uvm_oversubscription.py
```

### test_uvm.py
Unit tests for UVM allocator functionality.

```bash
VLLM_USE_UVM=1 pytest uvm_test/test_uvm.py -v
```

## Known Limitations

1. **CUDA Graphs incompatible** - Must use `enforce_eager=True`
2. **cuBLAS initialization** - cuBLAS is pre-initialized after UVM is enabled
3. **MoE models** - May require more memory for expert routing buffers
4. **Severe oversubscription** - >50% oversubscription may cause OOM during inference

## Troubleshooting

### "Can't swap an already initialized allocator"
UVM must be enabled before any CUDA operations. Ensure `VLLM_USE_UVM=1` is set before importing vLLM.

### "CUDA out of memory" during inference
- Try using a quantized model (FP8, GGUF)
- Reduce `max_model_len`
- Use a smaller model

### Slow performance
- This is expected with UVM due to page faults
- Quantized models have less data to migrate and perform better
- Consider using `VLLM_UVM_PREFETCH=1` for potential improvement

## Building

The UVM allocator is built automatically with vLLM:

```bash
# Development build
pip install -e . --no-build-isolation

# Or use CMake presets
cmake --preset default
cmake --build build --target uvm_allocator
```

## References

- [CUDA Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [PyTorch CUDAPluggableAllocator](https://pytorch.org/docs/stable/notes/cuda.html#using-custom-memory-allocators-for-cuda)
- [vLLM Documentation](https://docs.vllm.ai/)
