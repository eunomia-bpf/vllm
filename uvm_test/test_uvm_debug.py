#!/usr/bin/env python3
"""Debug UVM memory allocation for BF16 model."""

import os
import torch

# Enable UVM before any vLLM import
os.environ["VLLM_USE_UVM"] = "1"

import vllm
from vllm.device_allocator.uvm import is_uvm_enabled, get_uvm_stats

print(f"UVM enabled: {is_uvm_enabled()}")
print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'NOT SET')}")

# Check GPU memory
def print_gpu_memory(label=""):
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{label}] GPU: Free={free/1e9:.2f}GB, Total={total/1e9:.2f}GB, "
          f"PyTorch Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    try:
        stats = get_uvm_stats()
        print(f"[{label}] UVM Stats: {stats}")
    except:
        pass

print_gpu_memory("Initial")

# Try loading the model
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-30B-A3B"

print(f"\nLoading model: {MODEL}")

try:
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=2048,
    )
    print_gpu_memory("After load")

    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print(f"Generated: {outputs[0].outputs[0].text}")

except Exception as e:
    print(f"\nFailed: {type(e).__name__}: {e}")
    print_gpu_memory("After error")
