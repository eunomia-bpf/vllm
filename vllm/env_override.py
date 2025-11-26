# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

# see https://github.com/vllm-project/vllm/issues/10480
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1

# Enable UVM (Unified Virtual Memory) allocator if requested
# This MUST happen before any CUDA memory allocations
# UVM allows memory oversubscription but with significant performance overhead
if os.environ.get("VLLM_USE_UVM", "0").lower() in ("1", "true", "yes"):
    try:
        from vllm.device_allocator.uvm import enable_uvm_allocator
        enable_prefetch = os.environ.get(
            "VLLM_UVM_PREFETCH", "0"
        ).lower() in ("1", "true", "yes")
        verbose = os.environ.get(
            "VLLM_UVM_VERBOSE", "0"
        ).lower() in ("1", "true", "yes")
        enable_uvm_allocator(enable_prefetch=enable_prefetch, verbose=verbose)
        logger.warning(
            "UVM allocator is enabled. This allows memory oversubscription "
            "but has significant performance overhead. "
            "NOT recommended for production use."
        )
    except Exception as e:
        logger.error("Failed to enable UVM allocator: %s", e)
