#!/usr/bin/env python3
"""
Test MoE (Mixture of Experts) operations under UVM oversubscription.
"""

import os
import sys

# Must import vllm first to enable UVM
import vllm

import torch
from vllm.device_allocator.uvm import is_uvm_enabled, get_uvm_stats

print(f"UVM enabled: {is_uvm_enabled()}")

def test_moe_only():
    """Test just the MoE layer."""
    # Allocate memory to simulate model loading (56 GB for BF16 model)
    target_alloc = int(56 * 1e9)
    model_weights = torch.empty(target_alloc // 2, device='cuda', dtype=torch.bfloat16)
    print(f"Allocated {target_alloc/1e9:.2f} GB to simulate model weights")
    print(f"UVM stats: {get_uvm_stats()}")

    # Check GPU memory
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: Free={free/1e9:.2f} GB, Total={total/1e9:.2f} GB")

    # Now test the actual fused_moe function
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # MoE parameters for Qwen3-30B-A3B
    num_tokens = 8192  # Larger batch (same as profiling)
    num_experts = 128
    top_k = 8
    hidden_size = 2048
    intermediate_size = 768

    print(f"\nTesting fused_moe with:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")

    # Create input tensors
    hidden_states = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size, device='cuda', dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device='cuda', dtype=torch.bfloat16)

    # Create routing weights (top-k experts per token)
    topk_weights = torch.rand(num_tokens, top_k, device='cuda', dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device='cuda', dtype=torch.int32)

    print(f"\nInput tensors allocated. UVM stats: {get_uvm_stats()}")

    try:
        print("\nRunning fused_experts...")
        output = fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
        )
        torch.cuda.synchronize()
        print(f"SUCCESS! Output shape: {output.shape}")
        print(f"Final UVM stats: {get_uvm_stats()}")

    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_full_model():
    """Test full model loading and inference."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("Testing full model: Qwen/Qwen3-30B-A3B")
    print("=" * 60)

    try:
        llm = LLM(
            model='Qwen/Qwen3-30B-A3B',
            enforce_eager=True,
            max_model_len=2048,
        )

        outputs = llm.generate(['Hello'], SamplingParams(max_tokens=10))
        print(f"SUCCESS! Generated: {outputs[0].outputs[0].text}")

    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Test full model")
    args = parser.parse_args()

    if args.full:
        test_full_model()
    else:
        test_moe_only()
