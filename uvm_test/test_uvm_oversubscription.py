#!/usr/bin/env python3
"""
Test UVM oversubscription - allocate more memory than GPU has.

This script demonstrates that UVM allows allocating more GPU memory
than physically available by using CPU memory as backing store.

Usage:
    # Test FP8 model without UVM (will OOM)
    uv run python uvm_test/test_uvm_oversubscription.py

    # Test FP8 model with UVM (should succeed)
    VLLM_USE_UVM=1 uv run python uvm_test/test_uvm_oversubscription.py

    # Test BF16 model with UVM + LD_PRELOAD
    VLLM_USE_UVM=1 LD_PRELOAD=./uvm_test/libcudamalloc_managed.so \
        uv run python uvm_test/test_uvm_oversubscription.py --model Qwen/Qwen3-30B-A3B
"""

import os
import time
import torch

def print_kv_cache_calculation():
    """
    Print KV cache capacity calculation for Qwen3-30B-A3B-FP8.
    """
    print("\n" + "=" * 60)
    print("KV Cache Calculation for Qwen3-30B-A3B-FP8")
    print("=" * 60)

    # Model parameters (from HuggingFace config)
    num_layers = 48
    num_kv_heads = 4
    head_dim = 128
    dtype_bytes = 1  # FP8

    # Calculate KV cache per token
    # K + V = 2 × num_layers × num_kv_heads × head_dim × dtype_size
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    print(f"Model parameters:")
    print(f"  num_hidden_layers: {num_layers}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  KV cache dtype: FP8 ({dtype_bytes} byte)")

    print(f"\nKV cache per token:")
    print(f"  = 2 × {num_layers} × {num_kv_heads} × {head_dim} × {dtype_bytes}")
    print(f"  = {kv_per_token:,} bytes ({kv_per_token/1024:.1f} KB)")

    print(f"\nCapacity for different memory sizes:")
    for gb in [1, 4.33, 8, 16, 32]:
        bytes_available = gb * 1024**3
        tokens = int(bytes_available / kv_per_token)
        print(f"  {gb:5.2f} GB -> ~{tokens:>10,} tokens ({tokens//1024:>5}K context)")


def test_vllm_model(model: str = "Qwen/Qwen3-30B-A3B-FP8", max_model_len: int = 2048):
    """
    Test UVM with vLLM model.

    This demonstrates that even when a model "fits" in GPU memory,
    UVM is needed because runtime buffers (sampler, etc.) need extra space.
    """
    try:
        from vllm.device_allocator.uvm import is_uvm_enabled
        uvm_enabled = is_uvm_enabled()
    except ImportError:
        uvm_enabled = os.environ.get("VLLM_USE_UVM", "0").lower() in ("1", "true", "yes")

    # Check LD_PRELOAD status
    ld_preload = os.environ.get("LD_PRELOAD", "")
    has_ld_preload = "libcudamalloc_managed" in ld_preload

    print("\n" + "=" * 60)
    print(f"vLLM Model Test: {model}")
    print("=" * 60)
    print(f"UVM enabled: {uvm_enabled}")
    print(f"LD_PRELOAD: {has_ld_preload}")
    if has_ld_preload:
        print(f"  -> {ld_preload}")

    from vllm import LLM, SamplingParams

    print(f"\nLoading model: {model}")
    print(f"Max model length: {max_model_len}")

    try:
        llm = LLM(
            model=model,
            enforce_eager=True,  # Required for UVM
            max_model_len=max_model_len,
        )

        print("Model loaded successfully!")

        # Run inference
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(max_tokens=50, temperature=0.7)

        print("\nRunning inference...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        # Metrics
        output_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        elapsed = end_time - start_time
        tokens_per_sec = num_tokens / elapsed

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Generated: {output_text[:80]}...")
        print(f"Tokens: {num_tokens}, Time: {elapsed:.2f}s, Speed: {tokens_per_sec:.1f} t/s")

    except Exception as e:
        print("\n" + "=" * 60)
        print("FAILED!")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {e}")

        if not uvm_enabled:
            print("\nThis is expected without UVM!")
            print("The model loads but OOMs during warmup/inference")
            print("\nTry: VLLM_USE_UVM=1 uv run python uvm_test/test_uvm_oversubscription.py --vllm-test")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test UVM oversubscription with vLLM models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-FP8",
                        help="Model to test (default: Qwen/Qwen3-30B-A3B-FP8)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Max model length (default: 2048)")
    parser.add_argument("--kv-calc", action="store_true",
                        help="Print KV cache calculation")
    args = parser.parse_args()

    test_vllm_model(model=args.model, max_model_len=args.max_model_len)

    if args.kv_calc:
        print_kv_cache_calculation()