#!/usr/bin/env python3
"""Test if LD_PRELOAD is inherited by spawn subprocess."""

import os
import multiprocessing as mp

def child_func():
    import sys
    print(f'Child PID: {os.getpid()}', file=sys.stderr)
    print(f'Child LD_PRELOAD: {os.environ.get("LD_PRELOAD", "NOT SET")}', file=sys.stderr)

    # Check if libcudamalloc_managed.so is loaded
    with open(f'/proc/{os.getpid()}/maps', 'r') as f:
        maps = f.read()
        if 'libcudamalloc_managed' in maps:
            print('Child: libcudamalloc_managed.so IS loaded', file=sys.stderr)
        else:
            print('Child: libcudamalloc_managed.so NOT loaded', file=sys.stderr)

    import torch
    # Try a smaller allocation first to see if hook works
    print(f'Child allocating 100MB tensor...', file=sys.stderr)
    x = torch.zeros(25 * 1024 * 1024, device='cuda')  # 100MB
    print(f'Child allocated: {x.device}, size={x.numel() * 4 / 1e9:.2f} GB', file=sys.stderr)
    print('Child done', file=sys.stderr)

if __name__ == '__main__':
    print(f'Parent LD_PRELOAD: {os.environ.get("LD_PRELOAD", "NOT SET")}')

    # Test with spawn (what vLLM uses)
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=child_func)
    p.start()
    p.join()
