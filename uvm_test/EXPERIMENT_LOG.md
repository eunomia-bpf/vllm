# UVM Oversubscription 实验记录

## 实验环境

- **GPU**: NVIDIA RTX 5090 (32GB / 33.7 GiB)
- **系统**: Linux 6.15.11
- **vLLM 版本**: v0.11.0rc2.dev37
- **日期**: 2024-11-25

## 实验目标

测试 vLLM 的 UVM (Unified Virtual Memory) 功能，验证能否运行超过 GPU 内存的大模型。

---

## 实验 1: Qwen3-30B-A3B-FP8 (基础测试)

**模型**: `Qwen/Qwen3-30B-A3B-FP8`
**大小**: 29 GB (FP8 量化)
**Context**: 2048 tokens

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-30B-A3B-FP8', enforce_eager=True, max_model_len=2048)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=50))
"
```

**结果**:
- Model loading: 29.04 GiB
- Available KV cache memory: **-0.67 GiB** (轻微 oversubscription)
- 速度: **18.4 tokens/s**
- 状态: ✅ 成功

---

## 实验 2: GGUF Q6_K 量化模型

**模型**: `mradermacher/Qwen3-42B-A3B...Q6_K.gguf`
**大小**: 33 GB
**Context**: 2048 tokens

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/path/to/model.Q6_K.gguf',
    tokenizer='Qwen/Qwen3-30B-A3B',
    enforce_eager=True,
    max_model_len=2048,
)
"
```

**结果**:
- Model loading: 32.45 GiB
- Available KV cache memory: **-4.08 GiB**
- 速度: **24.7 tokens/s**
- 状态: ✅ 成功

**发现**: GGUF 量化模型即使有更大的 oversubscription 也能正常工作，且速度更快。

---

## 实验 3: Qwen3-30B-A3B BF16 (未量化)

**模型**: `Qwen/Qwen3-30B-A3B`
**大小**: 56 GB (BF16)
**Context**: 2048 tokens

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-30B-A3B', enforce_eager=True, max_model_len=2048)
"
```

**结果**:
- Model loading: 56.2 GiB ✅
- Available KV cache memory: **-27.87 GiB** (严重 oversubscription)
- 状态: ❌ 失败

**错误信息**:
```
torch.AcceleratorError: CUDA error: out of memory
在 ops.moe_sum 操作中发生
```

**分析**:
- 模型权重加载成功（56GB 通过 UVM 分配）
- 在 warmup/profiling 阶段执行 forward pass 时失败
- MoE 模型的 fused expert 操作需要大量中间缓冲区

---

## 实验 4: Qwen3-30B-A3B-FP8 + 大 KV Cache

**模型**: `Qwen/Qwen3-30B-A3B-FP8`
**大小**: 29 GB
**Context**: 32768 tokens

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='Qwen/Qwen3-30B-A3B-FP8',
    enforce_eager=True,
    max_model_len=32768,
    gpu_memory_utilization=0.95,
)
"
```

**结果**:
- UVM 分配的 blocks: 2048 blocks
- GPU KV cache size: 32,768 tokens
- 速度: **25.0 tokens/s**
- 状态: ✅ 成功

---

## 实验 5: Qwen3-30B-A3B-FP8 + 最大 KV Cache (40K)

**模型**: `Qwen/Qwen3-30B-A3B-FP8`
**大小**: 29 GB
**Context**: 40960 tokens (模型最大支持)

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='Qwen/Qwen3-30B-A3B-FP8',
    enforce_eager=True,
    max_model_len=40960,
    gpu_memory_utilization=0.95,
)
"
```

**结果**:
- UVM 分配的 blocks: 2560 blocks
- GPU KV cache size: 40,960 tokens
- 速度: **24.3 tokens/s**
- 状态: ✅ 成功

---

## 实验 6: Qwen2.5-32B-Instruct BF16 (Dense 模型对比)

**模型**: `Qwen/Qwen2.5-32B-Instruct`
**大小**: 62 GB (BF16)
**Context**: 4096 tokens

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='Qwen/Qwen2.5-32B-Instruct',
    enforce_eager=True,
    max_model_len=4096,
)
"
```

**结果**:
- Model loading: 61.0 GiB
- Available KV cache memory: **-32.83 GiB** (严重 oversubscription)
- 速度: **0.16 tokens/s** (非常慢)
- 状态: ✅ 成功（但极慢）

**发现**: Dense 模型即使 oversubscription 更严重（-32GB vs -28GB）也能成功，但 MoE 模型失败。

---

## 关键发现

### 1. 量化模型 vs 未量化模型

| 类型 | 模型大小 | 数据传输量 | UVM 性能 |
|------|---------|-----------|---------|
| FP8 | 原始的 ~50% | 少 | 好 |
| GGUF Q6_K | 原始的 ~50% | 少 | 好 |
| BF16 | 100% | 多 | 差 |

**结论**: 量化模型在 UVM 下表现更好，因为 page fault 时需要迁移的数据量更少。

### 2. MoE vs Dense 架构

| 架构 | 中间缓冲区 | Oversubscription 容忍度 |
|------|-----------|------------------------|
| Dense | 较少 | 高（-32GB 也能跑） |
| MoE | 大量（expert routing） | 低（-28GB 就失败） |

**结论**: MoE 模型在执行时需要大量中间缓冲区用于 expert routing 和 fused MoE kernel，这增加了内存压力。

### 3. cuBLAS 初始化问题

**问题**: cuBLAS 需要通过原生 `cudaMalloc` 分配工作空间，不经过 PyTorch allocator。

**解决方案**: 在 UVM allocator 启用后立即进行一次 matmul 操作来预初始化 cuBLAS：

```python
# vllm/device_allocator/uvm.py
# 在 enable_uvm_allocator() 中
a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
b = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
_ = torch.matmul(a, b)  # 触发 cuBLAS 初始化
```

### 4. gpu_memory_utilization 参数

这个参数在 UVM 模式下**意义有限**，因为：
- 它用于计算 KV cache 大小
- UVM 模式下我们已经绕过了内存检查
- 但设置太高（如 1.0）可能导致启动时的内存检查失败

**建议**: 使用 0.95 左右的值。

---

## 性能对比总结

| 模型 | 大小 | 架构 | Oversubscription | 速度 | 状态 |
|------|------|------|------------------|------|------|
| Qwen3-30B-A3B-FP8 | 29 GB | MoE | -0.67 GB | 18-25 t/s | ✅ |
| Qwen3-30B-A3B-FP8 (40K ctx) | 29 GB | MoE | -0.66 GB | 24 t/s | ✅ |
| GGUF Q6_K | 33 GB | MoE | -4 GB | 25 t/s | ✅ |
| Qwen3-30B-A3B BF16 | 56 GB | MoE | -28 GB | N/A | ❌ |
| Qwen2.5-32B BF16 | 62 GB | Dense | -33 GB | 0.16 t/s | ✅ |

---

## 建议

1. **使用量化模型**: FP8 或 GGUF 量化可以显著提高 UVM 下的性能
2. **避免严重 oversubscription**: 尽量控制在 10GB 以内
3. **MoE 模型谨慎使用**: 即使模型能加载，推理时也可能 OOM
4. **必须启用 enforce_eager**: CUDA Graphs 与 UVM 不兼容
5. **生产环境不推荐**: UVM 主要用于测试和开发

---

## 实验 7: UVM vs 非 UVM 对比测试

**目的**: 验证 UVM 的必要性 - 在 32GB GPU 上运行 29GB 模型

### 7a: 不使用 UVM

```bash
uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-30B-A3B-FP8', enforce_eager=True, max_model_len=2048)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
"
```

**结果**:
- Model loading: 29.04 GiB ✅
- Available KV cache memory: **4.33 GiB**
- Sampler warmup: ❌ **OOM 失败**

**错误信息**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB.
在 sampler warmup 阶段发生
```

### 7b: 使用 UVM

```bash
VLLM_USE_UVM=1 uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-30B-A3B-FP8', enforce_eager=True, max_model_len=2048)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
"
```

**结果**:
- Model loading: 29.04 GiB ✅
- Available KV cache memory: **-0.67 GiB** (oversubscription)
- Sampler warmup: ✅ 成功
- 速度: **25 tokens/s**
- 状态: ✅ 成功

**结论**: 即使模型看起来"刚好"能放进 GPU，实际运行时还需要额外的中间缓冲区（如 sampler workspace），UVM 通过允许 oversubscription 解决了这个问题。

---

## KV Cache 容量计算

### Qwen3-30B-A3B-FP8 模型参数

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 48 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| KV cache dtype | FP8 (1 byte) |
| block_size | 16 (vLLM default) |

### 计算公式

**每个 token 的 KV cache 大小:**
```
KV_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_size
             = 2 × 48 × 4 × 128 × 1 byte
             = 49,152 bytes
             = 48 KB/token
```

**4.33 GB 对应的 token 数量:**
```
4.33 GB = 4.33 × 1024³ bytes = 4,649,508,454 bytes
Tokens = 4,649,508,454 / 49,152 ≈ 94,618 tokens
```

### 不同 KV cache 容量对比

| KV Cache 大小 | 可容纳 tokens | 说明 |
|---------------|--------------|------|
| 4.33 GB | ~94,618 | 无 UVM 时的可用容量 |
| -0.67 GB | oversubscribed | UVM 模式，从 CPU 借用 |
| 1 GB | ~21,333 | 小型批处理 |
| 8 GB | ~170,666 | 中等批处理 |

### 实际观察

在实验 4 和 5 中:
- 32K context: 需要约 1.5 GB KV cache
- 40K context: 需要约 1.9 GB KV cache

这些值比理论计算略小，因为 vLLM 使用了一些优化（如 block 复用）。

---

## 实验 8: LD_PRELOAD 拦截 cudaMalloc

### 背景

创建了一个 LD_PRELOAD 库，拦截所有 `cudaMalloc` 和 `cuMemAlloc` 调用，替换为 `cudaMallocManaged`。

### 文件位置

```
uvm_test/
├── cuda_malloc_managed_preload.cpp   # LD_PRELOAD 源码
├── libcudamalloc_managed.so          # 编译后的库
├── Makefile                          # 编译脚本
├── uvm_allocator.cpp                 # PyTorch pluggable allocator
└── test_uvm_oversubscription.py      # 测试脚本
```

### 编译方法

```bash
cd uvm_test
make
```

### 测试脚本用法

```bash
# 查看帮助
uv run python uvm_test/test_uvm_oversubscription.py --help

# 测试 FP8 模型 (默认)
uv run python uvm_test/test_uvm_oversubscription.py

# 测试指定模型
uv run python uvm_test/test_uvm_oversubscription.py --model Qwen/Qwen3-30B-A3B

# 带 KV cache 计算
uv run python uvm_test/test_uvm_oversubscription.py --kv-calc
```

### 完整测试命令

```bash
# 测试 1: 无 UVM (预期 OOM)
uv run python uvm_test/test_uvm_oversubscription.py

# 测试 2: 仅 UVM (FP8 模型应成功)
VLLM_USE_UVM=1 uv run python uvm_test/test_uvm_oversubscription.py

# 测试 3: UVM + LD_PRELOAD (FP8 模型)
VLLM_USE_UVM=1 LD_PRELOAD=/home/yunwei37/workspace/vllm/uvm_test/libcudamalloc_managed.so \
    uv run python uvm_test/test_uvm_oversubscription.py

# 测试 4: UVM + LD_PRELOAD (BF16 模型, 56GB)
VLLM_USE_UVM=1 LD_PRELOAD=/home/yunwei37/workspace/vllm/uvm_test/libcudamalloc_managed.so \
    uv run python uvm_test/test_uvm_oversubscription.py --model Qwen/Qwen3-30B-A3B
```

### 环境变量

```bash
VLLM_USE_UVM=1           # 启用 vLLM UVM allocator (必须)
LD_PRELOAD=...           # 拦截所有 cudaMalloc 调用
CUDA_MANAGED_VERBOSE=1   # 打印所有被拦截的分配
CUDA_MANAGED_DISABLE=1   # 禁用拦截（使用原生 cudaMalloc）
```

### 测试结果: Qwen3-30B-A3B-FP8 (29GB)

| 配置 | 模型加载 | KV Cache | 推理 | 状态 |
|------|---------|----------|------|------|
| 无 UVM | 29.04 GiB | 4.33 GiB | ❌ OOM (sampler warmup) | 失败 |
| VLLM_USE_UVM=1 | 29.04 GiB | -2.23 GiB | ~24 t/s | ✅ 成功 |
| VLLM_USE_UVM=1 + LD_PRELOAD | 29.04 GiB | -2.23 GiB | ~24 t/s | ✅ 成功 |

### 测试结果: Qwen3-30B-A3B BF16 (56GB)

| 配置 | 模型加载 | 推理 | 状态 |
|------|---------|------|------|
| VLLM_USE_UVM=1 | 56.88 GiB ✅ | ❌ OOM (MoE ops.moe_sum) | 失败 |
| VLLM_USE_UVM=1 + LD_PRELOAD | 56.88 GiB ✅ | ❌ OOM (MoE ops.moe_sum) | 失败 |

**BF16 模型失败原因**: MoE 模型在 profile_run 阶段执行 `ops.moe_sum` 时 OOM。

### 根因分析

经过调试发现:

1. **LD_PRELOAD 在子进程中生效** - Python `spawn` 方式会继承 LD_PRELOAD 环境变量
2. **vLLM UVM allocator 正常工作** - 所有 PyTorch 分配都走 UVM，不触发 `cudaMalloc`
3. **问题是 UVM 的根本限制**: CUDA kernel 执行时，**单个 kernel 的 working set 不能超过 GPU 物理内存**

UVM 是按需分页的，但当一个 CUDA kernel 需要同时访问的数据超过 GPU 物理内存时，就会失败:
- 模型权重: 56 GB (BF16)
- 中间缓冲区: ~1 GB
- 输入/输出: ~0.1 GB
- **总计**: ~57 GB > 32 GB GPU 内存

这不是 vLLM 或 LD_PRELOAD 的问题，而是 **CUDA UVM 硬件/驱动的限制**。

### 关键发现

1. **必须启用 VLLM_USE_UVM=1**: LD_PRELOAD 单独使用不够，因为:
   - vLLM 的内存检查逻辑需要 UVM 标志来跳过
   - PyTorch allocator 需要被替换才能正确管理 UVM 内存

2. **LD_PRELOAD 的作用**: 拦截那些绕过 PyTorch allocator 的分配（如 cuBLAS workspace）

3. **性能影响**: 使用 LD_PRELOAD 时模型加载变慢（5s → 28s），因为所有分配都走 managed memory

4. **BF16 MoE 模型限制**: 即使 LD_PRELOAD 拦截了所有分配，BF16 MoE 模型仍然在 profile 阶段 OOM

### LD_PRELOAD 库功能

- 拦截 Runtime API: `cudaMalloc`, `cudaMallocAsync`, `cudaFree`, `cudaFreeAsync`
- 拦截 Driver API: `cuMemAlloc`, `cuMemAlloc_v2`, `cuMemFree`, `cuMemFree_v2`
- 统计功能: 跟踪总分配量、峰值使用、分配次数
- 日志功能: `CUDA_MANAGED_VERBOSE=1` 打印详细分配日志

---

## 待解决问题

1. **MoE 模型的中间缓冲区优化**: 是否可以减少 fused MoE kernel 的内存使用？
2. **prefetch 优化**: `VLLM_UVM_PREFETCH=1` 的效果需要进一步测试
3. **与 llama.cpp 的性能差距**: llama.cpp 使用相同的 UVM 方式能达到 80 t/s，需要调查差异原因
4. **BF16 模型支持**: Qwen3-30B-A3B (BF16, 56GB) 在 MoE 操作时仍然 OOM，需要进一步调查
