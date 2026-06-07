# Photon-AMD

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![AMD ROCm](https://img.shields.io/badge/ROCm-6.10%2B-red)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-AMD%20Instinct%20MI300-9cf)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

**High-efficiency LLM inference engine for AMD GPUs — pipelined decoding with GPU bubble elimination via ROCm/HIP.**

**面向 AMD GPU 的高效 LLM 推理引擎 —— 基于 ROCm/HIP 的流水化解码，消除 GPU 气泡。**

</div>

---

## Overview · 概述

**Photon-AMD** is an educational and practical implementation of Moondream's [Photon](https://moondream.ai/p/photon) inference engine, ported to AMD ROCm/HIP.
**Photon-AMD** 是 Moondream [Photon](https://moondream.ai/p/photon) 推理引擎的教育与实践实现，已移植至 AMD ROCm/HIP。

It demonstrates how to eliminate the **GPU bubble** — idle time where the GPU waits for CPU bookkeeping between decode steps — achieving **up to 56% higher decode throughput**.
它演示了如何消除 **GPU 气泡** —— 解码步骤之间 GPU 等待 CPU 完成簿记工作的空闲时间 —— 可提升高达 **56% 的解码吞吐量**。

This project serves two purposes. 本项目兼具两大用途：

1. **Learning resource · 学习资源**: Deep-dive analysis of Photon's three core mechanisms with mathematical cost models and empirical validation. 对 Photon 三大核心机制的深度分析，附有数学成本模型与实验验证。
2. **Reference implementation · 参考实现**: Clean, well-documented AMD ROCm code that can serve as a starting point for production inference engines. 清晰、文档完备的 ROCm 代码，可作为 AMD GPU 上生产推理引擎的起点。

### Background · 背景

Photon is Moondream's purpose-built inference engine that achieves **~2× faster inference than vLLM** on comparable workloads. It was originally implemented for NVIDIA GPUs (CUDA). This project analyses the Photon architecture from the blog post ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble), ports the three-mechanism pipelined decode loop to AMD ROCm/HIP, and validates the cost model on AMD hardware (gfx942 / MI300).
Photon 是 Moondream 专为视觉语言模型打造的推理引擎，在可比工作负载上比 vLLM 快约 **2×**。它最初为 NVIDIA GPU (CUDA) 实现。本项目解析博客文章 ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) 中的 Photon 架构，将三机制流水化解码循环移植至 AMD ROCm/HIP，并在 AMD 硬件 (gfx942 / MI300) 上验证了成本模型。

---

## Three Core Mechanisms · 三大核心机制

| # | Mechanism · 机制 | File · 文件 | Description · 描述 |
|---|----------|------|-------------|
| 1 | **Ping-pong slots · 乒乓槽位** | `decode_slot.py` | Two alternating buffer sets so step N+1 can run while step N's results are copied to CPU. 两套交替缓冲区使步骤 N+1 的计算可与步骤 N 的结果回传重叠。 |
| 2 | **Forward now, sample later · 先行推理，后采样** | `pipeline.py` | Forward launches before previous commit; sampling waits on constrained-decode mask. 前向在上一轮提交之前发射；采样等待约束解码掩码就绪后执行。 |
| 3 | **Zombies · 僵尸序列** | `zombie.py` | Finished requests ride one extra forward instead of requiring mid-flight cancellation. 已完成请求多跑一次前向作为"僵尸"，避免中途取消的复杂逻辑。 |

The speedup formula · 加速比公式：

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

| Hardware · 硬件 | Streams · 并发 | Blocking · 阻塞 | Pipelined · 流水线 | Speedup · 加速 |
|----------|------|----------|-----------|---------|
| RTX 3090 | 32 | 11.74 ms | 10.52 ms | **+11.6%** |
| B200 | 32 | 5.55 ms | 3.98 ms | **+35.4%** |
| MI300X (AMD) | 2 | 0.47 ms/tok | 0.30 ms/tok | **+56.1%** |

---

## Quick Start · 快速开始

Requirements · 环境要求：**AMD GPU** (CDNA3 / gfx942 / MI300), **ROCm** 6.10+ / PyTorch 2.5+, **Python** 3.10–3.12.

```bash
git clone https://github.com/Yunle-Lee/Babe-Photon.git
cd Babe-Photon
pip install -e ".[dev]"
```

Run benchmark · 运行基准测试：
```bash
python -m photon_amd.benchmark                 # default: 8 requests × 128 tokens
python examples/bench_real_model.py            # real PyTorch Transformer + Photon pipeline
```

Run real model demo · 运行真实模型：
```bash
# Requires: pip install llama-cpp-python
python examples/demo_real_model.py             # Gemma-4 12B through Photon pipeline
```

Programmatic usage · 编程调用：
```python
from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch

def my_decode(slot: DecodeSlot, batch: Batch, stream):
    with torch.cuda.stream(stream):
        # ... your model forward pass ...
        pass

config = PhotonConfig(max_batch_size=32)
engine = PhotonEngine(config, PipelineCallbacks(do_decode=my_decode))
engine.submit(prompt_token_ids=[1, 2, 3], max_new_tokens=256)
engine.run()
for seq_id, tokens in engine.collect_results().items():
    print(f"Seq {seq_id}: {len(tokens)} tokens")
```

---

## Architecture · 架构

```
                         ┌──────────────────────────┐
                         │     PhotonEngine          │
                         │  pipeline orchestration   │
                         │  流水线编排                │
                         ├──────────────────────────┤
                         │  StreamManager            │
                         │  compute + copy streams   │
                         │  计算流 + 拷贝流            │
                         ├──────────────────────────┤
                         │  Scheduler  ·  调度器      │
                         │  tick: launch→commit→     │
                         │        finalize           │
                         ├──────────┬───────────────┤
                         │  Slot 0  │  Slot 1       │
                         │  buffers │  buffers      │
                         │  graphs  │  graphs       │
                         │  events  │  events       │
                         ├──────────┴───────────────┤
                         │  GraphManager (HIP)       │
                         │  ZombieTracker            │
                         └──────────────────────────┘
```

Stream model · 流模型：
```
compute stream ──┬── Forward(N) ──────┬── Forward(N+1) ──┬── ...
 计算流           │                     │                    │
copy stream    ──┴── (idle) ───────────┴── D2H copy(N) ────┴── ...
 拷贝流              等待                   等待
                  step_done_event        step_done_event
```

Key insight · 核心洞见：the D2H copy goes on a separate stream, so the next forward can start immediately — the bubble is eliminated. D2H 拷贝在独立的拷贝流上执行，因此下一个前向推理可以立即开始 —— 气泡被消除。

---

## Project Structure · 项目结构

```
Babe-Photon/
├── README.md                          # this file · 本文件（中英双语）
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # package metadata · 包元信息
├── .github/workflows/tests.yml        # CI
│
├── docs/                              # English docs · 英文文档
│   ├── 01_photon_overview.md          #   architecture overview · 架构总览
│   ├── 02_mechanism_pingpong.md       #   mechanism 1: ping-pong slots
│   ├── 03_mechanism_fsl.md            #   mechanism 2: forward / sample split
│   ├── 04_mechanism_zombies.md        #   mechanism 3: zombie lifecycle
│   ├── 05_gpu_bubble_analysis.md      #   cost model & math · 成本模型
│   ├── 06_amd_rocm_porting.md         #   CUDA→ROCm porting guide · 移植指南
│   └── 07_blog_analysis.md            #   blog post analysis · 博客分析
│   └── zh/                            # 中文文档 · Chinese docs
│       ├── 01_photon_overview.md
│       ├── 02_mechanism_pingpong.md
│       ├── 03_mechanism_fsl.md
│       ├── 04_mechanism_zombies.md
│       ├── 05_gpu_bubble_analysis.md
│       ├── 06_amd_rocm_porting.md
│       └── 07_blog_analysis.md
│
├── photon_amd/                        # core implementation · 核心实现
│   ├── __init__.py                    #   public API
│   ├── config.py                      #   configuration
│   ├── stream_manager.py              #   HIP stream & event mgmt
│   ├── decode_slot.py                 #   ping-pong slot buffers
│   ├── graph.py                       #   HIP graph capture & replay
│   ├── zombie.py                      #   zombie lifecycle (mechanism 3)
│   ├── scheduler.py                   #   batch assembly & tick
│   ├── pipeline.py                    #   launch / commit / finalize
│   ├── benchmark.py                   #   blocking vs pipelined benchmark
│   └── adapter.py                     #   llama-cpp-python model adapter
│
├── examples/
│   ├── demo_pipelined.py              #   blocking vs pipelined demo
│   ├── demo_real_model.py             #   real Gemma-4 12B inference
│   ├── bench_real_gpu.py              #   GPU matmul proxy benchmark
│   └── bench_real_model.py            #   PyTorch Transformer benchmark
│
└── tests/
    ├── test_zombie.py                 #   zombie state machine
    ├── test_scheduler.py              #   scheduler & batch
    └── test_slots.py                  #   GPU buffer & slot (GPU required)
```

---

## Benchmarks · 基准测试

### Real model inference · 真实模型推理 (Gemma-4 12B)

```
[2+2]    ~5 tok/s → "4"
[who]    ~6 tok/s → "I am Gemma 4, a large language model developed by Google."
[haiku]  ~7 tok/s → "Lines of code flow fast, Logic builds a complex world, Errors fade away."
```

### Pipeline speedup · 流水线加速 (synthetic GPU matmul proxy)

| Load · 负载 | Blocking · 阻塞 | Pipelined · 流水线 | Speedup · 加速 |
|------|----------|-----------|------|
| 2 req × 128 tok | 2129 tok/s | 3323 tok/s | **+56.1%** |
| 4 req × 128 tok | 6013 tok/s | 5992 tok/s | −0.3% |

The speedup is largest when CPU bookkeeping is a meaningful fraction of step time (fewer concurrent requests → larger bubble → more to hide). The benefit shrinks at high batch sizes where the GPU is already saturated — exactly as the cost model predicts.
加速比在 CPU 簿记占步时较大比例时最显著（并发越少 → 气泡越大 → 越值得隐藏）。高批量下 GPU 已满载，加速比趋近于零 —— 与成本模型预测完全一致。

---

## Testing · 测试

```bash
pytest tests/ -v                    # all tests · 全部测试
pytest tests/ -v -k "not gpu"      # CPU-only · 仅 CPU
pytest tests/ -v -k gpu            # GPU tests (AMD + ROCm) · GPU 测试
```

---

## Porting Notes · 移植说明

Photon-AMD runs on AMD GPUs through PyTorch's ROCm backend, which provides transparent CUDA→HIP mapping.
Photon-AMD 通过 PyTorch 的 ROCm 后端在 AMD GPU 上运行，利用其透明的 CUDA→HIP 映射。

| CUDA API | ROCm/HIP | PyTorch API |
|----------|----------|-------------|
| `cudaStream_t` | `hipStream_t` | `torch.cuda.Stream` |
| `cudaEvent_t` | `hipEvent_t` | `torch.cuda.Event` |
| `cudaGraph_t` | `hipGraph_t` | `torch.cuda.CUDAGraph` |

Key differences · 主要差异：HIP graph capture requires stricter fixed-shape / pre-allocated buffer patterns (already satisfied by Photon's design). Runtime `hipMalloc` can trigger device-wide sync — all buffers MUST be pre-allocated. `float16` is recommended over `bfloat16` on gfx942 for optimal bandwidth.
HIP 图捕获对固定形状 / 预分配缓冲区的模式要求更严格（Photon 的设计天然符合）。运行时 `hipMalloc` 可能触发设备级同步 —— 所有缓冲区必须在初始化时预分配。建议使用 `float16` 而非 `bfloat16` 以获得 gfx942 上的最佳内存带宽利用率。

---

## References · 参考文献

- **Moondream Blog · 博客**: ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) (2026-06-04)
- **Moondream Docs · 文档**: [Running Locally](https://docs.moondream.ai/running-locally)
- **Kestrel** (Photon reference): [github.com/m87-labs/kestrel](https://github.com/m87-labs/kestrel)
- **Moondream**: [github.com/m87-labs/moondream](https://github.com/m87-labs/moondream)
- **AMD ROCm**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)

---

## License · 许可证

Apache 2.0 — see [LICENSE](./LICENSE). All credit for the Photon architecture and the three-mechanism design goes to **Moondream (M87 Labs)**. This project is an independent educational implementation and port.
Apache 2.0 —— 详见 [LICENSE](./LICENSE)。Photon 架构及三大机制设计的全部功劳归 **Moondream (M87 Labs)** 所有。本项目为独立的教育实现与移植。

---

## Appendix A: 中文深度解析文档

# Photon Architecture: An Academic Analysis

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04)
> **Repository**: [m87-labs/kestrel](https://github.com/m87-labs/kestrel)
> **Author's note**: This document is a deep-dive study of Photon for the
> purpose of creating an AMD ROCm/HIP port.  All credit for the original
> design goes to Moondream (M87 Labs).

---

## 1. What is Photon?

Photon is Moondream's purpose-built inference engine for vision-language
models.  Unlike generic engines (vLLM, TensorRT-LLM), Photon exploits
model-specific properties of Moondream to achieve higher throughput and
lower latency.  The headline figure is **~2× faster than vLLM** on
comparable Moondream workloads and **34 ms end-to-end inference on an
H100**.

The engine handles the full serving lifecycle: image preprocessing,
prefill (prompt + image encoding), autoregressive decode, structured
output generation (spatial grounding), and token streaming.

### 1.1 Key design principles

1. **Model-specific, not generic.** Photon only serves Moondream, so
   the scheduler, memory manager, image preprocessor, and streaming
   behaviour can all be tuned to the model's exact characteristics.

2. **No allocations on the hot path.** All GPU buffers, streams, and
   events are pre-allocated at engine init.  This avoids runtime
   ``hipMalloc`` / ``cudaMalloc`` calls, which can trigger device-wide
   synchronisation and introduce bubbles.

3. **Pipelined decode loop.** CPU bookkeeping (plan, launch, commit) is
   overlapped with GPU computation, eliminating the "GPU bubble" — idle
   time where the GPU waits for the CPU to finish housekeeping.

4. **HIP/CUDA graph capture.** The entire decode forward is captured as
   a single command buffer (graph) once per batch size and replayed,
   reducing hundreds of per-step kernel launches to one graph replay.

---

## 2. The GPU Bubble Problem

### 2.1 Why GPUs idle during decode

In autoregressive text generation, each token depends on all previous
tokens.  The decode loop is a tight round-trip:

```
CPU: plan → launch →                → wait → commit → ...
GPU:          [forward] → idle → idle →       [forward] →
```

The GPU forward for one token is **small**: a few hundred MB of weights
streamed through the compute units, taking ~2–10 ms depending on model
size and GPU memory bandwidth.  The CPU housekeeping (scheduling,
metadata upload, token detokenization, constrained-decode mask
construction) is a fixed cost per step, typically 0.5–2 ms.

When `T_forward < T_bookkeeping`, the GPU spends a significant fraction
of each step idle — the **GPU bubble**.

### 2.2 Why it matters

The bubble grows with GPU speed.  As memory bandwidth increases (e.g.,
H100 → B200) or models get smaller (Moondream 0.5B), the bubble becomes
a larger fraction of step time.  From the blog post's measurements:

| Hardware | Streams | T_block (ms) | T_pipe (ms) | Speedup |
|----------|---------|--------------|-------------|---------|
| RTX 3090 | 1       | 5.44         | 5.10        | +6.5%   |
| RTX 3090 | 32      | 11.74        | 10.52       | +11.6%  |
| B200     | 1       | 3.11         | 2.63        | +17.6%  |
| B200     | 32      | 5.55         | 3.98        | +35.4%  |

The speedup grows with GPU capability: +12% on a 3090 but +35% on a
B200 at the same batch size.  Pipelining is *insurance against the GPU
getting faster*.

### 2.3 Mathematical model

In blocking mode:

$$T_{\text{step}} = T_{\text{forward}} + T_{\text{bookkeeping}}$$

In pipelined mode:

$$T_{\text{step}} \approx \max(T_{\text{forward}}, T_{\text{bookkeeping}})$$

The speedup factor:

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

where $z \approx 1/L$ is the *zombie tax* — the cost of running an
extra forward for finished sequences.  At $L \approx 110$, $z \approx
1\%$ at batch=1 and vanishes at batch sizes ≫ 1.

---

## 3. System Architecture

### 3.1 Component diagram

```
                     ┌─────────────────────────┐
                     │     PhotonEngine         │
                     │  (pipeline orchestration)│
                     ├─────────────────────────┤
                     │  StreamManager           │
                     │  (compute + copy streams)│
                     ├─────────────────────────┤
                     │  Scheduler               │
                     │  (tick: launch→commit→   │
                     │   finalize)              │
                     ├───────────┬─────────────┤
                     │  Slot 0   │  Slot 1     │
                     │  (buffers,│  (buffers,  │
                     │   graphs, │   graphs,   │
                     │   events) │   events)   │
                     ├───────────┴─────────────┤
                     │  GraphManager (HIP)      │
                     │  ZombieTracker            │
                     └─────────────────────────┘
```

### 3.2 Stream model

```
compute_stream ──┬── Forward (step N) ──┬── Forward (step N+1) ──┬── ...
                 │                      │                        │
copy_stream    ──┴── (idle) ────────────┴── D2H copy (N) ───────┴── D2H copy (N+1) ...
                    wait on                wait on
                    step_done_event        step_done_event
```

- **compute_stream** (priority=0): All GPU forwards serialise here,
  preserving sequential token dependencies.
- **copy_stream** (priority=-1): Device-to-host copies of sampled
  outputs.  Waits on ``step_done_event`` recorded on compute_stream
  when a step's outputs are ready.

The two streams are independent, so the next forward can start
immediately while the previous step's copy is in flight — the bubble
is eliminated.

### 3.3 Per-tick lifecycle

```
Tick N:
  LAUNCH    batch for step N    (slot = N % 2)
  COMMIT    batch for step N-1  (slot = (N-1) % 2, via D2H copy)
  FINALIZE  batch for step N    (constrained-decode mask upload)
```

The three phases are detailed in the Mechanisms documents.

---

## 4. Three Mechanisms

See the dedicated documents for deep dives:

| Mechanism | Document | Summary |
|-----------|----------|---------|
| 1. Ping-pong slots | [02_mechanism_pingpong.md](./02_mechanism_pingpong.md) | Two alternating buffer sets prevent data races between pipelined steps. |
| 2. Forward now, sample later | [03_mechanism_fsl.md](./03_mechanism_fsl.md) | Forward launches before commit; sampling waits on mask from commit. |
| 3. Zombies | [04_mechanism_zombies.md](./04_mechanism_zombies.md) | Finished requests ride one extra forward instead of being cancelled mid-flight. |

---

## 5. AMD ROCm Porting Considerations

See [06_amd_rocm_porting.md](./06_amd_rocm_porting.md) for detailed
notes on porting the NVIDIA (CUDA) Photon implementation to AMD ROCm/HIP.

Key points:
- PyTorch's ROCm backend provides transparent CUDA→HIP mapping for
  streams, events, graphs, and pinned memory.
- HIP graph capture is supported and stable on gfx942 (MI300) but has
  stricter requirements than CUDA graphs (no dynamic shapes, no host
  callbacks).
- ``float16`` is the recommended dtype for memory-bandwidth-bound
  workloads on MI300.
- Memory allocation at runtime can trigger device-wide sync on HIP;
  all buffers MUST be pre-allocated.

---

## References

1. Moondream Blog: ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) (2026-06-04)
2. Moondream Docs: [Running Moondream Locally](https://docs.moondream.ai/running-locally)
3. Kestrel repository: [m87-labs/kestrel](https://github.com/m87-labs/kestrel)
4. NVIDIA: [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
5. AMD: [ROCm Documentation](https://rocm.docs.amd.com/)
# Mechanism 1: Ping-Pong Slots

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "Mechanism 1: ping-pong slots"

---

## 1. Problem Statement

A decode step needs GPU buffers: input staging, attention KV cache
references, forward output (logits), and a place to land the sampled
token.  If we only have one set of buffers, they stay in use until the
step is done, and we cannot start the next step until the current one
finishes.

To pipeline two steps, the second step needs its own working set —
otherwise it can overwrite the results of the first step before the
CPU has read them.

## 2. Solution

Allocate **two** complete sets of decode buffers ("slots") and alternate
between them, ping-pong style:

```
Step N     → Slot 0 (compute: forward + sample)
Step N+1   → Slot 1 (compute: forward + sample, while CPU commits Slot 0)
Step N+2   → Slot 0 (compute: forward + sample, while CPU commits Slot 1)
Step N+3   → Slot 1 (and so on...)
```

### 2.1 What goes in a slot

Each ``DecodeSlot`` bundles:

| Buffer | Location | Shape | Purpose |
|--------|----------|-------|---------|
| ``decode_token_ids`` | GPU | ``[max_batch]`` | Token ids input to forward |
| ``logits`` | GPU | ``[max_batch, vocab]`` | Forward output scores |
| ``hidden_last`` | GPU | ``[max_batch, hidden_dim]`` | Last hidden state (spatial decoder) |
| ``sampled_ids`` | GPU | ``[max_batch]`` | Sampled token id per row |
| ``sampled_logprobs`` | GPU | ``[max_batch]`` | Log-prob of sampled token |
| ``fa3_page_table`` | GPU | ``[max_batch, pages]`` | Paged-KV page table |
| ``fa3_seqused_k`` | GPU | ``[max_batch]`` | Per-sequence KV lengths |
| ``batch_idx`` | Pinned host | ``[max_batch]`` | Sequence batch indices |
| ``input_pos`` | Pinned host | ``[max_batch]`` | Token positions |
| ``disallow_mask`` | Pinned host | ``[max_batch, vocab]`` | Constrained-decode mask |

### 2.2 Pinned memory

Page-locked (pinned) host memory allows the GPU's DMA engine to copy data
without blocking the CPU.  Without pinning, every D2H transfer would
require the driver to first copy into a pinned staging buffer, serialising
the copy and the next kernel launch.

On AMD ROCm, ``torch.Tensor.pin_memory()`` maps to ``hipHostMalloc``,
identical semantics to CUDA's ``cudaHostAlloc``.

### 2.3 Slot lifecycle

```
┌─────────┐    launch     ┌─────────┐    D2H copy     ┌─────────┐
│  FREE   │ ────────────► │  IN USE  │ ────────────► │ COMMIT  │
└─────────┘               └─────────┘                └─────────┘
     ▲                                                     │
     │                                                     │
     └─────────────────────────────────────────────────────┘
                        commit done → FREE
```

A slot is "in use" from the moment it is launched until its
``commit_done_event`` is recorded (meaning the CPU has finished reading
from its pinned host buffer).  A slot can only be relaunched once it
returns to the FREE state.

### 2.4 WHY NOT GPU PARALLELISM?

The two slots do **not** put their forwards on separate streams for
concurrent execution.  Both use the **same** compute stream.  The slots
only exist so the CPU can process one slot's results while the GPU runs
the other slot's forward — they provide *CPU/GPU overlap*, not *GPU/GPU
parallelism*.

## 3. The Asynchronous D2H Copy

The key insight that makes ping-pong work: the D2H copy of sampled
outputs goes on the **copy stream**, not the compute stream.

```
compute_stream: ── Forward (Slot 0) ────────────── Forward (Slot 1) ── ...
                              │
                      step_done_event
                              │
copy_stream:    ──────────────┴── D2H copy (Slot 0) ── ...
```

The ``step_done_event`` is recorded on the compute stream when a slot's
forward outputs are ready.  The copy stream waits on this event before
starting the D2H copy, ensuring correctness.  But the compute stream
itself does NOT wait — the next forward can start immediately.

## 4. AMD ROCm Implementation Notes

The ``photon_amd.decode_slot`` module implements this pattern with:

- ``CpuGpuBuffer``: A paired pinned-host + device-memory buffer with
  ``copy_host_to_device`` and ``copy_device_to_host`` methods that
  accept a stream for asynchronous transfers.
- ``DecodeMetaBuffers``: The per-slot pinned host metadata (batch_idx,
  input_pos, disallow_mask).
- ``DecodeSlot``: The full bundle, created once at engine init via
  ``create_decode_slot()``.

All GPU buffers are pre-allocated at init time with fixed shapes to
satisfy HIP graph capture requirements.

## 5. References

1. Photon source: ``kestrel/models/moondream/decode_slot.py``
2. Blog post: [Mechanism 1 diagram](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-1-ping-pong-slots)
3. NVIDIA: [CUDA Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
4. AMD: [HIP Stream Management](https://rocm.docs.amd.com/projects/HIP/en/latest/programming_guide.html#stream-management)
# Mechanism 2: Forward Now, Sample Later

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "Mechanism 2: forward now, sample later"

---

## 1. The Dependency Puzzle

The next forward does not depend on anything the CPU does with the last
token — the forward only needs the token ID, which is already on the GPU.
So in principle, the forward can run ahead.

But *some* things about the next step do depend on the last step's
committed result:

1. **Batch membership**: If a request finished at step t, it should not
   be in step t+1. This is handled by Mechanism 3 (zombies).

2. **Constrained-decode mask**: Moondream's spatial skills (``point``,
   ``detect``, ``segment``) restrict which tokens the model may produce
   at each step.  The mask for step t+1 depends on the token sampled at
   step t — which the CPU hasn't committed yet when we want to launch
   step t+1.

## 2. Solution: Split Forward from Sampling

The key insight: the dependency is in **sampling**, not in the forward.
The forward itself needs no mask — it just computes logits for the
entire vocabulary.

The solution is to split each step into two phases that run at different
pipeline stages:

```
┌─────────────────────────────────────────────────────────┐
│  Tick N                                                  │
│                                                          │
│  LAUNCH:   Forward(step N) → [scores]                    │
│            (no mask needed — scores are unbiased)        │
│                                                          │
│  COMMIT:   Wait for token(step N-1) to land on CPU.      │
│            Advance state → build mask(step N).           │
│                                                          │
│  FINALIZE: Apply mask(step N) → Sample(step N).          │
│            (mask is now current because commit is done)   │
└─────────────────────────────────────────────────────────┘
```

This establishes **"commit-before-finalize" ordering**: the forward runs
immediately, but sampling waits until the previous step's commit has
made the per-sequence state current.

## 3. Unified Handling

The blog post describes how this elegantly handles both plain text and
constrained decoding with a single loop:

### 3.1 Plain text (no constrained decoding)

- No mask needed.
- Forward AND sampling can both run a step ahead.
- Both happen in the LAUNCH phase.

### 3.2 Constrained decoding (structured output)

- Forward runs a step ahead (in LAUNCH).
- Sampling waits on the previous commit (in FINALIZE).
- The commit phase builds the mask from the updated state,
  then FINALIZE applies it and samples.

### 3.3 Why this is elegant

No special-casing is needed.  The same scheduler code handles both:
- LAUNCH always runs the forward.
- COMMIT always processes the previous step's result.
- FINALIZE always does the mask-dependent work.

For plain text, FINALIZE is a no-op (mask is empty).  For constrained
sequences, FINALIZE waits on the commit that just happened.

## 4. The Disallow Mask

Moondream's constrained decoding uses a *disallow mask*: a boolean
tensor `[batch, vocab]` where `True` means "force this token's logit
to negative infinity before sampling."

The mask for each sequence is built from that sequence's current decode
state — which tokens it has emitted so far in the structured output.
For example:
- A `point` output always emits a coordinate `(x, y)`.
- A `detect` output walks an `x, y, width, height` cycle.

The mask for step t+1 is computed at the end of step t's commit, when
the sequence state is current.  It is then uploaded to the GPU slot
(via async H2D on the copy stream) so it is ready when FINALIZE runs.

## 5. Performance Implications

From the blog post's "It only pays once the bubble is actually hideable":

> We caught a bug where the pipelined numbers came out at *blocking*
> speed, traced to an accidental synchronous copy while building the
> constrained-decode mask.  Moving it to the copy stream was worth +11%
> on the 3090 and +34% on the B200.

The lesson: **any synchronisation on the hot path kills the pipelining
gain**.  Even a single synchronous copy within the critical path
destroys the overlap.

## 6. AMD ROCm Implementation Notes

In our ``photon_amd`` implementation:

- ``PipelineCallbacks.do_sample`` is called during ``_launch()`` for
  plain text (mask-independent sampling) or during ``_finalize()`` for
  constrained decoding.
- The disallow mask is stored in ``DecodeMetaBuffers.disallow_mask``
  and uploaded to GPU via the copy stream asynchronously.
- ``slot.mask_ready_event`` signals when the mask upload completes,
  so the sampling kernel can wait on it.

## 7. References

1. Photon source: ``kestrel/models/moondream/tokenizer.py`` (structured decode logic)
2. Blog post: [Mechanism 2 diagram](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-2-forward-now-sample-later)
3. NVIDIA: [Guided Decoding with Logits Processor](https://huggingface.co/docs/transformers/main_classes/logits)
# Mechanism 3: Zombies — Finalize Early, Release Late

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "Mechanism 3: zombies: finalize early, release late"

---

## 1. The Problem

In Mechanism 2, we noted that *batch membership* for step t+1 depends on
step t's committed result.  If a sequence finishes at step t (EOS or
length cap), it should not be in step t+1's batch.

But step t+1 was already launched **before** step t was committed — the
whole point of pipelining.  So the finished sequence is *already baked
into step t+1's forward*.  You cannot un-launch GPU work.

## 2. The Naive Option: Mid-Flight Cancellation

One could imagine cancelling the row mid-flight:
- Set the sequence's token to a "don't care" value.
- Or skip it in the attention kernel.

But this requires special-case logic in every kernel (attention, MLP,
sampling, commit), creating a "thicket of cancellation special cases."

## 3. The Elegant Solution: Let It Ride

Photon's approach is simpler: **let the finished sequence ride as a
zombie for one extra forward**.

```
Step t:     Sequence X is alive → forward computes real result
            Commit detects EOS → X is marked "finalized"
            Result is emitted to caller
            BUT: X's KV pages are NOT released

Step t+1:   X was already in the batch → forward includes X as "zombie"
            Commit sees X is finalized → SKIP, no token appended
            Release X's inflight ref

Refcount=0: X's KV pages and LoRA slot are recycled
```

The zombie occupies a batch slot and writes some KV cache entries that
nobody will ever read — but that's a small price to pay.

## 4. The Per-Sequence State Machine

Each sequence carries two extra fields:

| Field | Type | Meaning |
|-------|------|---------|
| ``finalized`` | ``bool`` | ``True`` after EOS or length cap detected. |
| ``inflight_refs`` | ``int`` | 0, 1, or 2 — how many in-flight forwards reference this sequence. |

State transitions:

```
IDLE (0 refs, not finalized)
  │
  ├── launch ──► ACTIVE (1 ref)
  │                │
  │                ├── launch ──► ACTIVE (2 refs, pipelined)
  │                │                │
  │                │                ├── commit (EOS) ──► ZOMBIE (2 refs, finalized)
  │                │                │                      │
  │                │                │                      ├── commit (skip)
  │                │                │                      └── release ──► ZOMBIE (1 ref)
  │                │                │
  │                │                ├── commit (normal) ──► ACTIVE (1 ref)
  │                │                └── release ──► ACTIVE (1 ref)
  │                │
  │                ├── commit (EOS) ──► ZOMBIE (1 ref, finalized)
  │                │                      │
  │                │                      └── release ──► IDLE (recycled)
  │                │
  │                └── commit (normal) ──► ACTIVE (1 ref)
  │
  └── release ──► IDLE (recycled if finalized, no-op otherwise)
```

## 5. Zombie Tax Analysis

From the blog post, the *zombie tax* $z$ is the fraction of wasted
forwards:

$$z \approx \frac{1}{L \cdot B}$$

where:
- $L$ = average generated tokens per request
- $B$ = batch size

### 5.1 At batch = 1

One wasted forward per $L$ tokens: $z \approx 1/L$.  For $L \approx 110$,
that is ~1% overhead — negligible.

### 5.2 At batch > 1

The zombie is just **one extra row** in a step that is already streaming
the full model weights through the compute units.  In the memory-bandwidth-bound
regime of decode, streaming the weights dominates, and the extra row adds
almost no cost.  The tax effectively vanishes at batch sizes ≫ 1.

### 5.3 Empirical validation

From the blog post:

| Hardware | Streams | L | Predicted | Observed |
|----------|---------|---|-----------|----------|
| 3090     | 1       | 104 | +5.7%     | +6.5%    |
| 3090     | 32      | 113 | +11.1%    | +11.6%   |
| B200     | 1       | 115 | +17.2%    | +17.6%   |
| B200     | 32      | 104 | +39.1%    | +35.4%   |

The predicted speedup (which accounts for zombie tax) closely matches
the observed values, confirming the model.

## 6. Resource Reclamation

Resources are **not released** when a sequence first finishes.  They are
only released when ``inflight_refs`` reaches 0:

- **KV cache pages**: Returned to the page allocator's free list.
- **LoRA slot**: Returned to the LoRA slot pool.
- **Batch slot**: Freed for a new sequence in the next schedule cycle.

This release-late policy is the price of not having mid-flight
cancellation logic.

## 7. AMD ROCm Implementation Notes

In ``photon_amd/zombie.py``:

- ``ZombieState``: The per-sequence fields (``finalized``, ``inflight_refs``,
  ``kv_page_start``, ``kv_page_count``, ``lora_slot``).
- ``ZombieTracker``: Central manager called at three hook points:
  - ``launch_ref()`` — increment refs when a sequence enters a batch.
  - ``commit()`` — detect EOS, return "finalized" or "skip".
  - ``release()`` — decrement refs, recycle when done.
- ``should_skip()`` — query whether a row is a zombie.

The scheduler calls these hooks in ``_assemble_batch()`` (launch ref),
``_commit()`` (commit), and ``_recycle_done()`` (release).

## 8. References

1. Blog post: [Mechanism 3 diagram](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-3-zombies-finalize-early-release-late)
2. Photon source: ``kestrel/models/moondream/zombie.py`` (approximate path)
3. Related: [Reference counting in garbage collection](https://en.wikipedia.org/wiki/Reference_counting)
# The GPU Bubble: A Mathematical Analysis

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04),
> Section "A cost model for the bubble"

---

## 1. Defining the Bubble

A decode step has three distinct pieces of work:

| Piece | Location | Description | Time |
|-------|----------|-------------|------|
| **Forward** | GPU | Weight-streaming matmuls (attention projections, MLP). Memory-bandwidth bound at decode. | $T_f$ |
| **Sampling** | GPU → CPU | Constrained-decode mask application, argmax/sample, spatial decode, D2H copy. | $T_s$ |
| **Bookkeeping** | CPU | Plan (batch assembly), launch (graph replay trigger), commit (token append, state advance, mask build). | $T_b$ |

### 1.1 Blocking Mode

```
Time →
CPU: [plan][launch] ──────────────── [sync] [commit][plan][launch] ────────────────
GPU:                [forward][sample] =idle=                  [forward][sample]
                    ◄──────── T_f + T_s ────────►
                                                 ◄── T_b ──►  ← GPU BUBBLE
```

Total step time: $T_{\text{block}} = T_f + T_s + T_b$

The GPU idle time is $T_b$ — the bubble.

### 1.2 Pipelined Mode

```
Time →
CPU: [plan][launch] ───────────────── [commit][plan][launch] ───────────────── [commit]
GPU:                [forward][sample]                       [forward][sample]
                    ◄────── T_f + T_s ──────►              ◄────── T_f + T_s ──►
```

Total step time: $T_{\text{pipe}} = \max(T_f + T_s, T_b)$

The GPU idle time is $0$ (assuming $T_f + T_s \ge T_b$).

## 2. The Speedup Formula

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

where **$z$** is the zombie tax from Mechanism 3:

$$z \approx \frac{1}{L} \quad \text{(at batch = 1)}$$

$$z \approx 0 \quad \text{(at batch ≫ 1)}$$

### 2.1 Two competing forces

1. **Bubble hiding** ($T_{\text{block}} / T_{\text{pipe}}$): The direct
   benefit of overlapping CPU bookkeeping with GPU computation.

2. **Zombie tax** ($z$): The indirect cost of letting finished requests
   ride one extra forward.  At batch=1 it is $1/L$ (~1%).  At large
   batch sizes, the cost amortises to near zero because the extra row
   is in a memory-bandwidth-bound step that already streams all the
   weights.

## 3. Why the Bubble Grows

The forward time $T_f$ scales with model weight size and GPU memory
bandwidth:

$$T_f \approx \frac{W}{B} \cdot \frac{1}{N_{\text{CU}}}$$

where:
- $W$ = total model weights in bytes
- $B$ = GPU memory bandwidth (bytes/s)
- $N_{\text{CU}}$ = number of active compute units

As GPU memory bandwidth increases (H100: 3.35 TB/s → B200: 8 TB/s) or
models get smaller (Moondream 2B → 0.5B), $T_f$ shrinks but $T_b$ stays
constant.  The bubble becomes a **larger fraction** of the total step
time.

### 3.1 Concrete example

| Model | GPU | W (GB) | B (TB/s) | $T_f$ (ms) | $T_b$ (ms) | Bubble % |
|-------|-----|--------|----------|------------|------------|----------|
| Moondream 2B | H100 | ~4 | 3.35 | ~1.2 | 1.5 | 55.6% |
| Moondream 2B | B200 | ~4 | 8.0 | ~0.5 | 1.5 | 75.0% |
| Moondream 0.5B | H100 | ~1 | 3.35 | ~0.3 | 1.5 | 83.3% |

The bubble is **dominant** for small models on fast GPUs — exactly the
regime where Moondream operates.

## 4. Prefill in the Same Pipeline

Prefill (processing a new request's prompt + image) is also a forward
pass, but over many tokens at once — it is compute-bound, not
memory-bandwidth-bound, so $T_f$ is larger and $T_s$ is more expensive.

Photon uses the **same two-slot pipeline for prefill and decode**.
A prefill is just a ``kind="prefill"`` launch that uses the same slot
and compute stream.  Because the pipeline only cares that a slot is
free, not what kind of work last used it, a prefill and a decode can
be transparently interleaved.

This matters most when outputs are short — a request that emits only
three tokens spends almost all its time in prefill, and a workload of
many short requests is effectively a stream of prefills with a little
decode sprinkled in.  Sharing one pipeline lets the prefill CPU
bookkeeping (admission, tokenisation) overlap with decode forwards
and vice versa.

## 5. Validation

From the blog post's empirical measurements on a B200:

| Mode | $T_f$ (ms) | $T_s$ (ms) | Period (ms) |
|------|-----------|-----------|-------------|
| 1 stream | 2.45 | 0.14 | 2.63 |
| 8 streams | 3.12 | 0.14 | 3.30 |
| 32 streams | 3.80 | 0.14 | 3.98 |

$T_f + T_s \approx \text{period}$, meaning the GPU is busy for
essentially the entire period — the bubble is eliminated.  The
remaining GPU idle is under 0.05 ms per step.

### 5.1 Blocking vs pipelined comparison

| Hardware | Streams | Blocking (ms) | Pipelined (ms) | Speedup |
|----------|---------|---------------|----------------|---------|
| RTX 3090 | 1 | 5.44 | 5.10 | +6.5% |
| RTX 3090 | 8 | 7.52 | 6.97 | +7.8% |
| RTX 3090 | 32 | 11.74 | 10.52 | +11.6% |
| B200 | 1 | 3.11 | 2.63 | +17.6% |
| B200 | 8 | 4.04 | 3.30 | +21.9% |
| B200 | 32 | 5.55 | 3.98 | +35.4% |

Three observations:
1. **The win grows with GPU speed** — +12% on 3090 but +35% on B200.
2. **Zombie tax is real but small** — ~1% at batch=1, essentially 0 at
   batch > 1.
3. **It only pays when the bubble is hideable** — any synchronous
   operation on the hot path kills the gain.

## 6. References

1. Blog post: [A cost model for the bubble](https://moondream.ai/blog/popping-the-gpu-bubble#a-cost-model-for-the-bubble)
2. NVIDIA: [GPU Memory Bandwidth](https://www.nvidia.com/en-us/data-center/a100/)
3. AMD: [MI300 Memory Bandwidth](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
# Porting Photon from NVIDIA CUDA to AMD ROCm

> A practical guide to porting Photon's inference engine from NVIDIA GPUs
> to AMD Instinct accelerators (gfx942 / MI300).

---

## 1. Overview

Photon (and its reference implementation in the ``kestrel`` repository)
is written against NVIDIA CUDA APIs.  This document catalogues every
CUDA-specific surface area and maps it to the equivalent AMD ROCm/HIP
API, noting where the two diverge.

The good news: **PyTorch provides a transparent compatibility layer.**
Most PyTorch code that runs on CUDA runs unchanged on ROCm because
PyTorch's CUDA API functions translate to HIP calls under the hood.
The areas that require attention are:

1. Stream and event management
2. GPU graph capture
3. Pinned memory
4. Kernel launch overhead
5. Memory allocation patterns
6. Hardware-specific performance characteristics

---

## 2. CUDA → HIP/ROCm Mapping

### 2.1 Streams

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaStream_t`` | ``hipStream_t`` | Wrapped by ``torch.cuda.Stream`` |
| ``torch.cuda.Stream()`` | ``torch.cuda.Stream()`` | Identical API |
| ``stream.synchronize()`` | ``stream.synchronize()`` | Identical |
| Priority | Priority | Both support 0 (default) and -1 (higher) |

**Verdict**: No code changes needed.  The PyTorch Stream API is fully
compatible.

### 2.2 Events

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaEvent_t`` | ``hipEvent_t`` | Wrapped by ``torch.cuda.Event`` |
| ``event.record(stream)`` | ``event.record(stream)`` | Identical |
| ``stream.wait_event(event)`` | ``stream.wait_event(event)`` | Identical |
| ``event.synchronize()`` | ``event.synchronize()`` | Identical |
| ``enable_timing=False`` | ``enable_timing=False`` | Disables profiling, reduces overhead |

**Verdict**: No code changes needed.

### 2.3 GPU Graphs

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaGraph_t`` | ``hipGraph_t`` | Wrapped by ``torch.cuda.CUDAGraph`` |
| ``torch.cuda.CUDAGraph()`` | ``torch.cuda.CUDAGraph()`` | Identical API |
| ``graph.capture()`` | ``graph.capture()`` | Same context manager |
| ``graph.replay()`` | ``graph.replay()`` | Same API |

**Stricter limitations on AMD**:
1. ❌ Dynamic shapes inside a captured graph → unsupported.
   **Mitigation**: Capture separate graphs per batch size (already done
   by Photon).
2. ❌ Host callbacks inside a captured graph → unsupported.
   **Mitigation**: All callbacks run outside the graph (before/after replay).
3. ✅ Fixed-shape pre-allocated buffers → supported.
   **Mitigation**: Photon's ping-pong slots pre-allocate all buffers.

**Verdict**: Code is compatible but requires stricter adherence to
fixed-shape, no-callback patterns.  Photon already satisfies these.

### 2.4 Pinned Memory

| CUDA | HIP/ROCm (PyTorch) | Notes |
|------|---------------------|-------|
| ``cudaHostAlloc`` | ``hipHostMalloc`` | ``torch.Tensor.pin_memory()`` |
| ``cudaMemcpyAsync`` | ``hipMemcpyAsync`` | ``tensor.copy_(..., non_blocking=True)`` |

**Verdict**: No code changes needed.

### 2.5 Memory Allocation

| CUDA | HIP/ROCm | Notes |
|------|-----------|-------|
| ``cudaMalloc`` | ``hipMalloc`` | ``torch.empty(device="cuda")`` |
| Allocation during hot path | **May trigger device-wide sync** | **Critical difference** |

**Key ROCm caveat**: ``hipMalloc`` on AMD GPUs can trigger a
device-wide synchronisation, unlike on NVIDIA where it is generally
non-blocking.  Photon's design of pre-allocating all buffers at init
time is therefore *even more important* for the AMD port.

**Verdict**: No code changes needed because Photon already avoids
runtime allocations.  But this is a correctness-critical invariant.

---

## 3. Performance Characteristics: NVIDIA vs AMD

### 3.1 Memory Bandwidth

| GPU | Peak BW | Architecture | Year |
|-----|---------|-------------|------|
| H100 | 3.35 TB/s | Hopper | 2023 |
| H200 | 4.8 TB/s | Hopper | 2024 |
| B200 | 8.0 TB/s | Blackwell | 2025 |
| MI300X | 5.3 TB/s | CDNA3 (gfx942) | 2024 |

The MI300X sits between an H100 and B200 in memory bandwidth, so the
GPU bubble is a meaningful problem for it.  Pipelining should yield
speedups comparable to the H100 row in the blog post's table (~15–25%
at batch).

### 3.2 Compute Characteristics

| Feature | NVIDIA | AMD |
|---------|--------|-----|
| Tensor Cores | Yes (FP16/BF16/FP8) | Matrix FP16 (equivalent) |
| FP16 throughput | 989 TFLOPS (H100) | 1307 TFLOPS (MI300X) |
| BF16 throughput | 989 TFLOPS (H100) | 1307 TFLOPS (MI300X) |
| Wavefront size | 32 (warp) | 64 (wavefront) |
| Max waves/CU | Varies | 32 per CU |

At decode time, the workload is memory-bandwidth-bound, so the FP16/BF16
throughput difference is less relevant.  The wavefront size difference
(32 vs 64) may affect occupancy in custom kernels.

### 3.3 Kernel Launch Overhead

AMD HIP has slightly higher kernel-launch overhead than CUDA (~5–10 μs
per launch vs 3–5 μs).  This makes HIP graph capture *even more
valuable* on AMD, as it collapses many launches into one.

---

## 4. Photon-AMD Implementation Notes

### 4.1 What works unchanged

The following Photon components map 1:1 to ROCm:

- **StreamManager** (``stream_manager.py``): Stream/Evet API is identical.
- **DecodeSlot** (``decode_slot.py``): Pinned memory and GPU buffer
  allocation via ``torch.empty`` is identical.
- **ZombieTracker** (``zombie.py``): Pure Python state machine, no GPU API.
- **Scheduler** (``scheduler.py``): Pure Python logic, no GPU API.
- **Pipeline** (``pipeline.py``): All GPU API calls map transparently.

### 4.2 What requires conditional handling

- **Graph capture** (``graph.py``): While the API is identical, AMD
  requires stricter adherence to the pre-allocation pattern.  We already
  satisfy this by using pre-allocated slot buffers.

### 4.3 What is AMD-custom

- **dtype preference**: ``float16`` is recommended over ``bfloat16`` on
  gfx942 for optimal memory-bandwidth utilisation, though both are supported.
- **Event overhead**: ``enable_timing=False`` is critical on AMD to avoid
  profiling overhead in the hot path.  Photon already does this.

---

## 5. Testing Strategy

### 5.1 Unit tests (CPU-able)

- ``test_zombie.py``: Zombie state machine transitions.
- ``test_scheduler.py``: Batch assembly and tick logic.

These run without a GPU and validate the pure-Python logic.

### 5.2 GPU tests (require AMD GPU + ROCm)

- ``test_slots.py``: Buffer allocation, pinned memory, H2D/D2H copies.
- ``test_benchmark.py``: Simulated pipeline throughput comparison.

Run with: ``pytest tests/ -v -k gpu --tb=short``

### 5.3 Integration benchmark

```bash
python -m photon_amd.benchmark
```

Compares blocking (1 slot) vs pipelined (2 slots + HIP graphs) modes
and reports the speedup.

---

## 6. Known Limitations

1. **HIP graph capture**: Only supported on gfx942 (MI300) and newer.
   Older CDNA2 (MI250) and RDNA3 (7900 XTX) GPUs do not support HIP
   graph capture.
2. **Multi-GPU**: This reference implementation supports single-GPU only.
   Multi-GPU inference (tensor parallelism, pipeline parallelism) would
   require additional HIP-aware inter-GPU communication primitives.
3. **Custom CUDA kernels**: Photon's production kernels (Moondream-specific
   attention, spatial decoder) are not included — this is a
   pipeline-level reference, not a full model runtime.

---

## 7. References

1. [ROCm Documentation](https://rocm.docs.amd.com/)
2. [HIP Porting Guide](https://rocm.docs.amd.com/en/latest/how-to/hipify/hip_porting_guide.html)
3. [PyTorch ROCm](https://pytorch.org/get-started/locally/)
4. Moondream Blog: [Photon Page](https://moondream.ai/p/photon)
5. Moondream Docs: [Supported Hardware](https://docs.moondream.ai/running-locally#supported-hardware)
# Analysis: "Popping the GPU Bubble" Blog Post

> **Source**: Moondream Blog, "Popping the GPU Bubble" (2026-06-04)
> **Author**: Vikhyat Korrapati, Moondream (M87 Labs)

---

## 1. Summary

This blog post presents Photon's core innovation — **pipelined decoding**
— and explains it with three independent but complementary mechanisms.
It also provides a cost model, empirical validation, and hardware-aware
analysis.

### 1.1 Key Claims

1. Photon achieves **up to 35% higher decode throughput** on a B200
   compared to a naive blocking decode loop.
2. The speedup comes from hiding CPU bookkeeping underneath GPU
   computation, not from making the GPU itself faster.
3. The pipelining is not a single monolithic change but three independent
   mechanisms that can be studied separately.
4. The win grows with GPU speed — pipelining is "insurance against the
   GPU getting faster" (or the model getting smaller).

### 1.2 Technical contributions

| Contribution | Mechanism | Impact |
|-------------|-----------|--------|
| Ping-pong slots | Alternating buffer sets prevent data races | Enables overlap |
| Forward now, sample later | Decouple forward from structured-output mask | Works for both plain and constrained text |
| Zombie lifecycle | Refcount finished requests instead of cancelling | Clean destruction without special cases |

---

## 2. Strengths of the Blog Post

### 2.1 Clarity of explanation

Each mechanism is introduced with a **problem statement → solution →
diagram → code reference** arc.  The diagrams are clean SVG renderings
that make the timing relationships immediately comprehensible.

### 2.2 Cost model with validation

The post goes beyond "we made it faster" by providing:

1. A **mathematical model** for the speedup.
2. **Predicted vs observed** numbers for 6 hardware configurations.
3. A discussion of where the prediction deviates (B200 32-stream:
   predicted +39.1%, observed +35.4%) and why (short run duration
   inflates prefill/ramp-up fraction).

This is the level of rigour expected in a systems paper, not just a
company blog post.

### 2.3 Hardware-aware analysis

The analysis spans 3090 → B200, showing how the benefit scales with
GPU capability.  The "growing bubble" insight is well-explained and
validated:

> "Pipelining is insurance against the GPU getting faster, which for
> us is the same thing as the model getting smaller."

### 2.4 Bug-catching anecdote

The post shares a debugging story:

> "We caught a bug where the pipelined numbers came out at *blocking*
> speed, traced to an accidental synchronous copy while building the
> constrained-decode mask. Moving it to the copy stream was worth +11%
> on the 3090 and +34% on the B200."

This humanises the engineering process and reinforces the lesson that
*any synchronisation on the hot path kills the gain*.

---

## 3. Areas Not Covered (Open Questions)

### 3.1 How does pipelining interact with batching at scale?

The blog post measures 1, 8, 32 streams, but Photon's production
serving likely handles hundreds of concurrent requests.  How does the
pipeline scale with very large KV caches and many active sequences?

### 3.2 Prefill-decode interleaving details

The post mentions that "a prefill is just another `kind="prefill"`
launch in the same two-slot pipeline" but doesn't show the scheduler
logic for deciding between prefill and decode in each tick — a
potentially complex scheduling decision.

### 3.3 Image preprocessing integration

Moondream is a vision-language model.  The blog post focuses on the
text decode loop but doesn't discuss how image preprocessing (resize,
tile, encode) fits into the pipeline.  This is a significant source of
prefill latency that could presumably also benefit from pipelining.

### 3.4 Multi-GPU implications

The post doesn't discuss tensor parallelism or pipeline parallelism
across multiple GPUs.  Would the pipelined decode loop need
modification for multi-GPU setups?

### 3.5 Comparison to other serving engines

The post mentions that Photon is ~2× faster than vLLM but doesn't
provide a point-by-point comparison of vLLM's and Photon's approaches
to batching, scheduling, and graph capture.  (This is defensible for a
blog post but would be expected in a paper.)

---

## 4. Key Insights for AMD Porting

### 4.1 The fundamental insight is GPU-architecture-independent

The bubble exists because CPU bookkeeping time is unrelated to GPU
speed.  This is true on AMD GPUs as much as on NVIDIA GPUs.  The
pipelining technique is directly portable.

### 4.2 AMD's higher kernel launch overhead makes graphs more valuable

HIP graph capture is *more impactful* on AMD than CUDA graphs on
NVIDIA because AMD's per-kernel launch overhead is ~2× higher.  The
graph collapse is the dominant source of latency savings on AMD.

### 4.3 The zombie tax model is universal

The $z \approx 1/(L \cdot B)$ formula depends only on request-level
statistics (average length, batch size), not on GPU architecture.
The same cost model applies directly.

### 4.4 Memory-bandwidth dominance at decode

Decode is memory-bandwidth-bound on both NVIDIA and AMD GPUs.  The
MI300X's 5.3 TB/s memory bandwidth is between an H100 (3.35 TB/s)
and a B200 (8.0 TB/s), so the Photon speedup should be in the ~15–25%
range at batch, consistent with the H100 row in the benchmark table.

---

## 5. Practical Takeaways for Photon-AMD

1. **Pre-allocate everything.** AMD's ``hipMalloc`` can trigger
   device-wide sync — every allocation in the hot path is a bubble
   source.

2. **Use async copies aggressively.** Move every D2H transfer to the
   copy stream, including the disallow mask upload.  Verify there are
   no synchronous copies left.

3. **Profile the bubble.** Before optimising, measure the actual bubble
   fraction with blocking-mode runs.  The blog post's method of
   directly measuring ``T_f, T_s, T_b`` separately provides a baseline.

4. **Capture graphs per batch size.** AMD HIP graphs require fixed
   shapes, but Photon's approach of capturing one graph per batch size
   already handles this.

5. **Don't forget the zombie tax.** For single-stream measurements,
   the ~1% overhead is visible.  For batching, it's negligible — don't
   over-engineer it.

---

## 6. References

1. Blog post: [Popping the GPU Bubble](https://moondream.ai/blog/popping-the-gpu-bubble)
2. Photon product page: [moondream.ai/p/photon](https://moondream.ai/p/photon)
3. Kestrel (Photon reference): [github.com/m87-labs/kestrel](https://github.com/m87-labs/kestrel)
4. Photon performance: [PERFORMANCE.md](https://github.com/m87-labs/kestrel/blob/main/PERFORMANCE.md)

---

## Appendix B: English Deep-Dive Documents · 英文深度解析文档

# Photon 架构：学术分析

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04)
> **代码仓库**: [m87-labs/kestrel](https://github.com/m87-labs/kestrel)
> **作者说明**: 本文是对 Photon 的深度研究，旨在创建 AMD ROCm/HIP 移植版本。
> 原始设计的全部功劳归 Moondream (M87 Labs) 所有。

---

## 1. 什么是 Photon？

Photon 是 Moondream 为视觉语言模型打造的专用推理引擎。
与通用引擎（vLLM、TensorRT-LLM）不同，Photon 利用 Moondream
模型特有的属性，实现更高的吞吐量和更低的延迟。
其核心指标是：在可比 Moondream 工作负载上比 vLLM **快约 2×**，
以及在 H100 上 **34 毫秒**的端到端推理延迟。

该引擎涵盖完整的推理服务生命周期：图像预处理、预填充（prompt + 图像编码）、
自回归解码、结构化输出生成（空间定位）以及 token 流式输出。

### 1.1 核心设计原则

1. **针对特定模型，而非通用。** Photon 仅服务于 Moondream，
   因此调度器、内存管理器、图像预处理器和流式行为均可
   针对模型的具体特性进行调优。

2. **热路径上零内存分配。** 所有 GPU 缓冲区、流和事件
   均在引擎初始化时预分配，避免了运行时 `hipMalloc`/`cudaMalloc`
   调用，后者可能触发设备级同步并引入气泡。

3. **流水化解码循环。** CPU 簿记工作（计划、发射、提交）与 GPU 计算
   重叠执行，消除了"GPU 气泡"——即 GPU 等待 CPU 完成簿记工作的空闲时间。

4. **HIP/CUDA 图捕获。** 整个解码前向过程按每种批量大小捕获为
   单个命令缓冲区（图），每次重放即可，将每次步骤的数百次内核发射
   减少为一次图重放。

---

## 2. GPU 气泡问题

### 2.1 GPU 为何在解码时空闲

在自回归文本生成中，每个 token 依赖于所有先前生成的 token。
解码循环是一个紧密的往返过程：

```
CPU: 计划 → 发射 →                → 等待 → 提交 → ...
GPU:          [前向] → 空闲 → 空闲 →       [前向] →
```

一个 token 的 GPU 前向计算是**少量计算**：几百 MB 的权重流过计算单元，
根据模型大小和 GPU 内存带宽，耗时约 2–10 毫秒。CPU 簿记工作
（调度、元数据上传、token 去标记化、约束解码掩码构建）
是每个步骤的固定开销，通常为 0.5–2 毫秒。

当 `T_forward < T_bookkeeping` 时，GPU 在每个步骤中处于空闲状态
的时间占比显著 —— 这就是 **GPU 气泡**。

### 2.2 为什么这个问题很重要

气泡随着 GPU 速度的提升而增长。随着内存带宽增加（例如 H100 → B200）
或者模型变得更小（Moondream 2B → 0.5B），气泡在步时的占比会越来越大。
以下是来自博客文章的测量数据：

| 硬件 | 并发数 | T_block (ms) | T_pipe (ms) | 加速比 |
|------|--------|--------------|-------------|--------|
| RTX 3090 | 1  | 5.44  | 5.10  | +6.5%  |
| RTX 3090 | 32 | 11.74 | 10.52 | +11.6% |
| B200     | 1  | 3.11  | 2.63  | +17.6% |
| B200     | 32 | 5.55  | 3.98  | +35.4% |

加速比随 GPU 能力的提升而增长：相同批量大小下，3090 上 +12%，
B200 上 +35%。流水化是**对 GPU 变快的保险**。

### 2.3 数学模型

阻塞模式下：

$$T_{\text{step}} = T_{\text{forward}} + T_{\text{bookkeeping}}$$

流水线模式下：

$$T_{\text{step}} \approx \max(T_{\text{forward}}, T_{\text{bookkeeping}})$$

加速比因子：

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

其中 $z \approx 1/L$ 为*僵尸税*——为已完成序列多跑一次前向的成本。
当 $L \approx 110$ 时，在 batch=1 下 $z \approx 1\%$，
在 batch ≫ 1 时几乎为零。

---

## 3. 系统架构

### 3.1 组件图

```
                     ┌─────────────────────────┐
                     │     PhotonEngine          │
                     │  (流水线编排)              │
                     ├─────────────────────────┤
                     │  StreamManager            │
                     │  (计算流 + 拷贝流)          │
                     ├─────────────────────────┤
                     │  Scheduler                │
                     │  (tick: launch→commit→    │
                     │   finalize)               │
                     ├───────────┬─────────────┤
                     │  Slot 0   │  Slot 1      │
                     │  (缓冲区,  │  (缓冲区,    │
                     │   图对象,  │   图对象,    │
                     │   事件)   │   事件)       │
                     ├───────────┴─────────────┤
                     │  GraphManager (HIP)       │
                     │  ZombieTracker            │
                     └─────────────────────────┘
```

### 3.2 流模型

```
计算流 ──┬── Forward(N) ────────┬── Forward(N+1) ──┬── ...
          │                     │                  │
拷贝流 ──┴── (空闲) ────────────┴── D2H拷贝(N) ───┴── D2H拷贝(N+1) ...
            等待                    等待
            step_done_event         step_done_event
```

- **compute_stream** (priority=0)：所有 GPU 前向在此串行化，
  保证 token 的序列依赖关系。
- **copy_stream** (priority=-1)：采样输出的设备→主机拷贝。
  等待在计算流上记录的 `step_done_event`（当某步的输出就绪时）。

两个流相互独立，因此当前一步的拷贝仍在进行时，
下一个前向可以立即开始 —— 气泡被消除了。

### 3.3 单 Tick 生命周期

```
Tick N:
  LAUNCH    步骤 N 的批量    (slot = N % 2)
  COMMIT    步骤 N-1 的批量  (slot = (N-1) % 2, 通过 D2H 拷贝)
  FINALIZE  步骤 N 的批量    (约束解码掩码上传)
```

三个阶段的详细说明见各机制文档。

---

## 4. 三大机制

详见各专门文档：

| 机制 | 文档 | 概要 |
|------|------|------|
| 1. 乒乓槽位 | [02_mechanism_pingpong.md](./02_mechanism_pingpong.md) | 两套交替缓冲区防止流水线步骤之间的数据竞争。 |
| 2. 先行推理，后采样 | [03_mechanism_fsl.md](./03_mechanism_fsl.md) | Forward 在提交之前发射；采样等待提交阶段构建的掩码。 |
| 3. 僵尸序列 | [04_mechanism_zombies.md](./04_mechanism_zombies.md) | 已完成请求多跑一次前向作为"僵尸"，避免中途取消。 |

---

## 5. AMD ROCm 移植注意事项

详见 [06_amd_rocm_porting.md](./06_amd_rocm_porting.md) 了解将 NVIDIA (CUDA)
Photon 实现移植到 AMD ROCm/HIP 的详细说明。

关键点：
- PyTorch 的 ROCm 后端为流、事件、图和钉住内存提供透明的 CUDA→HIP 映射。
- HIP 图捕获在 gfx942 (MI300) 上受支持且稳定，但对固定形状、无回调的要求
  比 CUDA 图更严格。
- 在 MI300 上推荐使用 `float16` 以获得内存带宽密集型工作负载的最佳性能。
- 运行时内存分配可能触发 HIP 上的设备级同步；所有缓冲区**必须**预分配。

---

## 参考文献

1. Moondream 博客: ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) (2026-06-04)
2. Moondream 文档: [本地运行 Moondream](https://docs.moondream.ai/running-locally)
3. Kestrel 代码仓库: [m87-labs/kestrel](https://github.com/m87-labs/kestrel)
4. NVIDIA: [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
5. AMD: [ROCm 文档](https://rocm.docs.amd.com/)
# 机制一：乒乓槽位

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04),
> "Mechanism 1: ping-pong slots" 章节

---

## 1. 问题陈述

一个解码步骤需要一组 GPU 缓冲区：输入暂存区、注意力 KV 缓存引用、
前向输出（logits）以及采样 token 的存放位置。如果只有一套缓冲区，
它们在该步骤完成之前始终处于"使用中"状态，我们无法在当前步骤完成之前
启动下一个步骤。

要实现两个步骤的流水化，第二步需要自己的工作集 ——
否则可能在上一步的结果尚未被 CPU 读取之前就覆盖它们。

## 2. 解决方案

分配**两套**完整的解码缓冲区（"槽位"），并以乒乓风格交替使用：

```
步骤 N     → Slot 0 (计算: 前向 + 采样)
步骤 N+1   → Slot 1 (计算: 前向 + 采样, 同时 CPU 提交 Slot 0)
步骤 N+2   → Slot 0 (计算: 前向 + 采样, 同时 CPU 提交 Slot 1)
步骤 N+3   → Slot 1 (以此类推...)
```

### 2.1 槽位的内容

每个 `DecodeSlot` 包含：

| 缓冲区 | 位置 | 形状 | 用途 |
|--------|------|------|------|
| `decode_token_ids` | GPU | `[max_batch]` | 输入前向的 token id |
| `logits` | GPU | `[max_batch, vocab]` | 前向输出分数 |
| `hidden_last` | GPU | `[max_batch, hidden_dim]` | 最后隐藏状态（用于空间解码器） |
| `sampled_ids` | GPU | `[max_batch]` | 每行采样的 token id |
| `sampled_logprobs` | GPU | `[max_batch]` | 采样 token 的对数概率 |
| `fa3_page_table` | GPU | `[max_batch, pages]` | 分页 KV 缓存页表 |
| `fa3_seqused_k` | GPU | `[max_batch]` | 每个序列的 KV 长度 |
| `batch_idx` | 钉住主机 | `[max_batch]` | 序列批量索引 |
| `input_pos` | 钉住主机 | `[max_batch]` | token 位置 |
| `disallow_mask` | 钉住主机 | `[max_batch, vocab]` | 约束解码掩码 |

### 2.2 钉住内存（Pinned Memory）

页锁定（钉住）主机内存允许 GPU 的 DMA 引擎在不阻塞 CPU 的前提下
执行数据拷贝。如果没有钉住，每次设备→主机 (D2H) 传输都需要驱动程序
先拷贝到一个钉住的暂存缓冲区，这会使拷贝和下一次内核发射串行化。

在 AMD ROCm 上，`torch.Tensor.pin_memory()` 映射到 `hipHostMalloc`，
其语义与 CUDA 的 `cudaHostAlloc` 完全相同。

### 2.3 槽位生命周期

```
┌────────┐   launch     ┌────────┐   D2H拷贝     ┌────────┐
│  空闲   │ ──────────► │ 使用中  │ ──────────► │ 提交中  │
└────────┘              └────────┘               └────────┘
     ▲                                                  │
     │                                                  │
     └──────────────────────────────────────────────────┘
                      提交完成 → 空闲
```

一个槽位从被发射 (launch) 的那一刻起算作"使用中"，直到其
`commit_done_event` 被记录（即 CPU 已从其钉住主机缓冲区读取完毕）。
只有返回"空闲"状态的槽位才能被重新发射。

### 2.4 为什么不是 GPU 并行？

两个槽位**并非**将各自的前向放在不同流上以实现并发执行。
两者都使用**同一条**计算流。槽位存在的唯一目的是
让 CPU 能够在一个槽位的 GPU 前向运行的同时处理另一个槽位的结果 ——
它们提供的是 *CPU/GPU 重叠*，而非 *GPU/GPU 并行*。

## 3. 异步 D2H 拷贝

使乒乓机制生效的关键洞见：采样输出的 D2H 拷贝在**拷贝流**上执行，
而非计算流。

```
计算流: ── Forward(Slot0) ──────────── Forward(Slot1) ── ...
                         │
                  step_done_event
                         │
拷贝流:    ──────────────┴── D2H拷贝(Slot0) ── ...
```

`step_done_event` 在某个槽位的前向输出就绪时记录在计算流上。
拷贝流在开始 D2H 拷贝前等待此事件，以确保正确性。但计算流本身
**并不等待** —— 下一个前向可以立即开始。

## 4. AMD ROCm 实现说明

`photon_amd.decode_slot` 模块以如下方式实现此模式：

- `CpuGpuBuffer`：配对的钉住主机 + 设备内存缓冲区，提供
  `copy_host_to_device` 和 `copy_device_to_host` 方法，
  接受流对象以执行异步传输。
- `DecodeMetaBuffers`：每个槽位的钉住主机元数据（batch_idx、input_pos、disallow_mask）。
- `DecodeSlot`：完整的资源包，在引擎初始化时通过 `create_decode_slot()` 一次性创建。

所有 GPU 缓冲区在初始化时以固定形状预分配，以满足 HIP 图捕获的要求。

## 5. 参考文献

1. Photon 源码: `kestrel/models/moondream/decode_slot.py`
2. 博客文章: [机制一示意图](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-1-ping-pong-slots)
3. NVIDIA: [CUDA Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
4. AMD: [HIP 流管理](https://rocm.docs.amd.com/projects/HIP/en/latest/programming_guide.html#stream-management)
# 机制二：先行推理，后采样

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04),
> "Mechanism 2: forward now, sample later" 章节

---

## 1. 依赖关系难题

下一个前向并不依赖于 CPU 对上一步 token 所做的任何处理 ——
前向只需要 token ID，而它已在 GPU 上了。因此原则上，前向可以提前执行。

但是，下一个步骤的**某些**方面确实依赖于上一步已提交的结果：

1. **批量成员**：如果某个请求在步骤 t 完成，它不应出现在步骤 t+1 中。
   这由机制三（僵尸序列）处理。

2. **约束解码掩码**：Moondream 的空间技能（`point`、`detect`、`segment`）
   限制模型在每个步骤可以产生哪些 token。步骤 t+1 的掩码取决于步骤 t
   已采样的 token —— 而当我们想发射步骤 t+1 时，该 token 尚未被 CPU 提交。

## 2. 解决方案：将前向与采样解耦

关键洞见：依赖关系在**采样**中，而非在前向中。
前向本身不需要掩码 —— 它只是计算整个词汇表的 logits。

解决方案是将每个步骤拆分为两个阶段，它们在不同的流水线阶段执行：

```
┌──────────────────────────────────────────────────────────┐
│  Tick N                                                   │
│                                                           │
│  LAUNCH:   Forward(第N步) → [分数]                      │
│            （不需要掩码 — 分数是无偏的）                   │
│                                                           │
│  COMMIT:   等待 token(第N-1步) 到达 CPU。               │
│            推进状态 → 构建 mask(第N步)。                │
│                                                           │
│  FINALIZE: 应用 mask(第N步) → Sample(第N步)。           │
│            （此时掩码已是最新的，因为 COMMIT 刚刚完成）     │
└──────────────────────────────────────────────────────────┘
```

这建立了**"先提交后定稿"的排序**：前向立即执行，但采样必须等到
上一步的提交完成，使每个序列的状态变为最新之后才进行。

## 3. 统一处理

博客文章描述了如何用一个循环优雅地同时处理纯文本和约束解码：

### 3.1 纯文本（无约束解码）

- 无需掩码。
- 前向**和**采样都可以提前一步执行。
- 两者都在 LAUNCH 阶段发生。

### 3.2 约束解码（结构化输出）

- 前向提前一步执行（在 LAUNCH 中）。
- 采样等待上一次提交（在 FINALIZE 中）。
- COMMIT 阶段根据更新后的状态构建掩码，
  然后 FINALIZE 应用掩码并进行采样。

### 3.3 优雅之处

无需任何特例判断。同一套调度器代码同时处理两者：
- LAUNCH 始终执行前向。
- COMMIT 始终处理上一步的结果。
- FINALIZE 始终执行掩码相关的工作。

对于纯文本，FINALIZE 是空操作（掩码为空）。对于约束序列，
FINALIZE 等待刚刚发生的提交。

## 4. 禁止掩码（Disallow Mask）

Moondream 的约束解码使用*禁止掩码*：一个布尔型张量 `[batch, vocab]`，
其中 `True` 表示"在采样前将该 token 的 logit 强制设为负无穷"。

每个序列的掩码由该序列当前的解码状态构建 ——
即它在结构化输出中已经输出了哪些 token。例如：
- `point` 输出总是输出一个坐标 `(x, y)`。
- `detect` 输出遍历一个 `x, y, width, height` 循环。

步骤 t+1 的掩码在步骤 t 的 COMMIT 结束时计算，此时序列状态是最新的。
然后通过异步 H2D 拷贝（在拷贝流上）上传到 GPU 槽位，
使其在 FINALIZE 执行时立即可用。

## 5. 性能影响

引自博客文章"It only pays once the bubble is actually hideable"一节：

> 我们发现了一个 bug，流水线数据却表现为*阻塞*速度，
> 追踪到在构建约束解码掩码时发生了一次意外的同步拷贝。
> 将其移至拷贝流后，在 3090 上获得了 +11% 的提升，
> 在 B200 上获得了 +34% 的提升。

教训：**热路径上的任何同步都会毁掉流水化的收益**。
即使是在关键路径内的单次同步拷贝，也足以破坏重叠效果。

## 6. AMD ROCm 实现说明

在我们的 `photon_amd` 实现中：

- `PipelineCallbacks.do_sample` 在 `_launch()` 期间为纯文本
  （掩码无关的采样）调用，或在 `_finalize()` 期间为约束解码调用。
- 禁止掩码存储在 `DecodeMetaBuffers.disallow_mask` 中，
  并通过拷贝流异步上传到 GPU。
- `slot.mask_ready_event` 标记掩码上传完成，采样内核可等待该事件。

## 7. 参考文献

1. Photon 源码: `kestrel/models/moondream/tokenizer.py`（结构化解码逻辑）
2. 博客文章: [机制二示意图](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-2-forward-now-sample-later)
3. NVIDIA: [Guided Decoding with Logits Processor](https://huggingface.co/docs/transformers/main_classes/logits)
# 机制三：僵尸序列 —— 提前定稿，延迟释放

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04),
> "Mechanism 3: zombies: finalize early, release late" 章节

---

## 1. 问题

在机制二中，我们指出步骤 t+1 的**批量成员**依赖于步骤 t 的提交结果。
如果某个序列在步骤 t 完成（EOS 或长度封顶），它不应出现在步骤 t+1 的批次中。

但步骤 t+1 是在步骤 t **提交之前**被发射的 —— 这正是流水化的全部意义。
所以已完成的序列**已被写入步骤 t+1 的前向中**。你无法取消已发射的 GPU 工作。

## 2. 朴素方案：飞行中取消

一个可以设想的方案是在飞行中取消该行：
- 将该序列的 token 设为"无关"值。
- 或在注意力内核中跳过它。

但这需要每个内核（注意力、MLP、采样、提交）都有特例逻辑，
形成"一团取消逻辑的乱麻"。

## 3. 优雅方案：让它多跑一次

Photon 的方法是更简单的：**让已完成的序列作为"僵尸"再多跑一次前向**。

```
步骤 t:     序列 X 存活 → 前向计算真实结果
            提交检测到 EOS → X 被标记为 "finalized"
            结果已输出给调用方
            但是：X 的 KV 页**并未释放**

步骤 t+1:   X 已在批次中 → 前向将 X 作为 "僵尸" 一并计算
            提交发现 X 已 finalized → 跳过，不追加 token
            释放 X 的飞行引用计数

Refcount=0: 回收 X 的 KV 页和 LoRA 槽位
```

僵尸占着一个批量槽位，写入一些永远不会被读取的 KV 缓存条目 ——
但这只是一个很小的代价。

## 4. 每序列状态机

每个序列额外携带两个字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `finalized` | `bool` | 当检测到 EOS 或达到长度上限后设置为 `True`。 |
| `inflight_refs` | `int` | 0、1 或 2 —— 有多少个飞行中的前向引用了该序列。 |

状态转移：

```
IDLE (0 refs, not finalized)
  │
  ├── launch ──► ACTIVE (1 ref)
  │                │
  │                ├── launch ──► ACTIVE (2 refs, 已流水化)
  │                │                │
  │                │                ├── commit (EOS) ──► ZOMBIE (2 refs, finalized)
  │                │                │                      │
  │                │                │                      ├── commit (skip)
  │                │                │                      └── release ──► ZOMBIE (1 ref)
  │                │                │
  │                │                ├── commit (normal) ──► ACTIVE (1 ref)
  │                │                └── release ──► ACTIVE (1 ref)
  │                │
  │                ├── commit (EOS) ──► ZOMBIE (1 ref, finalized)
  │                │                      │
  │                │                      └── release ──► IDLE (已回收)
  │                │
  │                └── commit (normal) ──► ACTIVE (1 ref)
  │
  └── release ──► IDLE (如果 finalized 则回收，否则无操作)
```

## 5. 僵尸税分析

来自博客文章的公式，*僵尸税* $z$ 是浪费前向的比例：

$$z \approx \frac{1}{L \cdot B}$$

其中：
- $L$ = 每个请求平均生成的 token 数
- $B$ = 批量大小

### 5.1 在 batch = 1 时

每 $L$ 个 token 浪费一次前向：$z \approx 1/L$。
在 $L \approx 110$ 时，约为 1% 的开销 —— 可忽略不计。

### 5.2 在 batch > 1 时

僵尸只是**多一行**数据，而该步骤已经在将全部模型权重
流过计算单元。在 decode 的内存带宽受限场景下，
权重传输才是主导因素，多一行几乎不增加任何成本。
当批量大小 ≫ 1 时，僵尸税实际上趋近于零。

### 5.3 实验验证

来自博客文章：

| 硬件 | 并发数 | L | 预测值 | 实测值 |
|------|--------|---|--------|--------|
| 3090 | 1  | 104 | +5.7%  | +6.5%  |
| 3090 | 32 | 113 | +11.1% | +11.6% |
| B200 | 1  | 115 | +17.2% | +17.6% |
| B200 | 32 | 104 | +39.1% | +35.4% |

预测加速比（已考虑僵尸税）与实测值高度吻合，证实了该模型的准确性。

## 6. 资源回收

资源**不会**在序列首次完成时释放。只有当 `inflight_refs` 降到 0 时才释放：

- **KV 缓存页**：归还到页面分配器的空闲列表。
- **LoRA 槽位**：归还到 LoRA 槽位池。
- **批量槽位**：在下一次调度周期中释放给新序列。

这种延迟释放策略是不需要飞行中取消逻辑的代价。

## 7. AMD ROCm 实现说明

在 `photon_amd/zombie.py` 中：

- `ZombieState`：每序列字段（`finalized`、`inflight_refs`、`kv_page_start`、`kv_page_count`、`lora_slot`）。
- `ZombieTracker`：中央管理器，在三个钩子点被调用：
  - `launch_ref()` —— 当序列进入批次时增加引用计数。
  - `commit()` —— 检测 EOS，返回 "finalized" 或 "skip"。
  - `release()` —— 减少引用计数，完成后回收资源。
- `should_skip()` —— 查询某行是否是僵尸。

调度器分别在 `_assemble_batch()`（发射引用）、`_commit()`（提交）
和 `_recycle_done()`（释放）中调用这些钩子。

## 8. 参考文献

1. 博客文章: [机制三示意图](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-3-zombies-finalize-early-release-late)
2. Photon 源码: `kestrel/models/moondream/zombie.py`（近似路径）
3. 相关: [引用计数在垃圾回收中的应用](https://en.wikipedia.org/wiki/Reference_counting)
# GPU 气泡：数学分析

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04),
> "A cost model for the bubble" 章节

---

## 1. 定义气泡

一个解码步骤包含三个不同的工作片段：

| 片段 | 位置 | 描述 | 时间 |
|------|------|------|------|
| **前向 (Forward)** | GPU | 权重传输矩阵乘法（注意力投影、MLP）。在 decode 时受内存带宽约束。 | $T_f$ |
| **采样 (Sampling)** | GPU → CPU | 约束解码掩码应用、argmax/sample、空间解码、D2H 拷贝。 | $T_s$ |
| **簿记 (Bookkeeping)** | CPU | 计划（批量组装）、发射（图重放触发）、提交（token 追加、状态推进、掩码构建）。 | $T_b$ |

### 1.1 阻塞模式

```
时间 →
CPU: [计划][发射] ────────────── [同步][提交][计划][发射] ──────────────
GPU:                [前向][采样] =空闲=                  [前向][采样]
                    ◄─── T_f + T_s ───►
                                       ◄── T_b ──►  ← GPU 气泡
```

总步时：$T_{\text{block}} = T_f + T_s + T_b$

GPU 空闲时间为 $T_b$ —— 即气泡。

### 1.2 流水线模式

```
时间 →
CPU: [计划][发射] ───────────────── [提交][计划][发射] ───────────────── [提交]
GPU:                [前向][采样]                       [前向][采样]
                    ◄── T_f + T_s ──►                 ◄── T_f + T_s ──►
```

总步时：$T_{\text{pipe}} = \max(T_f + T_s, T_b)$

GPU 空闲为 $0$（假设 $T_f + T_s \ge T_b$）。

## 2. 加速比公式

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

其中 **$z$** 是来自机制三的僵尸税：

$$z \approx \frac{1}{L} \quad \text{（batch = 1 时）}$$

$$z \approx 0 \quad \text{（batch ≫ 1 时）}$$

### 2.1 两种竞争力量

1. **气泡隐藏** ($T_{\text{block}} / T_{\text{pipe}}$)：将 CPU 簿记与 GPU 计算重叠的直接收益。

2. **僵尸税** ($z$)：允许已完成请求多跑一次前向的间接成本。
   在 batch=1 时为 $1/L$（约 1%）。在大批量下，
   成本平摊到接近为零，因为多出的一行处于已传输全部权重的
   内存带宽约束步骤中。

## 3. 为什么气泡会增长

前向时间 $T_f$ 与模型权重大小和 GPU 内存带宽相关：

$$T_f \approx \frac{W}{B} \cdot \frac{1}{N_{\text{CU}}}$$

其中：
- $W$ = 模型总权重（字节）
- $B$ = GPU 内存带宽（字节/秒）
- $N_{\text{CU}}$ = 活跃计算单元数

随着 GPU 内存带宽增加（H100: 3.35 TB/s → B200: 8 TB/s）或
模型变小（Moondream 2B → 0.5B），$T_f$ 缩小但 $T_b$ 保持不变。
气泡在步时中的**占比越来越大**。

### 3.1 具体示例

| 模型 | GPU | W (GB) | B (TB/s) | $T_f$ (ms) | $T_b$ (ms) | 气泡占比 |
|------|-----|--------|----------|------------|------------|----------|
| Moondream 2B | H100 | ~4 | 3.35 | ~1.2 | 1.5 | 55.6% |
| Moondream 2B | B200 | ~4 | 8.0 | ~0.5 | 1.5 | 75.0% |
| Moondream 0.5B | H100 | ~1 | 3.35 | ~0.3 | 1.5 | 83.3% |

气泡在**小模型 + 快速 GPU** 场景下是**主导因素** ——
这正是 Moondream 所运行的使用场景。

## 4. 预填充在同一管道中

预填充（处理新请求的 prompt + 图像）也是一种前向传递，
但一次处理多个 token —— 它是计算绑定而非内存带宽绑定的，
因此 $T_f$ 更大且 $T_s$ 更昂贵。

Photon 对预填充和解码使用**相同的双槽位管道**。
预填充只是一个 `kind="prefill"` 步骤，使用相同的槽位和计算流。
因为管道只关心槽位是否空闲，而不关心上次是什么类型的工作使用了它，
所以预填充和解码可以透明地交织。

这在输出较短时尤为重要 —— 一个仅输出三个 token 的请求，
几乎全部时间都消耗在预填充上，而大量短请求的工作负载本质上
就是一连串预填充中穿插少量解码步骤。
共享一条管道使得预填充的 CPU 簿记（接纳、分词）能够与解码前向重叠，
反之亦然。

## 5. 验证

来自博客文章在 B200 上的实验测量：

| 模式 | $T_f$ (ms) | $T_s$ (ms) | 周期 (ms) |
|------|-----------|-------------|-----------|
| 1 路 | 2.45 | 0.14 | 2.63 |
| 8 路 | 3.12 | 0.14 | 3.30 |
| 32 路 | 3.80 | 0.14 | 3.98 |

$T_f + T_s \approx$ 周期，意味着 GPU 在整个周期内几乎持续繁忙 ——
气泡已被消除。剩余的 GPU 空闲时间不到 0.05 ms。

### 5.1 阻塞 vs 流水线对比

| 硬件 | 并发数 | 阻塞 (ms) | 流水线 (ms) | 加速比 |
|------|--------|-----------|-------------|--------|
| RTX 3090 | 1  | 5.44  | 5.10  | +6.5%  |
| RTX 3090 | 8  | 7.52  | 6.97  | +7.8%  |
| RTX 3090 | 32 | 11.74 | 10.52 | +11.6% |
| B200 | 1  | 3.11  | 2.63  | +17.6% |
| B200 | 8  | 4.04  | 3.30  | +21.9% |
| B200 | 32 | 5.55  | 3.98  | +35.4% |

三点观察：
1. **收益随 GPU 速度增长** —— 3090 上 +12%，但 B200 上 +35%。
2. **僵尸税真实存在但很小** —— batch=1 时约 1%，batch > 1 时基本为零。
3. **只有当气泡可被隐藏时才有效** —— 热路径上的任何同步操作都会
   扼杀收益。

## 6. 参考文献

1. 博客文章: [A cost model for the bubble](https://moondream.ai/blog/popping-the-gpu-bubble#a-cost-model-for-the-bubble)
2. NVIDIA: [GPU 内存带宽](https://www.nvidia.com/en-us/data-center/a100/)
3. AMD: [MI300 内存带宽](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
# 将 Photon 从 NVIDIA CUDA 移植到 AMD ROCm

> 一份将 Photon 推理引擎从 NVIDIA GPU 移植到 AMD Instinct 加速器
> (gfx942 / MI300) 的实用指南。

---

## 1. 概述

Photon（及其在 `kestrel` 代码库中的参考实现）是基于 NVIDIA CUDA API
编写的。本文档记录了每一个 CUDA 特定的操作点，并将其映射到等价的
AMD ROCm/HIP API，标注出了两者的不同之处。

好消息：**PyTorch 提供了一层透明的兼容层。** 在 CUDA 上运行的大多数
PyTorch 代码无需修改即可在 ROCm 上运行，因为 PyTorch 的 CUDA API
函数在底层被转换为 HIP 调用。需要关注的领域有：

1. 流与事件管理
2. GPU 图捕获
3. 钉住内存
4. 内核发射开销
5. 内存分配模式
6. 硬件特定的性能特征

---

## 2. CUDA → HIP/ROCm 映射

### 2.1 流

| CUDA | HIP/ROCm (PyTorch) | 说明 |
|------|---------------------|------|
| `cudaStream_t` | `hipStream_t` | 由 `torch.cuda.Stream` 封装 |
| `torch.cuda.Stream()` | `torch.cuda.Stream()` | 完全相同的 API |
| `stream.synchronize()` | `stream.synchronize()` | 完全相同 |
| 优先级 | 优先级 | 两者都支持 0（默认）和 -1（更高） |

**结论**：无需代码修改。PyTorch 的 Stream API 完全兼容。

### 2.2 事件

| CUDA | HIP/ROCm (PyTorch) | 说明 |
|------|---------------------|------|
| `cudaEvent_t` | `hipEvent_t` | 由 `torch.cuda.Event` 封装 |
| `event.record(stream)` | `event.record(stream)` | 完全相同 |
| `stream.wait_event(event)` | `stream.wait_event(event)` | 完全相同 |
| `event.synchronize()` | `event.synchronize()` | 完全相同 |
| `enable_timing=False` | `enable_timing=False` | 禁用性能分析，减少开销 |

**结论**：无需代码修改。

### 2.3 GPU 图

| CUDA | HIP/ROCm (PyTorch) | 说明 |
|------|---------------------|------|
| `cudaGraph_t` | `hipGraph_t` | 由 `torch.cuda.CUDAGraph` 封装 |
| `torch.cuda.CUDAGraph()` | `torch.cuda.CUDAGraph()` | 完全相同的 API |
| `graph.capture()` | `graph.capture()` | 相同的上下文管理器 |
| `graph.replay()` | `graph.replay()` | 相同的 API |

**AMD 上更严格的限制**：
1. ❌ 捕获图内的动态形状 → 不支持。
   **应对方案**：为每个批量大小分别捕获图（Photon 已这样做）。
2. ❌ 捕获图内的主机回调 → 不支持。
   **应对方案**：所有回调在图外运行（在重放之前/之后）。
3. ✅ 固定形状的预分配缓冲区 → 支持。
   **应对方案**：Photon 的乒乓槽位预先分配了所有缓冲区。

**结论**：代码兼容，但需要对固定形状、无回调模式有更严格的要求。
Photon 已经满足这些要求。

### 2.4 钉住内存

| CUDA | HIP/ROCm (PyTorch) | 说明 |
|------|---------------------|------|
| `cudaHostAlloc` | `hipHostMalloc` | `torch.Tensor.pin_memory()` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | `tensor.copy_(..., non_blocking=True)` |

**结论**：无需代码修改。

### 2.5 内存分配

| CUDA | HIP/ROCm | 说明 |
|------|-----------|------|
| `cudaMalloc` | `hipMalloc` | `torch.empty(device="cuda")` |
| 在热路径中分配 | **可能触发设备级同步** | **关键差异** |

**ROCm 的关键注意事项**：与 NVIDIA 上一般非阻塞的行为不同，
AMD GPU 上的 `hipMalloc` 可能触发设备级同步。
因此，Photon 在初始化时预分配所有缓冲区的设计
在 AMD 移植中**更加重要**。

**结论**：无需代码修改，因为 Photon 已经避免了运行时分配。
但这是一个正确性关键的不变性约束。

---

## 3. 性能特征：NVIDIA vs AMD

### 3.1 内存带宽

| GPU | 峰值带宽 | 架构 | 年份 |
|-----|---------|------|------|
| H100 | 3.35 TB/s | Hopper | 2023 |
| H200 | 4.8 TB/s | Hopper | 2024 |
| B200 | 8.0 TB/s | Blackwell | 2025 |
| MI300X | 5.3 TB/s | CDNA3 (gfx942) | 2024 |

MI300X 的内存带宽介于 H100 和 B200 之间，因此 GPU 气泡对它来说
是一个有意义的问题。流水化在批量模式下应产生与博客文章表中
H100 行相当的速度提升（约 15–25%）。

### 3.2 计算特征

| 特征 | NVIDIA | AMD |
|------|--------|-----|
| Tensor Cores | 支持 (FP16/BF16/FP8) | 矩阵 FP16（等价） |
| FP16 吞吐量 | 989 TFLOPS (H100) | 1307 TFLOPS (MI300X) |
| BF16 吞吐量 | 989 TFLOPS (H100) | 1307 TFLOPS (MI300X) |
| Wavefront/warp 大小 | 32 (warp) | 64 (wavefront) |
| 每 CU 最大 waves | 不等 | 每 CU 32 |

在 decode 时，工作负载受内存带宽限制，因此 FP16/BF16 吞吐量差异
不那么重要。Wavefront 大小的差异 (32 vs 64) 可能影响自定义内核的占用率。

### 3.3 内核发射开销

AMD HIP 的内核发射开销略高于 CUDA（每次发射约 5–10 μs vs 3–5 μs）。
这使 HIP 图捕获在 AMD 上**更有价值**，因为它将多次发射合并为一次。

---

## 4. Photon-AMD 实现说明

### 4.1 无需修改即可使用的部分

以下 Photon 组件可 1:1 映射到 ROCm：

- **StreamManager** (`stream_manager.py`)：Stream/Event API 完全相同。
- **DecodeSlot** (`decode_slot.py`)：通过 `torch.empty` 分配钉住内存和
  GPU 缓冲区完全相同。
- **ZombieTracker** (`zombie.py`)：纯 Python 状态机，不涉及 GPU API。
- **Scheduler** (`scheduler.py`)：纯 Python 逻辑，不涉及 GPU API。
- **Pipeline** (`pipeline.py`)：所有 GPU API 调用透明映射。

### 4.2 需要条件处理的

- **图捕获** (`graph.py`)：虽然 API 相同，但 AMD 对预分配模式有更严格
  的要求。我们通过使用预分配的槽缓冲区已经满足了这一要求。

### 4.3 AMD 定制部分

- **dtype 偏好**：在 gfx942 上建议使用 `float16` 而非 `bfloat16`
  以获得最佳内存带宽利用率，尽管两者都受支持。
- **事件开销**：在 AMD 上 `enable_timing=False` 至关重要，以避免
  热路径中的性能分析开销。Photon 已经这样做了。

---

## 5. 测试策略

### 5.1 单元测试（可 CPU 运行）

- `test_zombie.py`：僵尸状态机转换。
- `test_scheduler.py`：批量组装与 tick 逻辑。

这些测试无需 GPU，用于验证纯 Python 逻辑。

### 5.2 GPU 测试（需要 AMD GPU + ROCm）

- `test_slots.py`：缓冲区分配、钉住内存、H2D/D2H 拷贝。
- `test_benchmark.py`：模拟的流水线吞吐量比较。

执行方式：`pytest tests/ -v -k gpu --tb=short`

### 5.3 集成基准测试

```bash
python -m photon_amd.benchmark
```

比较阻塞模式（1 槽）vs 流水线模式（2 槽 + HIP 图），
并报告加速比。

---

## 6. 已知限制

1. **HIP 图捕获**：仅支持 gfx942 (MI300) 及更新的架构。
   旧版 CDNA2 (MI250) 和 RDNA3 (7900 XTX) GPU 不支持 HIP 图捕获。
2. **多 GPU**：此参考实现仅支持单 GPU。多 GPU 推理
   （张量并行、流水线并行）需要额外的 HIP 感知的跨 GPU 通信原语。
3. **自定义 CUDA 内核**：Photon 的生产级内核（Moondream 特定的注意力、
   空间解码器）未包含在内 —— 这是一个管线级别的参考实现，
   而非完整的模型运行时。

---

## 7. 参考文献

1. [ROCm 文档](https://rocm.docs.amd.com/)
2. [HIP 移植指南](https://rocm.docs.amd.com/en/latest/how-to/hipify/hip_porting_guide.html)
3. [PyTorch ROCm](https://pytorch.org/get-started/locally/)
4. Moondream 博客: [Photon 产品页](https://moondream.ai/p/photon)
5. Moondream 文档: [支持的硬件](https://docs.moondream.ai/running-locally#supported-hardware)
# 分析："Popping the GPU Bubble" 博客文章

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04)
> **作者**: Vikhyat Korrapati, Moondream (M87 Labs)

---

## 1. 概述

这篇博客文章介绍了 Photon 的核心创新 —— **流水化解码** ——
并通过三个独立但互补的机制对其进行解释。文章还提供了成本模型、
实验验证和硬件感知分析。

### 1.1 核心观点

1. 与朴素的阻塞解码循环相比，Photon 在 B200 上实现了**高达 35% 的解码吞吐量提升**。
2. 加速来自将 CPU 簿记隐藏在 GPU 计算之下，而非让 GPU 本身更快。
3. 流水化并非单一的整体改动，而是三个可以分别研究的独立机制。
4. 收益随 GPU 速度增长 —— 流水化是"对 GPU 变快的保险"
  （或者对模型变小的保险）。

### 1.2 技术贡献

| 贡献 | 机制 | 影响 |
|------|------|------|
| 乒乓槽位 | 交替缓冲区集防止数据竞争 | 启用重叠 |
| 先行推理，后采样 | 将前向与结构化输出掩码解耦 | 同时适用于纯文本和约束文本 |
| 僵尸生命周期 | 对已完成请求进行引用计数，而非取消 | 无需特例判断即可干净销毁 |

---

## 2. 博客文章的优点

### 2.1 清晰的解释

每个机制都遵循 **问题陈述 → 解决方案 → 图示 → 代码引用** 的脉络引入。
图示是干净的 SVG 渲染，使时序关系一目了然。

### 2.2 带有验证的成本模型

文章不仅说"我们让它变快了"，还提供了：

1. 加速比的**数学模型**。
2. 6 种硬件配置的**预测值 vs 实测值**。
3. 对预测偏差的讨论（B200 32 路：预测 +39.1%，实测 +35.4%）
   及其原因（短运行时长放大了预填充/爬坡占比）。

这符合系统论文中的严谨性要求，而不仅仅是公司博客文章。

### 2.3 硬件感知分析

分析覆盖了 3090 → B200 的跨度，展示了收益如何随 GPU 能力增长。
"增长中的气泡"洞见解释清晰且经过验证：

> "流水化是对 GPU 变快的保险，而对我们来说，这与模型变小是同一回事。"

### 2.4 发现 Bug 的轶事

文章分享了一段调试故事：

> "我们发现了一个 bug，流水线数据却表现为*阻塞*速度，
> 追踪到是在构建约束解码掩码时发生了一次意外的同步拷贝。
> 将其移至拷贝流后，在 3090 上获得了 +11% 的提升，
> 在 B200 上获得了 +34% 的提升。"

这使工程过程更加人性化，并强化了这一教训：*热路径上的任何同步
都会扼杀收益*。

---

## 3. 未覆盖的领域（待解问题）

### 3.1 流水化如何与大规模批处理相互作用？

博客文章测量了 1、8、32 路并发，但 Photon 的生产级推理服务
可能要处理数百个并发请求。在非常大的 KV 缓存和大量活跃序列下，
管道如何扩展？

### 3.2 预填充-解码交织的细节

文章提到"预填充只是同一双槽管道中的另一种 `kind="prefill"` 步骤"，
但没有展示在每个 tick 中决定预填充还是解码的调度器逻辑 ——
这可能是一个复杂的调度决策。

### 3.3 图像预处理集成

Moondream 是一个视觉语言模型。博客文章聚焦于文本解码循环，
但没有讨论图像预处理（缩放、切片、编码）如何适配管道。
这是预填充延迟的重要来源，原则上也可以从流水化中受益。

### 3.4 多 GPU 影响

文章没有讨论跨多个 GPU 的张量并行或流水线并行。
流水化解码循环是否需要为多 GPU 设置做修改？

### 3.5 与其他推理引擎的对比

文章提到 Photon 比 vLLM 快 ~2×，但没有提供 vLLM 和 Photon
在批处理、调度和图捕获方法上的逐点对比。（这在博客文章中
是可以接受的，但在学术论文中会被期待。）

---

## 4. 对 AMD 移植的关键洞见

### 4.1 核心洞见与 GPU 架构无关

气泡之所以存在，是因为 CPU 簿记时间与 GPU 速度无关。
这在 AMD GPU 上和在 NVIDIA GPU 上一样成立。
流水化技术可以直接移植。

### 4.2 AMD 更高的内核发射开销使图更有价值

HIP 图捕获在 AMD 上比 CUDA 图在 NVIDIA 上*影响更大*，
因为 AMD 的每次内核发射开销约为 NVIDIA 的 2 倍。
图的折叠是 AMD 上延迟节省的主要来源。

### 4.3 僵尸税模型是通用的

$z \approx 1/(L \cdot B)$ 公式仅依赖于请求级别的统计信息
（平均长度、批量大小），与 GPU 架构无关。
相同的成本模型可以直接应用。

### 4.4 解码时的内存带宽主导地位

Decode 在 NVIDIA 和 AMD GPU 上都是受内存带宽限制的。
MI300X 的 5.3 TB/s 内存带宽介于 H100 (3.35 TB/s) 和 B200 (8.0 TB/s)
之间，因此 Photon 的加速比在批量模式下应在 ~15–25% 范围，
与基准测试表中 H100 那一行一致。

---

## 5. Photon-AMD 的实践要点

1. **预先分配一切。** AMD 的 `hipMalloc` 可能触发设备级同步 ——
   热路径中的每一次分配都是气泡来源。

2. **积极使用异步拷贝。** 将每一次 D2H 传输移至拷贝流，
   包括禁止掩码的上传。验证没有遗留的同步拷贝。

3. **对气泡进行性能分析。** 在优化之前，先用阻塞模式运行
   测量实际的气泡占比。博客文章直接分别测量 $T_f, T_s, T_b$ 的方法
   可以提供一个基线。

4. **按批量大小捕获图。** AMD HIP 图要求固定形状，
   但 Photon 为每个批量大小捕获一份图的方法已经处理了这一点。

5. **不要忘记僵尸税。** 对于单路测量，约 1% 的开销是可以察觉的。
   对于批量处理，它可忽略不计 —— 不要过度设计它。

---

## 6. 参考文献

1. 博客文章: [Popping the GPU Bubble](https://moondream.ai/blog/popping-the-gpu-bubble)
2. Photon 产品页面: [moondream.ai/p/photon](https://moondream.ai/p/photon)
3. Kestrel (Photon 参考实现): [github.com/m87-labs/kestrel](https://github.com/m87-labs/kestrel)
4. Photon 性能: [PERFORMANCE.md](https://github.com/m87-labs/kestrel/blob/main/PERFORMANCE.md)