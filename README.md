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
