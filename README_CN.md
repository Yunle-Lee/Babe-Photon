# Photon-AMD

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![AMD ROCm](https://img.shields.io/badge/ROCm-6.10%2B-red)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-AMD%20Instinct%20MI300-9cf)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

**面向 AMD GPU 的高效 LLM 推理引擎 —— 基于 ROCm/HIP 的流水化解码
以消除 GPU 气泡。**

📖 [English README](./README.md)

</div>

---

## 概述

**Photon-AMD** 是 Moondream [Photon](https://moondream.ai/p/photon) 推理引擎的教育与实践实现，
已移植至 AMD ROCm/HIP。它演示了如何消除 **GPU 气泡** —— 即解码步骤之间
GPU 等待 CPU 完成簿记工作的空闲时间 —— 从而将解码吞吐量**提升高达 35%**。

本项目兼具两大用途：

1. **📖 学习资源**：对 Photon 三个核心机制进行深度分析，
   附有数学成本模型与实验验证。
2. **🔧 参考实现**：清晰、文档完备的 AMD ROCm 代码，
   可作为 AMD GPU 上生产推理引擎的起点。

### 项目背景

Photon 是 Moondream 专为视觉语言模型打造的推理引擎，
在可比工作负载下比 vLLM 快约 **~2×**，
在 H100 上实现 **34 毫秒**端到端推理。它最初为 NVIDIA GPU (CUDA) 实现。
本项目的工作是：

- **解析** 博客文章
  ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) 中描述的 Photon 架构
- **移植** 三机制流水化解码循环至 AMD ROCm/HIP
- **验证** AMD 硬件（gfx942 / MI300）上的成本模型

---

## 三大核心机制

| # | 机制 | 代码文件 | 描述 |
|---|------|----------|------|
| 1 | **乒乓槽位** | [`decode_slot.py`](./photon_amd/decode_slot.py) | 两套交替缓冲区，使步骤 N+1 的计算可与步骤 N 的结果回传重叠执行。 |
| 2 | **先行推理，后采样** | [`pipeline.py`](./photon_amd/pipeline.py) | Forward 在上一轮 Commit 之前发射；采样等待约束解码掩码就绪后再执行。 |
| 3 | **僵尸序列** | [`zombie.py`](./photon_amd/zombie.py) | 已完成请求多跑一次前向推理作为"僵尸"，避免中途取消的复杂逻辑。 |

每个机制均在[博客文章](https://moondream.ai/blog/popping-the-gpu-bubble)中配有可视化图表，
并在 `docs/zh/` 目录中有详细的中文解读。

### 加速比公式

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

其中 $z \approx 1/L$ 为*僵尸税*（详见[机制三中文文档](./docs/zh/04_mechanism_zombies.md)）。

| 硬件 | 并发数 | 阻塞模式 | 流水线模式 | 加速比 |
|----------|------|----------|-----------|---------|
| RTX 3090 | 32 | 11.74 ms | 10.52 ms | **+11.6%** |
| B200 | 32 | 5.55 ms | 3.98 ms | **+35.4%** |
| MI300X | 32 | ~6.5 ms | ~5.2 ms | **~20%** _(估计)_ |

---

## 快速开始

### 环境要求

- **AMD GPU**，CDNA3 架构 (gfx942 / MI300 系列)
- **ROCm** 6.10+ 及 PyTorch 2.5+
- **Python** 3.10–3.12

```bash
# 克隆仓库
git clone https://github.com/your-org/photon-amd.git
cd photon-amd

# 开发模式安装
pip install -e ".[dev]"
```

### 运行基准测试

```bash
# 默认：8 个请求，每个 128 token，自动校准 GPU 步时间
python -m photon_amd.benchmark

# 自定义：16 请求，256 token，1.0 ms CPU 开销
python -m photon_amd.benchmark 16 256 1.0
```

### 运行演示

```bash
# 基于 sleep 的模拟（无真实 GPU 计算）
python examples/demo_pipelined.py

# 真实 GPU 模拟（使用 torch matmul）
python examples/demo_pipelined.py --real-gpu
```

### 编程调用

```python
import torch
from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch

# 定义模型的解码 forward
def my_decode(slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        bs = batch.batch_size
        # ... 模型前向传播 ...
        # 将结果写入 slot.logits[:bs], slot.hidden_last[:bs]

# 创建并运行
config = PhotonConfig(max_batch_size=32)
callbacks = PipelineCallbacks(do_decode=my_decode)
engine = PhotonEngine(config, callbacks)

engine.submit(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=256)
engine.run()

for seq_id, tokens in engine.collect_results().items():
    print(f"序列 {seq_id}: 生成 {len(tokens)} 个 token")
```

---

## 项目结构

```
photon-amd/
├── README.md                          # ← 英文说明（你在此处）
├── README_CN.md                       # ← 中文说明
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # 包元信息与依赖
├── .gitignore
├── .github/
│   └── workflows/tests.yml            # CI（CPU 测试）
│
├── docs/                              # 📖 学习与分析（英文）
│   ├── 01_photon_overview.md          #   架构概览
│   ├── 02_mechanism_pingpong.md       #   机制一：乒乓槽位
│   ├── 03_mechanism_fsl.md            #   机制二：先行推理，后采样
│   ├── 04_mechanism_zombies.md        #   机制三：僵尸序列
│   ├── 05_gpu_bubble_analysis.md      #   成本模型与数学分析
│   ├── 06_amd_rocm_porting.md         #   NVIDIA CUDA → AMD ROCm 移植指南
│   ├── 07_blog_analysis.md            #   博客文章详细分析
│   └── zh/                            # 📖 学习与分析（中文）
│       ├── 01_photon_overview.md
│       ├── 02_mechanism_pingpong.md
│       ├── 03_mechanism_fsl.md
│       ├── 04_mechanism_zombies.md
│       ├── 05_gpu_bubble_analysis.md
│       ├── 06_amd_rocm_porting.md
│       └── 07_blog_analysis.md
│
├── photon_amd/                        # 🔧 核心实现
│   ├── __init__.py                    #   公共 API
│   ├── config.py                      #   配置（dataclass）
│   ├── stream_manager.py              #   HIP 流与事件管理
│   ├── decode_slot.py                 #   乒乓槽位缓冲区
│   ├── graph.py                       #   HIP 图捕获与重放
│   ├── zombie.py                      #   僵尸生命周期（机制三）
│   ├── scheduler.py                   #   批量组装与 tick 编排
│   ├── pipeline.py                    #   主管线 (launch/commit/finalize)
│   └── benchmark.py                   #   阻塞 vs 流水线基准测试
│
├── examples/
│   └── demo_pipelined.py              #   演示：阻塞 vs 流水线
│
└── tests/
    ├── __init__.py
    ├── test_zombie.py                 #   僵尸状态机测试
    ├── test_scheduler.py              #   调度器与批量测试
    └── test_slots.py                  #   GPU 缓冲区与槽位测试（需 GPU）
```

---

## 架构

```
                         ┌──────────────────────────┐
                         │     PhotonEngine          │
                         │  (流水线编排)              │
                         ├──────────────────────────┤
                         │  StreamManager            │
                         │  (计算流 + 拷贝流)          │
                         ├──────────────────────────┤
                         │  Scheduler                │
                         │  (tick: launch→commit→    │
                         │   finalize)               │
                         ├──────────┬───────────────┤
                         │  Slot 0  │  Slot 1       │
                         │  (缓冲区  │  (缓冲区      │
                         │   图对象  │   图对象      │
                         │   事件)   │   事件)       │
                         ├──────────┴───────────────┤
                         │  GraphManager (HIP)       │
                         │  ZombieTracker            │
                         └──────────────────────────┘
```

### 流模型

```
计算流 (compute) ──┬── Forward(N) ──────┬── Forward(N+1) ──┬── ...
                   │                    │                  │
拷贝流 (copy)    ──┴── (空闲) ──────────┴── D2H拷贝(N) ───┴── ...
                     等待                   等待
                     step_done_event        step_done_event
```

**核心洞见**：设备→主机 (D2H) 拷贝在独立的拷贝流上执行，因此下一个
前向推理可以立即开始 —— 气泡被消除了。

---

## 测试

```bash
# 仅 CPU 测试（无需 GPU）
pytest tests/ -v -k "not gpu"

# GPU 测试（需要 AMD GPU + ROCm）
pytest tests/ -v -k gpu --tb=short

# 全部测试
pytest tests/ -v
```

---

## 文档索引

`docs/` 和 `docs/zh/` 中的每份文档均为独立专题：

| 文档 | 内容 |
|----------|------|
| [`01_photon_overview.md`](./docs/zh/01_photon_overview.md) | 系统架构、组件图、流模型、每 tick 生命周期 |
| [`02_mechanism_pingpong.md`](./docs/zh/02_mechanism_pingpong.md) | 乒乓槽位协议、钉住内存、拷贝流上的 D2H 拷贝 |
| [`03_mechanism_fsl.md`](./docs/zh/03_mechanism_fsl.md) | 前向/采样分离、先提交后定稿排序、约束解码 |
| [`04_mechanism_zombies.md`](./docs/zh/04_mechanism_zombies.md) | 僵尸生命周期、飞行引用计数、僵尸税分析、资源回收 |
| [`05_gpu_bubble_analysis.md`](./docs/zh/05_gpu_bubble_analysis.md) | 数学模型、气泡随 GPU 速度增长、预填充交织 |
| [`06_amd_rocm_porting.md`](./docs/zh/06_amd_rocm_porting.md) | CUDA→HIP 映射、性能特性、测试策略 |
| [`07_blog_analysis.md`](./docs/zh/07_blog_analysis.md) | 对博客文章的分析与批判、待解问题、移植洞见 |

---

## 移植说明

Photon-AMD 通过 PyTorch 的 ROCm 后端在 AMD GPU 上运行，利用其透明的 CUDA→HIP 映射：

| CUDA API | ROCm/HIP 等价 | PyTorch API |
|----------|---------------|-------------|
| `cudaStream_t` | `hipStream_t` | `torch.cuda.Stream` |
| `cudaEvent_t` | `hipEvent_t` | `torch.cuda.Event` |
| `cudaGraph_t` | `hipGraph_t` | `torch.cuda.CUDAGraph` |
| `cudaHostAlloc` | `hipHostMalloc` | `tensor.pin_memory()` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | `tensor.copy_(non_blocking=True)` |

主要差异：
- **HIP 图捕获** 对固定形状、预分配缓冲区的模式要求更严格（Photon 的设计天然符合）。
- **运行时 `hipMalloc` 可能触发设备级同步** —— 所有缓冲区**必须**在初始化时预分配（Photon 已做到）。
- **建议使用 `float16`** 而非 `bfloat16` 以获得 gfx942 上的最佳内存带宽利用率。

详见 [`docs/zh/06_amd_rocm_porting.md`](./docs/zh/06_amd_rocm_porting.md)。

---

## 基准测试

### AMD Instinct MI300 (gfx942)

执行：`python -m photon_amd.benchmark`

预期输出模式：
```
Mode                                tok/s    ms/step
---------------------------------------------------------
  blocking (1 slot, no overlap)      48.5      20.64
  pipelined (2 slots, no graphs)     52.3      19.14
  pipelined (2 slots + HIP graphs)   54.7      18.31

  Pipeline speedup: +12.8%
  Effective GPU utilisation improvement: 92% → 100%
```

该基准测试使用基于 CPU-sleep 的仿真，GPU 步时间根据实际测量值
（通过 `llama-bench` 对 Gemma-4-12B 量化模型）进行校准，
以此将流水线效率与模型性能解耦。

---

## 参考文献

1. **Moondream 博客**: ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) (2026-06-04)
   —— 描述 Photon 三大机制的原始博客文章。
2. **Moondream 文档**: [本地运行](https://docs.moondream.ai/running-locally)
   —— Photon 的官方文档。
3. **Kestrel**: [github.com/m87-labs/kestrel](https://github.com/m87-labs/kestrel)
   —— Photon 的参考实现 (NVIDIA CUDA)。
4. **Moondream**: [github.com/m87-labs/moondream](https://github.com/m87-labs/moondream)
   —— 开源视觉语言模型。
5. **AMD ROCm**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)
6. **HIP 移植指南**: [AMD HIP 文档](https://rocm.docs.amd.com/en/latest/how-to/hipify/hip_porting_guide.html)

---

## 许可证

Apache 2.0 —— 详见 [LICENSE](./LICENSE)。

Photon 架构及三大机制设计的全部功劳归 **Moondream (M87 Labs)** 所有。
本项目为独立的教育实现与移植。
