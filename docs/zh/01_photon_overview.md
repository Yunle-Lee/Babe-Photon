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
