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
