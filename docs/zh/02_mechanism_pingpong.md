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
