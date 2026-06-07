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
