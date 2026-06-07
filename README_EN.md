# Photon-AMD

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![AMD ROCm](https://img.shields.io/badge/ROCm-6.10%2B-red)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-AMD%20Instinct%20MI300-9cf)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

**High-efficiency LLM inference engine for AMD GPUs — pipelined decoding with GPU bubble elimination via ROCm/HIP.**

[中文](./README.md)

</div>

---

## Overview

**Photon-AMD** is an educational and practical implementation of Moondream's [Photon](https://moondream.ai/p/photon) inference engine, ported to AMD ROCm/HIP.

It demonstrates how to eliminate the **GPU bubble** — idle time where the GPU waits for CPU bookkeeping between decode steps — achieving **up to 56% higher decode throughput**.

This project serves two purposes:

1. **Learning resource**: Deep-dive analysis of Photon's three core mechanisms with mathematical cost models and empirical validation.
2. **Reference implementation**: Clean, well-documented AMD ROCm code as a starting point for production inference engines.

### Background

Photon is Moondream's purpose-built inference engine that achieves **~2× faster inference than vLLM** on comparable workloads. Originally implemented for NVIDIA GPUs (CUDA). This project analyses the Photon architecture from the blog post ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble), ports the three-mechanism pipelined decode loop to AMD ROCm/HIP, and validates the cost model on AMD hardware (gfx942 / MI300).

---

## Three Core Mechanisms

| # | Mechanism | File | Description |
|---|----------|------|-------------|
| 1 | **Ping-pong slots** | `decode_slot.py` | Two alternating buffer sets so step N+1 can run while step N's results are copied to CPU. |
| 2 | **Forward now, sample later** | `pipeline.py` | Forward launches before previous commit; sampling waits on constrained-decode mask. |
| 3 | **Zombies** | `zombie.py` | Finished requests ride one extra forward instead of requiring mid-flight cancellation. |

The speedup formula:

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

| Hardware | Streams | Blocking | Pipelined | Speedup |
|----------|---------|----------|-----------|---------|
| RTX 3090 | 32 | 11.74 ms | 10.52 ms | **+11.6%** |
| B200 | 32 | 5.55 ms | 3.98 ms | **+35.4%** |
| MI300X (AMD) | 2 | 0.47 ms/tok | 0.30 ms/tok | **+56.1%** |

---

## Quick Start

Requirements: **AMD GPU** (CDNA3 / gfx942 / MI300), **ROCm** 6.10+ / PyTorch 2.5+, **Python** 3.10–3.12.

```bash
git clone https://github.com/Yunle-Lee/Babe-Photon.git
cd Babe-Photon
pip install -e ".[dev]"
```

Run benchmark:
```bash
python -m photon_amd.benchmark                 # default: 8 requests x 128 tokens
python examples/bench_real_model.py            # PyTorch Transformer + Photon pipeline
```

Run real model:
```bash
# Requires: pip install llama-cpp-python
python examples/demo_real_model.py             # Gemma-4 12B through Photon pipeline
```

Programmatic usage:
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

## Architecture

```
                         ┌──────────────────────────┐
                         │     PhotonEngine          │
                         │  pipeline orchestration   │
                         ├──────────────────────────┤
                         │  StreamManager            │
                         │  compute + copy streams   │
                         ├──────────────────────────┤
                         │  Scheduler                │
                         │  tick: launch->commit->   │
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

Stream model:
```
compute stream ──┬── Forward(N) ──────┬── Forward(N+1) ──┬── ...
                 │                     │                    │
copy stream    ──┴── (idle) ───────────┴── D2H copy(N) ────┴── ...
                    wait                  wait
                 step_done_event       step_done_event
```

Key insight: the D2H copy goes on a separate stream, so the next forward can start immediately — the bubble is eliminated.

---

## Project Structure

```
Babe-Photon/
├── README.md                          # 中文版
├── README_EN.md                       # English version (this file)
├── LICENSE                            # Apache 2.0
├── pyproject.toml
├── .github/workflows/tests.yml        # CI
│
├── docs/                              # English docs
│   └── zh/                            # Chinese docs
│
├── photon_amd/                        # core implementation
│   ├── __init__.py                    # public API
│   ├── config.py                      # configuration
│   ├── stream_manager.py              # HIP stream & event mgmt
│   ├── decode_slot.py                 # ping-pong slot buffers
│   ├── graph.py                       # HIP graph capture & replay
│   ├── zombie.py                      # zombie lifecycle
│   ├── scheduler.py                   # batch assembly & tick
│   ├── pipeline.py                    # launch / commit / finalize
│   ├── benchmark.py                   # blocking vs pipelined benchmark
│   └── adapter.py                     # llama-cpp-python model adapter
│
├── examples/
│   ├── demo_pipelined.py              # blocking vs pipelined demo
│   ├── demo_real_model.py             # real Gemma-4 12B inference
│   ├── bench_real_gpu.py              # GPU matmul proxy benchmark
│   └── bench_real_model.py            # PyTorch Transformer benchmark
│
└── tests/
    ├── test_zombie.py                 # zombie state machine
    ├── test_scheduler.py              # scheduler & batch
    └── test_slots.py                  # GPU buffer & slot (GPU required)
```

---

## Benchmarks

### Real model inference (Gemma-4 12B)

```
[2+2]    ~5 tok/s -> "4"
[who]    ~6 tok/s -> "I am Gemma 4, a large language model developed by Google."
[haiku]  ~7 tok/s -> "Lines of code flow fast, Logic builds a complex world, Errors fade away."
```

### Pipeline speedup (synthetic GPU matmul proxy)

| Load | Blocking | Pipelined | Speedup |
|------|----------|-----------|---------|
| 2 req x 128 tok | 2129 tok/s | 3323 tok/s | **+56.1%** |
| 4 req x 128 tok | 6013 tok/s | 5992 tok/s | -0.3% |

The speedup is largest when CPU bookkeeping is a meaningful fraction of step time (fewer concurrent requests -> larger bubble -> more to hide).

---

## Testing

```bash
pytest tests/ -v                    # all tests
pytest tests/ -v -k "not gpu"      # CPU-only
pytest tests/ -v -k gpu            # GPU tests (AMD + ROCm)
```

---

## Porting Notes

Photon-AMD runs on AMD GPUs through PyTorch's ROCm backend, which provides transparent CUDA->HIP mapping:

| CUDA API | ROCm/HIP | PyTorch API |
|----------|----------|-------------|
| `cudaStream_t` | `hipStream_t` | `torch.cuda.Stream` |
| `cudaEvent_t` | `hipEvent_t` | `torch.cuda.Event` |
| `cudaGraph_t` | `hipGraph_t` | `torch.cuda.CUDAGraph` |

Key differences: HIP graph capture requires stricter fixed-shape / pre-allocated buffer patterns (already satisfied by Photon's design). Runtime `hipMalloc` can trigger device-wide sync — all buffers MUST be pre-allocated. `float16` is recommended over `bfloat16` on gfx942 for optimal bandwidth.

---

## References

- **Moondream Blog**: ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) (2026-06-04)
- **Moondream Docs**: [Running Locally](https://docs.moondream.ai/running-locally)
- **Kestrel** (Photon reference): [github.com/m87-labs/kestrel](https://github.com/m87-labs/kestrel)
- **Moondream**: [github.com/m87-labs/moondream](https://github.com/m87-labs/moondream)
- **AMD ROCm**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)

---

## License

Apache 2.0 — see [LICENSE](./LICENSE). All credit for the Photon architecture and the three-mechanism design goes to **Moondream (M87 Labs)**. This project is an independent educational implementation and port.

---

## Appendix: English Deep-Dive Documents

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