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
