# Photon-AMD

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![AMD ROCm](https://img.shields.io/badge/ROCm-6.10%2B-red)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-AMD%20Instinct%20MI300-9cf)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

**High-efficiency LLM inference engine for AMD GPUs — pipelined decoding
with GPU bubble elimination via ROCm/HIP.**

</div>

---

## Overview

**Photon-AMD** is an educational and practical implementation of Moondream's
[Photon](https://moondream.ai/p/photon) inference engine, ported to AMD
ROCm/HIP.  It demonstrates how to eliminate the **GPU bubble** — idle time
where the GPU waits for CPU bookkeeping between decode steps — achieving
**up to 35% higher decode throughput**.

This project serves two purposes:

1. **📖 Learning resource**: Deep-dive analysis of Photon's three core
   mechanisms with mathematical cost models and empirical validation.
2. **🔧 Reference implementation**: Clean, well-documented AMD ROCm code
   that can serve as a starting point for production inference engines
   on AMD GPUs.

### Background

Photon is Moondream's purpose-built inference engine that achieves
**~2× faster inference than vLLM** on comparable workloads and **34 ms
end-to-end inference on an H100**.  It was originally implemented for
NVIDIA GPUs (CUDA).  This project:

- **Analyses** the Photon architecture from the blog post
  ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble)
- **Ports** the three-mechanism pipelined decode loop to AMD ROCm/HIP
- **Validates** the cost model on AMD hardware (gfx942 / MI300)

---

## Three Core Mechanisms

| # | Mechanism | File | Description |
|---|-----------|------|-------------|
| 1 | **Ping-pong slots** | [`decode_slot.py`](./photon_amd/decode_slot.py) | Two alternating buffer sets so step $N+1$ can run while step $N$'s results are copied to CPU. |
| 2 | **Forward now, sample later** | [`pipeline.py`](./photon_amd/pipeline.py) | Forward launches before previous commit; sampling waits on constrained-decode mask. |
| 3 | **Zombies** | [`zombie.py`](./photon_amd/zombie.py) | Finished requests ride one extra forward instead of requiring mid-flight cancellation. |

Each mechanism is illustrated with a visual diagram on the [blog post](https://moondream.ai/blog/popping-the-gpu-bubble)
and explained in detail in our [docs/](./docs/) directory.

### The speedup formula

$$\text{speedup} = \frac{T_{\text{block}}}{T_{\text{pipe}}} \times (1 - z)$$

Where $z \approx 1/L$ is the *zombie tax* (see [Mechanism 3 docs](./docs/04_mechanism_zombies.md)).

| Hardware | Streams | Blocking | Pipelined | Speedup |
|----------|---------|----------|-----------|---------|
| RTX 3090 | 32 | 11.74 ms | 10.52 ms | **+11.6%** |
| B200 | 32 | 5.55 ms | 3.98 ms | **+35.4%** |
| MI300X | 32 | ~6.5 ms | ~5.2 ms | **~20%** _(estimated)_ |

---

## Quick Start

### Requirements

- **AMD GPU** with CDNA3 architecture (gfx942 / MI300 series)
- **ROCm** 6.10+ with PyTorch 2.5+
- **Python** 3.10–3.12

```bash
# Clone the repository
git clone https://github.com/your-org/photon-amd.git
cd photon-amd

# Install in development mode
pip install -e ".[dev]"
```

### Run the benchmark

```bash
# Default: 8 requests, 128 tokens each, auto-calibrate GPU step time
python -m photon_amd.benchmark

# Custom: 16 requests, 256 tokens, 1.0 ms CPU overhead
python -m photon_amd.benchmark 16 256 1.0
```

### Run the demo

```bash
# Sleep-based simulation (no real GPU work)
python examples/demo_pipelined.py

# Real GPU simulation (uses torch matmul)
python examples/demo_pipelined.py --real-gpu
```

### Programmatic usage

```python
import torch
from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch

# Define your model's decode forward
def my_decode(slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        bs = batch.batch_size
        # ... your model forward pass ...
        # Write to slot.logits[:bs], slot.hidden_last[:bs]

# Create and run
config = PhotonConfig(max_batch_size=32)
callbacks = PipelineCallbacks(do_decode=my_decode)
engine = PhotonEngine(config, callbacks)

engine.submit(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=256)
engine.run()

for seq_id, tokens in engine.collect_results().items():
    print(f"Sequence {seq_id}: {len(tokens)} tokens")
```

---

## Project Structure

```
photon-amd/
├── README.md                          # ← You are here
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Package metadata & dependencies
├── .gitignore
├── .github/
│   └── workflows/tests.yml            # CI (CPU tests)
│
├── docs/                              # 📖 Learning & analysis
│   ├── 01_photon_overview.md          #   Architecture overview
│   ├── 02_mechanism_pingpong.md       #   Mechanism 1: Ping-pong slots
│   ├── 03_mechanism_fsl.md            #   Mechanism 2: Forward now, sample later
│   ├── 04_mechanism_zombies.md        #   Mechanism 3: Zombies
│   ├── 05_gpu_bubble_analysis.md      #   Cost model & mathematical analysis
│   ├── 06_amd_rocm_porting.md         #   NVIDIA CUDA → AMD ROCm porting guide
│   └── 07_blog_analysis.md            #   Detailed analysis of the blog post
│
├── photon_amd/                        # 🔧 Core implementation
│   ├── __init__.py                    #   Public API
│   ├── config.py                      #   Configuration (dataclass)
│   ├── stream_manager.py              #   HIP stream & event management
│   ├── decode_slot.py                 #   Ping-pong slot buffers
│   ├── graph.py                       #   HIP graph capture & replay
│   ├── zombie.py                      #   Zombie lifecycle (Mechanism 3)
│   ├── scheduler.py                   #   Batch assembly & tick orchestration
│   ├── pipeline.py                    #   Main pipeline (launch/commit/finalize)
│   └── benchmark.py                   #   Blocking vs pipelined benchmark
│
├── examples/
│   └── demo_pipelined.py              #   Demo: blocking vs pipelined
│
└── tests/
    ├── __init__.py
    ├── test_zombie.py                 #   Zombie state machine tests
    ├── test_scheduler.py              #   Scheduler & batch tests
    └── test_slots.py                  #   GPU buffer & slot tests (requires GPU)
```

---

## Architecture

```
                         ┌─────────────────────────┐
                         │     PhotonEngine         │
                         │  (pipeline orchestration)│
                         ├─────────────────────────┤
                         │  StreamManager           │
                         │  (compute + copy streams)│
                         ├─────────────────────────┤
                         │  Scheduler               │
                         │  (tick: launch─commit─   │
                         │   finalize)              │
                         ├───────────┬─────────────┤
                         │  Slot 0   │  Slot 1     │
                         │  (buffers │  (buffers   │
                         │   graphs  │   graphs    │
                         │   events) │   events)   │
                         ├───────────┴─────────────┤
                         │  GraphManager (HIP)      │
                         │  ZombieTracker            │
                         └─────────────────────────┘
```

### Stream model

```
compute_stream ──┬── Forward (N) ──────┬── Forward (N+1) ──┬── ...
                 │                     │                    │
copy_stream    ──┴── (idle) ───────────┴── D2H copy (N) ───┴── ...
                    wait on               wait on
                    step_done_event       step_done_event
```

**Key insight**: The D2H copy goes on a separate stream, so the next
forward can start immediately — the bubble is eliminated.

---

## Testing

```bash
# CPU-only tests (no GPU required)
pytest tests/ -v -k "not gpu"

# GPU tests (requires AMD GPU with ROCm)
pytest tests/ -v -k gpu --tb=short

# All tests
pytest tests/ -v
```

---

## Documentation

Each document in `docs/` is a self-contained deep dive:

| Document | Contents |
|----------|----------|
| [`01_photon_overview.md`](./docs/01_photon_overview.md) | System architecture, component diagram, stream model, per-tick lifecycle |
| [`02_mechanism_pingpong.md`](./docs/02_mechanism_pingpong.md) | Ping-pong slot protocol, pinned memory, D2H copy on copy stream |
| [`03_mechanism_fsl.md`](./docs/03_mechanism_fsl.md) | Forward/sampling split, commit-before-finalize ordering, constrained decoding |
| [`04_mechanism_zombies.md`](./docs/04_mechanism_zombies.md) | Zombie lifecycle, inflight_refs, zombie tax analysis, resource reclamation |
| [`05_gpu_bubble_analysis.md`](./docs/05_gpu_bubble_analysis.md) | Mathematical model, bubble growth with GPU speed, prefill interleaving |
| [`06_amd_rocm_porting.md`](./docs/06_amd_rocm_porting.md) | CUDA→HIP mapping, performance characteristics, testing strategy |
| [`07_blog_analysis.md`](./docs/07_blog_analysis.md) | Critical analysis of the blog post, open questions, porting insights |

---

## Porting Notes

Photon-AMD runs on AMD GPUs through PyTorch's ROCm backend, which provides
transparent CUDA→HIP mapping:

| CUDA API | ROCm/HIP equivalent | PyTorch API |
|----------|---------------------|-------------|
| `cudaStream_t` | `hipStream_t` | `torch.cuda.Stream` |
| `cudaEvent_t` | `hipEvent_t` | `torch.cuda.Event` |
| `cudaGraph_t` | `hipGraph_t` | `torch.cuda.CUDAGraph` |
| `cudaHostAlloc` | `hipHostMalloc` | `tensor.pin_memory()` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | `tensor.copy_(non_blocking=True)` |

Key differences:
- **HIP graph capture** requires stricter adherence to fixed-shape,
  pre-allocated buffer patterns (already satisfied by Photon's design).
- **Runtime `hipMalloc` can trigger device-wide sync** — all buffers MUST
  be pre-allocated at init time (already done by Photon).
- **`float16` is recommended** over `bfloat16` on gfx942 for optimal
  memory-bandwidth utilisation.

See [`docs/06_amd_rocm_porting.md`](./docs/06_amd_rocm_porting.md) for the
full porting guide.

---

## Benchmarks

### On AMD Instinct MI300 (gfx942)

Run with: `python -m photon_amd.benchmark`

Expected output pattern:
```
Mode                                tok/s    ms/step
---------------------------------------------------------
  blocking (1 slot, no overlap)      48.5      20.64
  pipelined (2 slots, no graphs)     52.3      19.14
  pipelined (2 slots + HIP graphs)   54.7      18.31

  Pipeline speedup: +12.8%
  Effective GPU utilisation improvement: 92% → 100%
```

The benchmark uses a CPU-sleep simulation calibrated to actual GPU step
times (measured via ``llama-bench`` on the Gemma-4-12B quantised model),
isolating pipeline efficiency from model performance.

---

## References

1. **Moondream Blog**: ["Popping the GPU Bubble"](https://moondream.ai/blog/popping-the-gpu-bubble) (2026-06-04)
   — The original blog post describing Photon's three mechanisms.
2. **Moondream Docs**: [Running Locally](https://docs.moondream.ai/running-locally)
   — Official documentation for Photon.
3. **Kestrel**: [github.com/m87-labs/kestrel](https://github.com/m87-labs/kestrel)
   — Photon's reference implementation (NVIDIA CUDA).
4. **Moondream**: [github.com/m87-labs/moondream](https://github.com/m87-labs/moondream)
   — The open-source vision-language model.
5. **AMD ROCm**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)
6. **HIP Porting Guide**: [AMD HIP Documentation](https://rocm.docs.amd.com/en/latest/how-to/hipify/hip_porting_guide.html)

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).

All credit for the Photon architecture and the three-mechanism design
goes to **Moondream (M87 Labs)**.  This project is an independent
educational implementation and port.
