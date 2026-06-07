"""
Photon-AMD: High-efficiency LLM inference engine for AMD GPUs.

Photon implements the pipelined-decoding architecture described in
Moondream's "Popping the GPU Bubble" blog post (June 2026). It eliminates
GPU idle time (bubbles) between decode steps by overlapping CPU bookkeeping
with GPU forward passes using three mechanisms:

  1. **Ping-pong slots** — two alternating buffer sets so step N+1 can
     run while step N's results are being copied back to the CPU.
  2. **Forward now, sample later** — the forward pass is launched before
     the previous step is committed. Sampling waits on the constrained-decode
     mask built during commit (commit-before-finalize ordering).
  3. **Zombies** — finished requests ride one extra forward instead of
     requiring mid-flight cancellation. Refcounted, released when safe.

The implementation is adapted for AMD ROCm/HIP using PyTorch's CUDA-API
compatibility layer, which maps transparently to HIP on AMD hardware.
HIP graph capture (via torch.cuda.CUDAGraph) replaces CUDA graphs for
kernel-launch overhead reduction.

Target hardware: AMD Instinct MI300 (gfx942) and other CDNA3 accelerators.

Usage:

    from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks

    config = PhotonConfig(max_batch_size=32)
    engine = PhotonEngine(config, callbacks)
    seq_id = engine.submit(prompt_tokens)
    engine.run()
    results = engine.collect_results()

References:
  - Moondream Blog: "Popping the GPU Bubble" (2026-06-04)
    https://moondream.ai/blog/popping-the-gpu-bubble
  - Moondream Docs: https://docs.moondream.ai
  - Kestrel (Photon reference implementation): https://github.com/m87-labs/kestrel
"""

from .config import PhotonConfig
from .pipeline import PhotonEngine, PipelineCallbacks, _ZOMBIE_SENTINEL

__all__ = [
    "PhotonConfig",
    "PhotonEngine",
    "PipelineCallbacks",
    "_ZOMBIE_SENTINEL",
]

__version__ = "0.2.0"
