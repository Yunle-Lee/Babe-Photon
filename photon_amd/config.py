"""Photon inference engine configuration.

All GPU-capacity and pipeline-behaviour parameters are centralised here
so the scheduler, slot allocator, and pipeline can be sized consistently
without implicit coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PhotonConfig:
    """Configuration for a Photon inference engine instance.

    Attributes:
        max_batch_size: Maximum concurrent sequences (batch slots).
            Includes reserved slot 0 for padding when needed.
        max_seq_len: Maximum sequence length in tokens.
        max_prefill_tokens: Hard cap on prefill tokens per step, used
            to bound prefill latency and keep decode responsive.
        num_slots: Number of ping-pong decode slots. Fixed at 2 for the
            Photon pipeline; exposed for testing single-slot (blocking) mode
            to measure the bubble.
        vocab_size: Model vocabulary size for logits buffer allocation.
        hidden_dim: Model hidden dimension for last-hidden-state buffer.
        dtype: Model compute dtype. float16 is recommended for ROCm.
            bfloat16 is also supported on MI300 (gfx942).
        coord_dtype: Dtype for structured-output coordinate values
            (e.g. bounding-box coordinates).
        size_dtype: Dtype for structured-output size values.
        max_loras: Maximum concurrent LoRA adapters. 0 = no LoRA support.
        kv_cache_pages: Number of paged-KV-cache pages. Each page holds
            `page_size` tokens of key/value state per sequence.
        page_size: Tokens per KV cache page. Typical values: 1 (fine-grained),
            16, 32 (trade memory for fewer page-table entries).
        capture_graphs: Whether to capture HIP graphs for the decode step.
            Strongly recommended; eliminates per-step kernel-launch overhead
            on AMD GPUs.  On gfx942 (MI300), HIP graph capture is supported
            and stable for fixed-shape workloads.
        graph_batch_sizes: Explicit batch sizes for which to capture graphs.
            If None, captures for [1, 2, 4, 8, ..., max_batch_size].
        enable_zombies: Whether to pipeline finished sequences as zombies
            for one extra step. Disabling gives blocking behaviour.
        use_async_copy: Whether to use a separate copy stream for D2H
            transfers (enables the pipeline overlap). Without this, the
            pipeline degrades to blocking mode.

    Platform notes (AMD ROCm):
        - HIP graph capture requires all buffers to be pre-allocated with
          fixed addresses — the ping-pong slots satisfy this.
        - Dynamic shapes and host-callbacks during capture are unsupported
          on HIP; we avoid both.
        - Use float16 for optimal memory-bandwidth utilisation on MI300.
    """

    # ---- GPU resource caps ------------------------------------------------
    max_batch_size: int = 32
    max_seq_len: int = 4096
    max_prefill_tokens: int = 4096

    # ---- Pipeline depth ---------------------------------------------------
    num_slots: int = 2

    # ---- Model dimensions -------------------------------------------------
    vocab_size: int = 256000
    hidden_dim: int = 3840
    eos_token_id: int = 0    # End-of-sequence token (0 for most tokenizers)
    dtype: torch.dtype = torch.float16
    coord_dtype: torch.dtype = torch.float32
    size_dtype: torch.dtype = torch.float32

    # ---- Advanced features ------------------------------------------------
    max_loras: int = 0
    kv_cache_pages: int = 4096
    page_size: int = 1

    # ---- Performance ------------------------------------------------------
    capture_graphs: bool = True
    graph_batch_sizes: Optional[list[int]] = None
    enable_zombies: bool = True
    use_async_copy: bool = True

    def __post_init__(self):
        if self.graph_batch_sizes is None and self.capture_graphs:
            sizes: list[int] = []
            n = 1
            while n <= self.max_batch_size:
                sizes.append(n)
                n *= 2
            if sizes and sizes[-1] != self.max_batch_size:
                sizes.append(self.max_batch_size)
            self.graph_batch_sizes = sizes

    @property
    def device(self) -> torch.device:
        """Default device for all tensors in this engine."""
        return torch.device("cuda:0")

    @property
    def bubble_fraction(self) -> float:
        """Theoretical fraction of decode-step time lost to the GPU bubble
        in blocking mode.  This is a design-time estimate; the actual
        value depends on model weight size and memory bandwidth."""
        return 0.0  # Computed at runtime from measured step timings.
