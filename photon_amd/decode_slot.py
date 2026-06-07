"""Per-slot buffer resources for pipelined decoding.

Each ``DecodeSlot`` bundles the GPU and pinned-host resources for one decode
step.  With two slots, the scheduler can pipeline: while slot A's forward
runs on the GPU, slot B's D2H transfer completes on the copy stream
and the CPU commits its results.

Ping-pong protocol
------------------
  Slot 0 → Slot 1 → Slot 0 → ...

  A slot is "in use" while its ``step_done_event`` has not been waited
  on (D2H copy in flight) or its ``commit_done_event`` has not been
  recorded.  The scheduler's launch → commit → finalize cycle ensures a
  slot is never reused until the CPU has finished reading from its
  pinned host buffers.

Buffer catalogue
----------------
  GPU-side (device memory):
    - ``decode_token_ids`` — token ids input to the forward  [max_batch]
    - ``logits``           — forward output scores [max_batch x vocab]
    - ``hidden_last``      — last hidden state [max_batch x hidden_dim]
    - ``sampled_ids``      — sampled token id per row [max_batch]
    - ``sampled_logprobs`` — log-probability of sampled token [max_batch]
    - ``fa3_page_table``   — paged-KV page table [max_batch x pages]
    - ``fa3_seqused_k``    — per-sequence KV lengths [max_batch]

  Pinned-host-side (page-locked, accessible by both CPU and GPU DMA):
    - ``batch_idx``        — sequence batch indices [max_batch]
    - ``input_pos``        — token positions [max_batch]
    - ``disallow_mask``    — constrained-decode vocabulary mask
                             [max_batch x vocab] (bool)

Why pinned memory?
  Page-locked (pinned) host memory allows the GPU's DMA engine to
  copy data without blocking the CPU.  Without pinning, every D2H
  transfer would require the driver to first copy into a pinned buffer,
  serialising the copy and the next kernel launch.

AMD ROCm notes
--------------
  - PyTorch's ``pin_memory=True`` is supported on ROCm and uses
    ``hipHostMalloc`` under the hood.
  - GPU buffer allocation at init time (not in the hot path) avoids
    runtime ``hipMalloc`` calls which can trigger device-wide sync.
  - HIP graph capture requires fixed buffer addresses — satisfied by
    one-time allocation in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CpuGpuBuffer:
    """A buffer with both pinned-host and device-memory views.

    Pinned (page-locked) host memory allows the GPU to DMA-copy data
    without blocking the CPU.
    """

    host: torch.Tensor
    device: torch.Tensor

    @classmethod
    def allocate(
        cls,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        extra: int = 0,
    ) -> "CpuGpuBuffer":
        """Allocate pinned host + GPU buffer pair.

        Args:
            size: Primary dimension (e.g. max_batch).
            dtype: Tensor dtype.
            device: GPU device.
            extra: If > 0, allocates ``[size + extra]`` shape for
                   multi-dimensional meta buffers (e.g. vocab-sized
                   trailing dim for disallow_mask).
        """
        shape = (size + extra,) if extra else (size,)
        host = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        dev = torch.empty(shape, dtype=dtype, device=device)
        return cls(host=host, device=dev)

    def copy_host_to_device(self, count: int, stream: torch.cuda.Stream) -> None:
        """Async copy first *count* elements host → device on *stream*."""
        self.device[:count].copy_(self.host[:count], non_blocking=True)

    def copy_device_to_host(self, count: int, stream: torch.cuda.Stream) -> None:
        """Async copy first *count* elements device → host on *stream*."""
        with torch.cuda.stream(stream):
            self.host[:count].copy_(self.device[:count], non_blocking=True)


@dataclass
class DecodeMetaBuffers:
    """Per-slot pinned host metadata for H2D copies.

    These hold the batch metadata that must be copied to GPU at the start
    of each decode step.  Each slot needs its own set so the CPU does not
    overwrite pinned source data while a previous step's H2D copy is
    still in flight.

    Fields
    ------
    batch_idx : CpuGpuBuffer  (int64 [max_batch])
        Maps each batch row index to its logical sequence id.
    input_pos : CpuGpuBuffer  (int32 [max_batch])
        Position in the sequence for positional encoding.
    disallow_mask : CpuGpuBuffer  (bool [max_batch, vocab])
        Per-sequence per-token mask: True → force logit to -∞ before
        sampling.  Used for constrained/structured decoding.
    """

    batch_idx: CpuGpuBuffer       # int64  [max_batch]
    input_pos: CpuGpuBuffer       # int32  [max_batch]
    disallow_mask: CpuGpuBuffer   # bool   [max_batch * vocab]

    @classmethod
    def allocate(
        cls,
        max_batch: int,
        vocab_size: int,
        device: torch.device,
    ) -> "DecodeMetaBuffers":
        return cls(
            batch_idx=CpuGpuBuffer.allocate(max_batch, torch.int64, device),
            input_pos=CpuGpuBuffer.allocate(max_batch, torch.int32, device),
            # Flattened [max_batch, vocab] → [max_batch * vocab].
            disallow_mask=CpuGpuBuffer.allocate(
                max_batch * vocab_size, torch.bool, device
            ),
        )


@dataclass
class DecodeSlot:
    """Bundled resources for one ping-pong decode slot.

    Attributes
    ----------
    slot_id : int
        0 or 1.
    meta : DecodeMetaBuffers
        Per-slot pinned host metadata buffers.
    compute_stream : torch.cuda.Stream
        Reference to the shared decode compute stream.
    fa3_page_table : torch.Tensor
        Per-slot page-table rows for paged KV cache, shape
        ``[max_batch, kv_cache_pages]`` (int32).
    fa3_seqused_k : torch.Tensor
        Per-slot per-sequence KV lengths, shape ``[max_batch]`` (int32).
    sampled_ids : torch.Tensor
        GPU buffer for sampled token IDs, shape ``[max_batch]`` (int64).
    sampled_logprobs : torch.Tensor
        GPU buffer for sampled log-probabilities, shape ``[max_batch]``
        (float32).
    logits : torch.Tensor
        Forward output logits, shape ``[max_batch, vocab]`` (dtype).
    hidden_last : torch.Tensor
        Last-hidden-state from forward, shape ``[max_batch, hidden_dim]``
        (dtype).  Used for structured-output (spatial) decoder kernels.
    decode_token_ids : torch.Tensor
        Decode input token ids, shape ``[max_batch]`` (int64).
    cuda_graphs : dict[int, torch.cuda.CUDAGraph] | None
        Captured HIP graphs keyed by batch size.
    step_done_event : torch.cuda.Event
        Recorded on compute_stream when forward+sample outputs are staged
        and ready for D2H copy.
    commit_done_event : torch.cuda.Event
        Recorded on copy_stream after CPU has finished reading from the
        pinned host buffer.
    mask_ready_event : torch.cuda.Event
        Recorded on copy_stream when the disallow-mask upload has
        completed and sampling can proceed.
    """

    slot_id: int
    meta: DecodeMetaBuffers
    compute_stream: torch.cuda.Stream

    # Paged-KV metadata
    fa3_page_table: torch.Tensor
    fa3_seqused_k: torch.Tensor

    # GPU staging for sampled outputs (per-slot to avoid clobber)
    sampled_ids: torch.Tensor        # [max_batch] int64
    sampled_logprobs: torch.Tensor   # [max_batch] float32

    # Forward output buffers (also used as graph output buffers)
    logits: torch.Tensor             # [max_batch, vocab] dtype
    hidden_last: torch.Tensor        # [max_batch, hidden_dim] dtype

    # Decode input staging (also used as graph input buffers)
    decode_token_ids: torch.Tensor   # [max_batch] int64

    # HIP graphs (captured using this slot's buffers)
    cuda_graphs: Optional[dict[int, torch.cuda.CUDAGraph]] = None

    # Pre-allocated events (non-timing, non-blocking for performance)
    step_done_event: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(
            enable_timing=False, blocking=False
        )
    )
    commit_done_event: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(
            enable_timing=False, blocking=False
        )
    )
    mask_ready_event: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(
            enable_timing=False, blocking=False
        )
    )

    def record_step_done(self, stream: torch.cuda.Stream) -> None:
        """Record that this slot's forward + sample outputs are ready."""
        self.step_done_event.record(stream)

    def record_commit_done(self, stream: torch.cuda.Stream) -> None:
        """Record that CPU has finished reading this slot's results."""
        self.commit_done_event.record(stream)


def create_decode_slot(
    slot_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch: int,
    kv_cache_pages: int,
    vocab_size: int,
    hidden_dim: int,
    compute_stream: torch.cuda.Stream,
) -> DecodeSlot:
    """Allocate all GPU and pinned-host resources for one decode slot.

    This should be called at engine initialisation time, not in the
    hot path.  All buffers are fixed-size for the lifetime of the
    engine and are reused across steps.
    """
    meta = DecodeMetaBuffers.allocate(max_batch, vocab_size, device)

    fa3_page_table = torch.empty(
        (max_batch, kv_cache_pages), dtype=torch.int32, device=device
    )
    fa3_seqused_k = torch.empty(
        (max_batch,), dtype=torch.int32, device=device
    )

    sampled_ids = torch.empty((max_batch,), dtype=torch.long, device=device)
    sampled_logprobs = torch.empty(
        (max_batch,), dtype=torch.float32, device=device
    )

    logits = torch.empty(
        (max_batch, vocab_size), dtype=dtype, device=device
    )
    hidden_last = torch.empty(
        (max_batch, hidden_dim), dtype=dtype, device=device
    )
    decode_token_ids = torch.empty(
        (max_batch,), dtype=torch.long, device=device
    )

    return DecodeSlot(
        slot_id=slot_id,
        meta=meta,
        compute_stream=compute_stream,
        fa3_page_table=fa3_page_table,
        fa3_seqused_k=fa3_seqused_k,
        sampled_ids=sampled_ids,
        sampled_logprobs=sampled_logprobs,
        logits=logits,
        hidden_last=hidden_last,
        decode_token_ids=decode_token_ids,
        cuda_graphs=None,
    )
