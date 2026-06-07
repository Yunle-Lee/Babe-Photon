"""HIP stream and event management for AMD GPUs.

Manages the two-stream model for Photon's pipelined decode:

  compute_stream — All GPU forwards (prefill + decode) serialise here.
    This preserves sequential token dependencies because every step's
    forward is enqueued on the same stream in order.

  copy_stream   — All device-to-host (D2H) copies of sampled outputs
    go here.  They wait on step_done_event (recorded on compute_stream
    when a step's staging buffers are ready), so copies never start
    before the forward that produced them.  Because the copy stream is
    independent of the compute stream, the next forward can start
    immediately — the GPU bubble is the time the copy would have taken
    on the compute stream.

Stream model (AMD ROCm):
  PyTorch's torch.cuda.Stream maps to HIP streams on AMD GPUs via the
  ROCm backend.  The semantics are identical to CUDA streams:
  - Work on the same stream is serialised in submission order.
  - Work on different streams may execute concurrently.
  - Events are stream-synchronisation primitives.

  Priority 0 is default; -1 gives higher priority (may reduce tail
  latency for copies).

Events are pre-allocated per slot (not per-step) to avoid runtime
allocation overhead, which can cause device-wide synchronisation on
AMD GPUs.
"""

from __future__ import annotations

import torch


class StreamManager:
    """Owns the two streams and pre-allocated per-slot events for the
    entire pipeline lifetime.

    Events are reused across steps — the event for slot *k* is recycled
    when slot *k* is used for a new step.
    """

    def __init__(self, device: torch.device):
        self._device = device

        # -- Streams -------------------------------------------------------
        # compute_stream: serialises all GPU forwards.
        # priority=0: default priority.
        self.compute_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=device, priority=0
        )

        # copy_stream: handles D2H copies in the background.
        # priority=-1: slight priority boost to reduce tail latency
        # for the D2H copy, ensuring the CPU sees results promptly.
        self.copy_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=device, priority=-1
        )

        # Per-slot events.  Two slots → two events of each type.
        # enable_timing=False to avoid profiling overhead on the hot path.
        # blocking=False for the same reason.
        self._step_done_events: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in range(2)
        ]
        self._commit_done_events: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in range(2)
        ]

    # -- Slot-scoped event access -------------------------------------------

    def step_done_event(self, slot_id: int) -> torch.cuda.Event:
        """Event signalled when slot *slot_id*'s forward+sample outputs
        are ready for D2H copy."""
        return self._step_done_events[slot_id]

    def commit_done_event(self, slot_id: int) -> torch.cuda.Event:
        """Event signalled when the CPU has finished reading slot
        *slot_id*'s D2H copy and the slot is safe to reuse."""
        return self._commit_done_events[slot_id]

    # -- Convenience synchronisation ---------------------------------------

    def sync_compute(self) -> None:
        """Block CPU until all compute-stream work finishes."""
        self.compute_stream.synchronize()

    def sync_copy(self) -> None:
        """Block CPU until all copy-stream work finishes."""
        self.copy_stream.synchronize()

    def sync_all(self) -> None:
        """Block CPU until all GPU work (both streams) finishes."""
        self.sync_compute()
        self.sync_copy()
